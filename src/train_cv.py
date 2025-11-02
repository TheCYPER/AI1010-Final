# src/train_cv.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier
#from xgboost.callback import EarlyStopping

from config import TrainConfig
from preprocess import infer_column_types, build_preprocessor


# ===== 可调超参 / 行为开关 =====
N_SPLITS = 2
SHUFFLE = True
RANDOM_STATE = 42

USE_CLASS_WEIGHTS = True          # 类别加权
CLASS_WEIGHT_POWER = 1.0          # 权重指数（>1放大不平衡，=1为反频率，<1温和一点）

USE_EARLY_STOPPING = True         # 早停
EARLY_STOPPING_ROUNDS = 80        # 早停耐心轮次

SAVE_PER_FOLD_PIPELINE = False    # 是否保存每折训练好的 pipeline（可能较大）


def make_outdir(base_dir: str | Path = "models/cv") -> Path:
    outdir = Path(base_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def compute_sample_weight(y: pd.Series, power: float = 1.0) -> np.ndarray:
    """
    生成按反频率的样本权重，并归一到均值≈1，避免损失尺度变化太大。
    w_c = (N / count_c) ** power
    """
    cls, cnt = np.unique(y, return_counts=True)
    total = len(y)
    wmap = {c: (total / cnt[i]) ** power for i, c in enumerate(cls)}
    w = y.map(wmap).astype(float).values
    w = w / np.mean(w)  # 归一化
    return w


def agg_reports_mean_std(acc_list: list[float]) -> Dict[str, float]:
    return {
        "accuracy_mean": float(np.mean(acc_list)),
        "accuracy_std": float(np.std(acc_list))
    }


def combine_classification_reports(reports: list[Dict[str, Any]]) -> Dict[str, Any]:
    """
    将多折的 classification_report（output_dict=True）按 support 加权合并。
    返回结构类似 sklearn 的 report（含 accuracy/weighted avg/macro avg 近似聚合）。
    """
    # 收集所有标签键（'0','1',...）
    label_keys = sorted([k for k in reports[0].keys() if k.isdigit()], key=lambda x: int(x))
    combined = {}
    total_support = {k: 0.0 for k in label_keys}

    # 先累加每类的 (precision, recall, f1, support)
    sums = {k: {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0.0} for k in label_keys}
    accs = []

    for rpt in reports:
        # accuracy
        accs.append(rpt.get("accuracy", np.nan))
        for k in label_keys:
            s = float(rpt[k]["support"])
            total_support[k] += s
            sums[k]["precision"] += float(rpt[k]["precision"]) * s
            sums[k]["recall"]    += float(rpt[k]["recall"]) * s
            sums[k]["f1-score"]  += float(rpt[k]["f1-score"]) * s
            sums[k]["support"]   += s

    # 计算每类加权平均
    for k in label_keys:
        s = sums[k]["support"]
        if s > 0:
            combined[k] = {
                "precision": sums[k]["precision"] / s,
                "recall":    sums[k]["recall"]    / s,
                "f1-score":  sums[k]["f1-score"]  / s,
                "support":   s
            }
        else:
            combined[k] = {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0.0}

    # 计算 weighted avg / macro avg
    total = sum(total_support.values())
    if total > 0:
        w_precision = sum(combined[k]["precision"] * combined[k]["support"] for k in label_keys) / total
        w_recall    = sum(combined[k]["recall"]    * combined[k]["support"] for k in label_keys) / total
        w_f1        = sum(combined[k]["f1-score"]  * combined[k]["support"] for k in label_keys) / total
    else:
        w_precision = w_recall = w_f1 = 0.0

    m_precision = np.mean([combined[k]["precision"] for k in label_keys])
    m_recall    = np.mean([combined[k]["recall"]    for k in label_keys])
    m_f1        = np.mean([combined[k]["f1-score"]  for k in label_keys])

    combined["weighted avg"] = {"precision": w_precision, "recall": w_recall, "f1-score": w_f1, "support": total}
    combined["macro avg"]    = {"precision": m_precision, "recall": m_recall, "f1-score": m_f1, "support": total}
    combined["accuracy"]     = float(np.nanmean(accs))  # 简单取均值

    return combined


def build_xgb(cfg: TrainConfig, num_class: int) -> XGBClassifier:
    return XGBClassifier(
        objective=cfg.xgb.objective,
        num_class=num_class,
        n_estimators=cfg.xgb.n_estimators,
        learning_rate=cfg.xgb.learning_rate,
        max_depth=cfg.xgb.max_depth,
        subsample=cfg.xgb.subsample,
        colsample_bytree=cfg.xgb.colsample_bytree,
        reg_lambda=cfg.xgb.reg_lambda,
        gamma = cfg.xgb.gamma,
        min_child_weight = cfg.xgb.min_child_weight,
        reg_alpha=getattr(cfg.xgb, "reg_alpha", 0.0),
        random_state=cfg.xgb.random_state,
        eval_metric=cfg.xgb.eval_metric
    )


def run_fold(
    fold_idx: int,
    X_tr: pd.DataFrame, y_tr: pd.Series,
    X_va: pd.DataFrame, y_va: pd.Series,
    num_cols: list[str], cat_cols: list[str],
    cfg: TrainConfig,
    outdir: Path
) -> Tuple[float, Dict[str, Any]]:
    """
    训练单折，返回 (accuracy, report_dict)。
    同时将本折的 metrics JSON 落盘。
    """
    # 预处理（fit 在本折训练集；若内部有目标编码需要 y_tr）
    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_tr_t = preprocessor.fit_transform(X_tr, y_tr)
    X_va_t = preprocessor.transform(X_va)

    # 类别加权（基于本折训练集标签）
    sample_weight = None
    if USE_CLASS_WEIGHTS:
        sample_weight = compute_sample_weight(y_tr, power=CLASS_WEIGHT_POWER)

    # 模型
    classes = sorted(pd.unique(y_tr))
    num_class = len(classes)
    model = build_xgb(cfg, num_class)


    # 训练（带早停）
    fit_kwargs = dict(
        X=X_tr_t, y=y_tr,
        eval_set=[(X_va_t, y_va)],
        verbose=False
    )
    '''
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    if USE_EARLY_STOPPING:
        fit_kwargs["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    '''
    model.fit(**fit_kwargs)

    # === 新增：训练集精度 ===
    y_tr_pred = model.predict(X_tr_t)
    train_acc = accuracy_score(y_tr, y_tr_pred)
    print(f"[fold {fold_idx}] train_acc={train_acc:.4f}")
    # 评估
    y_pred = model.predict(X_va_t)
    acc = accuracy_score(y_va, y_pred)
    report = classification_report(y_va, y_pred, output_dict=True)

    # 保存当折 pipeline（可选）
    if SAVE_PER_FOLD_PIPELINE:
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        joblib.dump(pipe, outdir / f"pipeline_fold{fold_idx}.joblib")

    # 保存当折 metrics
    per_fold_metrics = {
        "fold": fold_idx,
        "train_accuracy": train_acc,
        "accuracy": acc,
        "report": report,
        "num_samples_train": int(len(y_tr)),
        "num_samples_valid": int(len(y_va)),
        "use_class_weights": USE_CLASS_WEIGHTS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS if USE_EARLY_STOPPING else None
    }
    with open(outdir / f"metrics_fold{fold_idx}.json", "w", encoding="utf-8") as f:
        json.dump(per_fold_metrics, f, ensure_ascii=False, indent=2)

    print(f"[fold {fold_idx}] acc={acc:.4f}")
    return acc, report


def main(cfg: TrainConfig = TrainConfig()):
    # 输出目录
    outdir = make_outdir("models/cv")

    # 读取数据
    df = pd.read_csv(cfg.paths.train_csv)
    target = cfg.cols.target

    # 列类型
    if cfg.cols.numeric is None or cfg.cols.categorical is None:
        num_cols, cat_cols = infer_column_types(df, target)
    else:
        num_cols, cat_cols = cfg.cols.numeric, cfg.cols.categorical

    X = df.drop(columns=[target])
    y = df[target]

    # 5 折分层
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=SHUFFLE, random_state=RANDOM_STATE)

    acc_list = []
    rpt_list = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr, y_va = y.iloc[tr_idx].copy(), y.iloc[va_idx].copy()

        acc, rpt = run_fold(
            fold_idx, X_tr, y_tr, X_va, y_va,
            num_cols, cat_cols, cfg, outdir
        )
        acc_list.append(acc)
        rpt_list.append(rpt)

    # 汇总
    summary = agg_reports_mean_std(acc_list)
    rpt_combined = combine_classification_reports(rpt_list)

    overall = {
        "n_splits": N_SPLITS,
        "shuffle": SHUFFLE,
        "random_state": RANDOM_STATE,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "class_weight_power": CLASS_WEIGHT_POWER,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS if USE_EARLY_STOPPING else None,
        "accuracy_per_fold": [float(a) for a in acc_list],
        **summary,
        "report_combined": rpt_combined
    }

    with open(outdir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    print("\n=== CV Summary ===")
    print(f"acc (mean±std): {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"results saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
