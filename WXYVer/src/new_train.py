# src/train.py
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectFromModel, RFE
from xgboost import XGBClassifier

from config import TrainConfig
from preprocess import infer_column_types, build_preprocessor

# ---- 可调开关：特征筛选配置 ----
FS_METHOD = "importance"   # 可选: "importance" 或 "rfe" 或 None
FS_THRESHOLD = "mean"    # importance 模式阈值: "mean" / "median" / float
RFE_NUM_FEATURES = 50      # rfe 模式：保留的特征数
RFE_STEP = 0.2             # rfe 每轮剔除比例(0~1]

DO_SHAP_DIAG = False       # 需要 SHAP 诊断时设 True（可耗时）

def main(cfg: TrainConfig = TrainConfig()):
    # 1) 读取数据
    df = pd.read_csv(cfg.paths.train_csv)
    target = cfg.cols.target

    # 2) 列类型
    if cfg.cols.numeric is None or cfg.cols.categorical is None:
        num_cols, cat_cols = infer_column_types(df, target)
    else:
        num_cols, cat_cols = cfg.cols.numeric, cfg.cols.categorical

    X = df.drop(columns=[target])
    y = df[target]
    classes = sorted(pd.unique(y))
    num_class = len(classes)

    # 3) 划分数据（分层抽样）
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=cfg.test_size, random_state=42, stratify=y
    )

    # 4) 预处理器：fit 在训练集
    preprocessor = build_preprocessor(num_cols, cat_cols)
    # 若预处理里包含目标编码，务必传 y_tr
    X_tr_t = preprocessor.fit_transform(X_tr, y_tr)
    X_va_t = preprocessor.transform(X_va)

    # 5) 基线模型（用于特征筛选/或直接最终模型）
    base_params = dict(
        objective=cfg.xgb.objective,
        num_class=num_class,
        n_estimators=max(300, cfg.xgb.n_estimators // 2),  #筛选阶段用较轻配置更快
        learning_rate=max(0.05, cfg.xgb.learning_rate),
        max_depth=min(cfg.xgb.max_depth, 6),
        subsample=cfg.xgb.subsample,
        colsample_bytree=cfg.xgb.colsample_bytree,
        reg_lambda=max(1.0, cfg.xgb.reg_lambda),
        reg_alpha=getattr(cfg.xgb, "reg_alpha", 0.0),
        random_state=cfg.xgb.random_state,
        eval_metric=cfg.xgb.eval_metric
    )
    base_model = XGBClassifier(**base_params)

    # 6) 训练基线（为 importance/RFE 提供拟合）
    base_model.fit(X_tr_t, y_tr, eval_set=[(X_va_t, y_va)], verbose=False)

    # 7) 特征重要性与（可选）SHAP 诊断
    try:
        import shap
        if DO_SHAP_DIAG:
            explainer = shap.TreeExplainer(base_model)
            # 对多分类，shap_values 可能是 list；这里只做一次静态计算不落盘
            _ = explainer.shap_values(X_va_t[:256])
    except Exception:
        pass  # 没装 shap 或异常则跳过

    # 8) 特征筛选（可选）
    selector = None
    if FS_METHOD == "importance":
        # 用已拟合的 base_model 做嵌入式筛选
        selector = SelectFromModel(base_model, threshold=FS_THRESHOLD, prefit=True)
        X_tr_sel = selector.transform(X_tr_t)
        X_va_sel = selector.transform(X_va_t)
    elif FS_METHOD == "rfe":
        # RFE 会在内部重新拟合一个克隆模型（建议较轻配置）
        rfe_est = clone(base_model)
        selector = RFE(estimator=rfe_est, n_features_to_select=RFE_NUM_FEATURES, step=RFE_STEP)
        selector.fit(X_tr_t, y_tr)
        X_tr_sel = selector.transform(X_tr_t)
        X_va_sel = selector.transform(X_va_t)
    else:
        # 不筛选
        X_tr_sel, X_va_sel = X_tr_t, X_va_t

    # 9) 最终模型（更强配置）
    final_model = XGBClassifier(
        objective=cfg.xgb.objective,
        num_class=num_class,
        n_estimators=cfg.xgb.n_estimators,
        learning_rate=cfg.xgb.learning_rate,
        max_depth=cfg.xgb.max_depth,
        subsample=cfg.xgb.subsample,
        colsample_bytree=cfg.xgb.colsample_bytree,
        reg_lambda=cfg.xgb.reg_lambda,
        reg_alpha=getattr(cfg.xgb, "reg_alpha", 0.0),
        random_state=cfg.xgb.random_state,
        eval_metric=cfg.xgb.eval_metric
    )

    # 10) 训练最终模型（如需早停，按你本地 XGBoost 版本启用）
    final_model.fit(
        X_tr_sel, y_tr,
        eval_set=[(X_va_sel, y_va)],
        verbose=False
        # callbacks=...  # 若 3.1 版本需要 callbacks 方式再加
    )

    # 11) 验证评估
    y_pred = final_model.predict(X_va_sel)
    acc = accuracy_score(y_va, y_pred)
    report = classification_report(y_va, y_pred, output_dict=True)
    print(f"[val] accuracy={acc:.4f}")
    print(f"[fs] method={FS_METHOD}, input_dim={X_tr_t.shape[1]}, selected_dim={X_tr_sel.shape[1]}")

    # 12) 组装端到端 pipeline（预处理 + 选择器 + 已训练模型）
    steps = [("preprocess", preprocessor)]
    if selector is not None:
        steps.append(("feature_select", selector))  # 直接放入已拟合选择器
    steps.append(("model", final_model))
    pipeline = Pipeline(steps=steps)

    # 13) 保存产物
    joblib.dump(pipeline, cfg.paths.pipeline_out)
    with open(cfg.paths.metrics_out, "w", encoding="utf-8") as f:
        json.dump(
            {"accuracy": acc, "report": report, "classes": list(map(str, classes)),
             "fs_method": FS_METHOD,
             "in_dim": int(X_tr_t.shape[1]),
             "sel_dim": int(X_tr_sel.shape[1])},
            f, ensure_ascii=False, indent=2
        )

    print(f"Saved pipeline -> {cfg.paths.pipeline_out}")
    print(f"Saved metrics  -> {cfg.paths.metrics_out}")

if __name__ == "__main__":
    main()
