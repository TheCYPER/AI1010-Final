import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from config import TrainConfig
from preprocess import infer_column_types, build_preprocessor

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
    X_tr_t = preprocessor.fit_transform(X_tr,y_tr)
    X_va_t = preprocessor.transform(X_va)

    # 5) 模型（多分类）
    model = XGBClassifier(
        objective=cfg.xgb.objective,
        num_class=num_class,
        n_estimators=cfg.xgb.n_estimators,
        learning_rate=cfg.xgb.learning_rate,
        max_depth=cfg.xgb.max_depth,
        subsample=cfg.xgb.subsample,
        colsample_bytree=cfg.xgb.colsample_bytree,
        reg_lambda=cfg.xgb.reg_lambda,
        reg_alpha=cfg.xgb.reg_alpha,
        random_state=cfg.xgb.random_state,
        eval_metric=cfg.xgb.eval_metric
    )

    # 6) 训练（早停）
    model.fit(
        X_tr_t, y_tr,
        eval_set=[(X_va_t, y_va)],
        #early_stopping_rounds=cfg.early_stopping_rounds,
        verbose=False
    )

    # 7) 验证评估
    y_pred = model.predict(X_va_t)
    acc = accuracy_score(y_va, y_pred)
    report = classification_report(y_va, y_pred, output_dict=True)
    print(f"[val] accuracy={acc:.4f}")

    # 8) 组装一个“端到端” pipeline（预处理 + 已训练模型）
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    # 9) 保存产物
    joblib.dump(pipeline, cfg.paths.pipeline_out)
    with open(cfg.paths.metrics_out, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "report": report, "classes": list(map(str, classes))}, f, ensure_ascii=False, indent=2)

    print(f"Saved pipeline -> {cfg.paths.pipeline_out}")
    print(f"Saved metrics  -> {cfg.paths.metrics_out}")

if __name__ == "__main__":
    main()
