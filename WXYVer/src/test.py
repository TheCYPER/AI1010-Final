# src/infer.py
import joblib
import pandas as pd
from config import TrainConfig

def main(input_csv: str = None, cfg: TrainConfig = TrainConfig()):
    pipe = joblib.load(cfg.paths.pipeline_out)

    # 1) 读入需要预测的数据（无目标列）
    csv_path = input_csv or cfg.paths.test_csv
    X = pd.read_csv(csv_path)

    # 2) 直接预测
    y_pred = pipe.predict(X)
    y_proba = getattr(pipe.named_steps["model"], "predict_proba", None)
    proba = y_proba(pipe.named_steps["preprocess"].transform(X)) if y_proba else None

    # 3) 保存/打印
    out = pd.DataFrame()
    out["Id"] = range(len(y_pred))
    out["OfficeCategory"] = y_pred
    out.to_csv(cfg.paths.preds_out, index=False)
    print(f"Predictions saved -> {cfg.paths.preds_out}")

if __name__ == "__main__":
    main()
