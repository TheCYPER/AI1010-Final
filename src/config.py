from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Paths:
    train_csv: str = "datas/office_train.csv"
    test_csv: str  = "datas/office_test.csv"
    pipeline_out: str = "models/xgb_multiclass_pipeline.joblib"
    metrics_out: str  = "models/metrics.json"
    preds_out: str    = "models/predictions.csv"

@dataclass
class Columns:
    target: str = "OfficeCategory"
    numeric: Optional[List[str]] = None
    categorical: Optional[List[str]] = None

@dataclass
class XGBParams:
    objective: str = "multi:softprob"
    n_estimators: int = 1500
    learning_rate: float = 0.06
    gamma = 1.0
    min_child_weight = 10
    max_depth: int = 4
    subsample: float = 0.75
    colsample_bytree: float = 0.55
    reg_lambda: float = 10.0
    reg_alpha: float = 3
    random_state: int = 42
    eval_metric: str = "mlogloss"

@dataclass
class TrainConfig:
    paths: Paths = field(default_factory=Paths)
    cols: Columns = field(default_factory=Columns)
    xgb: XGBParams = field(default_factory=XGBParams)
    test_size: float = 0.2
    #early_stopping_rounds: int = 50
