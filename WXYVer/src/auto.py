# preprocess.py
from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
import numpy as np

# ==== 可选：自动化特征工程（autofeat） ====
try:
    from autofeat import AutoFeatClassifier
    _HAS_AUTOFE = True
except Exception:
    AutoFeatClassifier = None
    _HAS_AUTOFE = False


def infer_column_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    """根据 pandas dtype 简单推断数值/类别列。必要时在 config 里显式指定更稳。"""
    features = df.drop(columns=[target])
    cat_cols = [c for c in features.columns if features[c].dtype == "object" or str(features[c].dtype) == "category"]
    num_cols = [c for c in features.columns if c not in cat_cols]
    return num_cols, cat_cols


# ========= 业务缺失指示 =============
def crq_missing_indicator(X):
    if isinstance(X, pd.DataFrame):
        s = X.iloc[:, 0]
    elif isinstance(X, pd.Series):
        s = X
    else:
        s = pd.Series(np.asarray(X).ravel())
    s2 = s.copy()
    mask_str = s2.apply(lambda v: isinstance(v, str))
    s2.loc[mask_str] = s2.loc[mask_str].str.strip()
    missing = s2.eq("").fillna(False) | s2.isna()
    return missing.astype(int).to_numpy().reshape(-1, 1)


# ========= 自动化特征工程封装（可被 sklearn clone & joblib 序列化） =========
from sklearn.base import BaseEstimator, TransformerMixin

class AutoFETransformer(BaseEstimator, TransformerMixin):
    """
    使用 autofeat 的 AutoFeatClassifier 自动生成/筛选特征。
    - 仅在 fit 时需要 y（监督型 FE）。
    - __init__ 不改写入参，满足 sklearn clone 规范。
    """
    def __init__(self, feateng_steps: int = 1, featsel_runs: int = 1,
                 max_time_secs: int | None = None, n_jobs: int = 1, verbose: int = 0):
        self.feateng_steps = feateng_steps
        self.featsel_runs = featsel_runs
        self.max_time_secs = max_time_secs
        self.n_jobs = n_jobs
        self.verbose = verbose
        # learned states
        self._af = None
        self.out_cols_ = None

    def fit(self, X, y=None):
        if not _HAS_AUTOFE:
            # 若未安装 autofeat，则作为“直通”变换（不做任何处理）
            self._af = None
            # 记一下输出列名（保持列数不变）
            n_out = X.shape[1] if hasattr(X, "shape") else len(X[0])
            self.out_cols_ = [f"auto_passthrough_{i}" for i in range(n_out)]
            return self

        # 保证是 DataFrame（autofeat 更依赖列名）
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(np.asarray(X), columns=[f"f{i}" for i in range(np.asarray(X).shape[1])])

        if y is None:
            raise ValueError("AutoFETransformer.fit 需要 y（监督型特征工程）。请在 preprocessor.fit(X, y) 里传入 y。")

        # 构建并拟合
        self._af = AutoFeatClassifier(
            feateng_steps=self.feateng_steps,
            featsel_runs=self.featsel_runs,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        X_new = self._af.fit_transform(X_df, pd.Series(y))
        # 记录输出列名
        self.out_cols_ = list(X_new.columns)
        return self

    def transform(self, X):
        if not _HAS_AUTOFE or self._af is None:
            # 未启用 autofeat，原样输出
            if isinstance(X, pd.DataFrame):
                return X.to_numpy(dtype=float, copy=False)
            return np.asarray(X, dtype=float)

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(np.asarray(X), columns=[f"f{i}" for i in range(np.asarray(X).shape[1])])

        X_new = self._af.transform(X_df)
        return X_new.to_numpy(dtype=float)


# ========= 组装预处理器 =========
def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    - 删除: AlleyAccess, ExteriorFinishType, RecreationQuality, MiscellaneousFeature
    - ConferenceRoomQuality 的“有业务含义的缺失”：
        * 类别支线: 缺失填 "__MISSING__" + One-Hot（缺失成为独立类别）
        * 额外分支: 生成 ConferenceRoomQuality_missing (0/1)
    - （可选）自动化特征工程分支：对数值列做 AutoFE（若已安装 autofeat）
    """
    policy_drop = ["AlleyAccess", "ExteriorFinishType", "RecreationQuality", "MiscellaneousFeature"]
    business_missing_col = "ConferenceRoomQuality"

    # --- 删列，维护列清单 ---
    num_cols = [c for c in num_cols if c not in policy_drop]
    cat_cols = [c for c in cat_cols if c not in policy_drop]

    # 确保 CRQ 在类别支线（如果既不在数值也不在类别，就追加到类别）
    if (business_missing_col not in num_cols) and (business_missing_col not in cat_cols):
        cat_cols.append(business_missing_col)

    crq_missing_tf = FunctionTransformer(crq_missing_indicator, validate=False)

    # --- 数值/类别支线 ---
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # --- 自动化特征工程支线（可选；仅对数值列） ---
    # 为了稳健，先做一次中位数填充再交给 AutoFE（防 NaN）
    auto_fe_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("autofe", AutoFETransformer(
            feateng_steps=1,      # 建议先从 1 步开始，快且稳
            featsel_runs=1,       # 选择 1 次特征选择，控制时长
            max_time_secs=None,   # 也可设置为总秒数上限，如 300
            n_jobs=1,
            verbose=0
        ))
    ])

    transformers = [
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
        ("crq_missing", crq_missing_tf, [business_missing_col]),  # 输出 0/1 指示
    ]

    # 若环境里装了 autofeat，就追加自动化 FE 分支；否则自动跳过
    if _HAS_AUTOFE and len(num_cols) > 0:
        transformers.append(("auto_fe", auto_fe_pipe, num_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    return preprocessor
