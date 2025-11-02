from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
import numpy as np
from typing import List


def infer_column_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    """根据 pandas dtype 简单推断数值/类别列。必要时在 config 里显式指定更稳。"""
    features = df.drop(columns=[target])
    cat_cols = [c for c in features.columns if features[c].dtype == "object" or str(features[c].dtype) == "category"]
    num_cols = [c for c in features.columns if c not in cat_cols]
    return num_cols, cat_cols

'''
def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        # 树模型不需要标准化；需要的话可加 StandardScaler()
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )
    return preprocessor
'''


#2
# 缺失指示列的 transform（鲁棒到 object/string/None/空串）
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

def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    - 删除: AlleyAccess, ExteriorFinishType, RecreationQuality, MiscellaneousFeature
    - 对 ConferenceRoomQuality 做“有业务含义的缺失”处理：
        * 类别支线: 缺失填 "__MISSING__" + One-Hot（缺失成为独立类别）
        * 额外分支: 生成 ConferenceRoomQuality_missing (0/1) 作为数值特征
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
            ("crq_missing", crq_missing_tf, [business_missing_col]),  # 输出 0/1 指示
        ],
        remainder="drop"
    )
    return preprocessor


