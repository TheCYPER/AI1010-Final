from typing import Tuple, List, Sequence, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

def infer_column_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    """根据 pandas dtype 简单推断数值/类别列。必要时在 config 里显式指定更稳。"""
    features = df.drop(columns=[target])
    cat_cols = [c for c in features.columns if features[c].dtype == "object" or str(features[c].dtype) == "category"]
    num_cols = [c for c in features.columns if c not in cat_cols]
    return num_cols, cat_cols

# -------------------- 顶层：ConferenceRoomQuality 缺失指示 --------------------
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

# -------------------- 频率编码（无监督，fit 后映射） --------------------
# ⛳ 修正版：不要在 __init__ 里 list(cols)
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        # 关键：原样保存参数（不包 list/tuple，不改写）
        self.cols = cols
        # 学习状态用后缀 '_' 的属性
        self.maps_ = None

    def fit(self, X, y=None):
        n = len(X)
        self.maps_ = {}
        for c in self.cols:
            vc = X[c].value_counts(dropna=False)
            self.maps_[c] = vc / n
        return self

    def transform(self, X):
        outs = []
        for c in self.cols:
            m = self.maps_[c]
            vals = X[c].map(m).fillna(0.0).to_numpy().reshape(-1, 1)
            outs.append(vals)
        return np.hstack(outs)


# -------------------- K 折目标编码（防泄漏） --------------------
class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols, n_splits=5, shuffle=True, random_state=42):
        # 关键：原样保存所有入参
        self.cols = cols
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

        # 学习状态
        self.global_mean_ = None
        self.maps_ = None

    def fit(self, X, y):
        y = pd.Series(y).astype(float)
        self.global_mean_ = float(y.mean())
        self.maps_ = {}

        # 用“全量均值映射”供 transform 使用；折内泄漏需由外层 CV 控制
        for c in self.cols:
            self.maps_[c] = X.join(y.rename('__y__')).groupby(c)['__y__'].mean()
        return self

    def transform(self, X):
        outs = []
        for c in self.cols:
            m = self.maps_[c]
            vals = X[c].map(m).fillna(self.global_mean_).to_numpy().reshape(-1, 1)
            outs.append(vals)
        return np.hstack(outs)

# -------------------- 宽特征构建器（一次产出你列出的所有派生数值特征） --------------------
class WideFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    生成以下数值派生特征（有则用，无则跳过，缺失/异常置 NaN）：
    - 年龄：BuildingAge, YearsSinceRenovation, IsRenovated(0/1), RenovationAge
    - 面积：TotalLivingArea, TotalBasementArea（若无 BasementArea，则尝试已完工面积求和）
    - 比率：BasementFinishRatio, OfficeSpaceRatio, PlotCoverage, RoomDensity,
            ParkingPerArea, BasementUtilization
    - 质量组合：OverallQuality, ExteriorScore, BasementOverallScore
    - 组合：ProximityScore, HasParking(0/1)
    - 季节与年代：SeasonListed(1-4), ConstructionDecade
    - 分箱：PlotSize_binned(等宽 5 箱), BuildingAge_binned([0,10,25,50,100,200])
    """
    def __init__(self):
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        # 仅记录输入列，实际统计不需要
        return self

    def transform(self, X: pd.DataFrame):
        df = X.copy()
        out: Dict[str, Any] = {}

        def to_num(col):
            return pd.to_numeric(df[col], errors='coerce') if col in df.columns else pd.Series([np.nan]*len(df))

        # --- 年龄类 ---
        y_listed = to_num("YearListed")
        y_const  = to_num("ConstructionYear")
        y_reno   = to_num("RenovationYear")

        building_age = (y_listed - y_const)
        building_age = building_age.mask(building_age < 0, np.nan)
        out["BuildingAge"] = building_age

        years_since_reno = (y_listed - y_reno)
        years_since_reno = years_since_reno.mask(years_since_reno < 0, np.nan)
        out["YearsSinceRenovation"] = years_since_reno

        is_renov = (y_reno > y_const).astype(float)  # 0/1
        is_renov = is_renov.where(~(y_reno.isna() | y_const.isna()), np.nan)  # 缺失置 NaN
        out["IsRenovated"] = is_renov

        out["RenovationAge"] = years_since_reno  # 与上等价，单独命名方便下游

        # --- 面积 ---
        ground = to_num("GroundFloorArea")
        upper  = to_num("UpperFloorArea")
        basement_area = to_num("BasementArea")
        fin_b1 = to_num("FinishedBasementArea1")
        fin_b2 = to_num("FinishedBasementArea2")
        unfinished = to_num("UnfinishedBasementArea")

        total_living = ground + upper
        out["TotalLivingArea"] = total_living

        # TotalBasementArea：优先用 BasementArea；否则用已完工 + 未完工求和
        total_basement = basement_area.copy()
        fallback = fin_b1.fillna(0) + fin_b2.fillna(0) + unfinished.fillna(0)
        total_basement = total_basement.where(~total_basement.isna(), fallback.where(fallback>0, np.nan))
        out["TotalBasementArea"] = total_basement

        # --- 比率（除数<=0 置 NaN） ---
        def safe_ratio(num, den):
            return np.where((den > 0) & (~np.isnan(den)), num/den, np.nan)

        out["BasementFinishRatio"] = safe_ratio((fin_b1.fillna(0) + fin_b2.fillna(0)).to_numpy(),
                                                basement_area.to_numpy())
        office = to_num("OfficeSpace")
        out["OfficeSpaceRatio"] = safe_ratio(office.to_numpy(),
                                             (total_living.to_numpy()))
        plot_size = to_num("PlotSize")
        out["PlotCoverage"] = safe_ratio(total_living.to_numpy(), plot_size.to_numpy())

        total_rooms = to_num("TotalRooms")
        out["RoomDensity"] = safe_ratio(total_rooms.to_numpy(), total_living.to_numpy())

        parking = to_num("ParkingSpots")
        out["ParkingPerArea"] = safe_ratio(parking.to_numpy(), total_living.to_numpy())
        out["HasParking"] = np.where(parking.fillna(0) > 0, 1.0, 0.0)

        out["BasementUtilization"] = safe_ratio((fin_b1.fillna(0) + fin_b2.fillna(0)).to_numpy(),
                                                (basement_area.fillna(0).to_numpy() + 1))  # +1 防 0

        # --- 质量组合 ---
        def prod(a, b, c=None):
            x = to_num(a) * to_num(b) if c is None else to_num(a) * to_num(b) * to_num(c)
            return x.replace([np.inf, -np.inf], np.nan)
        out["OverallQuality"] = prod("BuildingGrade", "BuildingCondition")
        out["ExteriorScore"]  = prod("ExteriorQuality", "ExteriorCondition")
        out["BasementOverallScore"] = prod("BasementQuality", "BasementCondition", "BasementExposure")

        # --- 组合/位置/季节/年代 ---
        prox1 = to_num("Proximity1")
        prox2 = to_num("Proximity2")
        out["ProximityScore"] = prox1.add(prox2, fill_value=np.nan)

        month = to_num("MonthListed")
        out["SeasonListed"] = ((month % 12) // 3 + 1).where(~month.isna(), np.nan)

        out["ConstructionDecade"] = (y_const // 10 * 10).where(~y_const.isna(), np.nan)

        # --- 分箱 ---
        # PlotSize 等宽 5 箱
        if "PlotSize" in df.columns:
            try:
                out["PlotSize_binned"] = pd.cut(plot_size, bins=5, labels=False).astype("float")
            except Exception:
                out["PlotSize_binned"] = np.nan
        else:
            out["PlotSize_binned"] = np.nan

        # BuildingAge 自定义分箱
        try:
            out["BuildingAge_binned"] = pd.cut(out["BuildingAge"],
                                               bins=[0,10,25,50,100,200],
                                               labels=False, include_lowest=True).astype("float")
        except Exception:
            out["BuildingAge_binned"] = np.nan

        # 汇总输出
        out_df = pd.DataFrame(out)
        self.feature_names_ = list(out_df.columns)
        return out_df.to_numpy(dtype=float)

# -------------------- 组装 ColumnTransformer --------------------
def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    - 删除: AlleyAccess, ExteriorFinishType, RecreationQuality, MiscellaneousFeature
    - ConferenceRoomQuality 的“有业务含义缺失”：
        * 类别支线: 缺失填 "__MISSING__" + One-Hot
        * 分支: 生成 ConferenceRoomQuality_missing (0/1)
    - 新增宽特征分支：一次性产出你列出的所有派生特征（数值）
    - 可选：频率编码 / 目标编码（KFold 防泄漏）
    """
    policy_drop = ["AlleyAccess", "ExteriorFinishType", "RecreationQuality", "MiscellaneousFeature"]
    business_missing_col = "ConferenceRoomQuality"

    # 列清单清理
    num_cols = [c for c in num_cols if c not in policy_drop]
    cat_cols = [c for c in cat_cols if c not in policy_drop]

    # 确保 CRQ 走类别分支
    if (business_missing_col not in num_cols) and (business_missing_col not in cat_cols):
        cat_cols.append(business_missing_col)

    # 频率编码列（可按需调整/扩展）
    freq_cols = ['RoofType', 'ExteriorCovering1', 'FoundationType']
    freq_cols = [c for c in freq_cols if c in (num_cols + cat_cols)]

    # 目标编码列（高基数类别，谨慎使用；需要在 Pipeline.fit(X, y) 时传 y）
    te_cols = ['ZoningClassification', 'BuildingType', 'BuildingStyle', 'HeatingType']
    te_cols = [c for c in te_cols if c in (num_cols + cat_cols)]

    # WideFeatureBuilder 需要的输入列集合（存在则用）
    wide_needed = [
        "YearListed","ConstructionYear","RenovationYear",
        "GroundFloorArea","UpperFloorArea","BasementArea",
        "FinishedBasementArea1","FinishedBasementArea2","UnfinishedBasementArea",
        "OfficeSpace","PlotSize","TotalRooms","ParkingSpots",
        "BuildingGrade","BuildingCondition","ExteriorQuality","ExteriorCondition",
        "BasementQuality","BasementCondition","BasementExposure",
        "Proximity1","Proximity2","MonthListed"
    ]
    wide_input_cols = [c for c in wide_needed if c in (num_cols + cat_cols)]

    # 分支：数值/类别
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # 分支：ConferenceRoomQuality 缺失指示
    crq_missing_tf = FunctionTransformer(crq_missing_indicator, validate=False)

    # 分支：宽特征（你的 FE 汇总）-> 中位数填充
    wide_pipe = Pipeline(steps=[
        ("builder", WideFeatureBuilder()),
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # 分支：频率编码（可选，无监督）
    if freq_cols:
        freq_pipe = Pipeline(steps=[
            ("freq", FrequencyEncoder(freq_cols))
        ])

    # 分支：目标编码（可选，KFold 防泄漏；需要 y）
    if te_cols:
        te_pipe = Pipeline(steps=[
            ("te", KFoldTargetEncoder(te_cols, n_splits=5, shuffle=True, random_state=42))
        ])

    transformers = [
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
        ("crq_missing", crq_missing_tf, [business_missing_col]),
    ]

    if wide_input_cols:
        transformers.append(("wide_feats", wide_pipe, wide_input_cols))
    if freq_cols:
        transformers.append(("freq_enc", freq_pipe, freq_cols))
    if te_cols:
        transformers.append(("target_enc", te_pipe, te_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    return preprocessor
