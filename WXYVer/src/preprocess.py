# preprocess.py
from typing import Tuple, List, Dict, Any, Sequence
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ------------------------------------------------------------
# 列类型推断
# ------------------------------------------------------------
def infer_column_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    """根据 pandas dtype 简单推断数值/类别列。必要时在 config 里显式指定更稳。"""
    features = df.drop(columns=[target])
    cat_cols = [c for c in features.columns if features[c].dtype == "object" or str(features[c].dtype) == "category"]
    num_cols = [c for c in features.columns if c not in cat_cols]
    return num_cols, cat_cols


# ------------------------------------------------------------
# ConferenceRoomQuality 缺失指示（鲁棒到 object/string/None/空串）
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# 频率编码（无监督）
# ------------------------------------------------------------
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    将类别取值映射为其在训练集中出现的频率。
    注意：fit 时仅使用训练集；transform 中未见类别频率置 0。
    """
    def __init__(self, cols: List[str]):
        self.cols = cols
        self.maps_: Dict[str, pd.Series] | None = None
        self.feature_names_out_: List[str] = []
        self._transform_output = None  # sklearn set_output 兼容

    def set_output(self, *, transform=None):
        self._transform_output = transform
        return self

    def fit(self, X: pd.DataFrame, y=None):
        n = len(X)
        self.maps_ = {}
        for c in self.cols:
            vc = X[c].value_counts(dropna=False)
            self.maps_[c] = (vc / n)
        self.feature_names_out_ = [f"FE__{c}" for c in self.cols]
        return self

    def transform(self, X: pd.DataFrame):
        outs = []
        for c in self.cols:
            m = self.maps_[c]
            vals = X[c].map(m).fillna(0.0).to_numpy().reshape(-1, 1)
            outs.append(vals)
        return np.hstack(outs)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_ or [])


# ------------------------------------------------------------
# 多分类目标编码（拉普拉斯平滑）
# 输出：对每个列 c，拼接 K 维 [P(y=k|c=x)]，未见类别回退到全局先验
# ------------------------------------------------------------
class MultiClassTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols: List[str], alpha: float = 10.0):
        self.cols = cols
        self.alpha = alpha
        self.classes_: np.ndarray | None = None
        self.prior_: np.ndarray | None = None
        self.maps_: Dict[str, Dict[Any, np.ndarray]] | None = None
        self.feature_names_out_: List[str] = []
        self._transform_output = None  # sklearn set_output 兼容

    def set_output(self, *, transform=None):
        self._transform_output = transform
        return self

    def fit(self, X: pd.DataFrame, y):
        y = pd.Series(y).astype(int)
        self.classes_ = np.sort(y.unique())
        K = len(self.classes_)

        counts = y.value_counts().reindex(self.classes_, fill_value=0).to_numpy()
        self.prior_ = (counts / counts.sum()).astype(float)

        self.maps_ = {}
        for c in self.cols:
            dfc = pd.DataFrame({c: X[c].astype("category"), "__y__": y})
            total = dfc.groupby(c)["__y__"].count()
            by_cls = {k: dfc[dfc["__y__"] == k].groupby(c)["__y__"].count()
                      for k in self.classes_}

            mapping: Dict[Any, np.ndarray] = {}
            for cat, tot in total.items():
                post = np.zeros(K, dtype=float)
                for i, k in enumerate(self.classes_):
                    cnt_k = float(by_cls[k].get(cat, 0.0))
                    post[i] = (cnt_k + self.alpha * self.prior_[i]) / (tot + self.alpha)
                mapping[cat] = post
            self.maps_[c] = mapping

        outs = []
        for c in self.cols:
            outs += [f"TE__{c}__p{int(k)}" for k in self.classes_]
        self.feature_names_out_ = outs
        return self

    def transform(self, X: pd.DataFrame):
        n = len(X)
        outs = []
        for c in self.cols:
            m = self.maps_[c]
            # 默认 prior
            M = np.tile(self.prior_, (n, len(self.classes_)))
            if c in X.columns:
                rows = []
                col_vals = X[c].astype("category")
                for val in col_vals:
                    rows.append(m.get(val, self.prior_))
                M = np.vstack(rows).astype(float)
            outs.append(M)
        return np.hstack(outs)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_ or [])


# ------------------------------------------------------------
# 宽特征构建器（一次产出派生数值特征 + 高级时间/交互/领域知识）
# ------------------------------------------------------------
class WideFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    生成以下派生特征（有列则用，无则跳过；非法/负值置 NaN）：
    - 年龄：BuildingAge, YearsSinceRenovation, IsRenovated(0/1), RenovationAge
    - 面积：TotalLivingArea, TotalBasementArea（缺乏 BasementArea 时用完工+未完工求和）
    - 比率：BasementFinishRatio, OfficeSpaceRatio, PlotCoverage, RoomDensity
            ParkingPerArea, BasementUtilization
    - 质量组合：OverallQuality, ExteriorScore, BasementOverallScore
    - 组合：ProximityScore, HasParking(0/1)
    - 季节/年代：SeasonListed(1-4), ConstructionDecade
    - 分箱：PlotSize_binned(等宽 5)、BuildingAge_binned([0,10,25,50,100,200])
    - ✅ 高级时间特征：BuildingLifeStage / RenovationEffectiveness / SeasonalStrength
    - ✅ 交互特征：QualityAreaProximity / BasementEfficiency / ValueDensity
    - ✅ 领域知识：RoomSizeAdequacy / ParkingAdequacy / RenovationNeed / LandUtilization
    """
    def __init__(self):
        self.feature_names_: List[str] = []
        self._transform_output = None

    def set_output(self, *, transform=None):
        self._transform_output = transform
        return self

    # ---------- helpers ----------
    @staticmethod
    def _to_num(df: pd.DataFrame, col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors='coerce') if col in df.columns else pd.Series([np.nan]*len(df))

    @staticmethod
    def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        den = np.asarray(den, dtype=float)
        num = np.asarray(num, dtype=float)
        return np.where((den > 0) & (~np.isnan(den)), num / den, np.nan)

    # ---------- 你提供的三组方法，稍作健壮性调整 ----------
    def add_advanced_temporal_features(self, df: pd.DataFrame, out: Dict[str, Any]) -> Dict[str, Any]:
        temporal: Dict[str, Any] = {}

        building_age = pd.to_numeric(pd.Series(out.get("BuildingAge")), errors="coerce")
        reno_age = pd.to_numeric(pd.Series(out.get("RenovationAge")), errors="coerce")

        # 建筑生命周期阶段 1-5
        temporal["BuildingLifeStage"] = np.select(
            [
                building_age <= 5,
                building_age <= 15,
                building_age <= 30,
                building_age <= 50,
                building_age > 50
            ],
            [1, 2, 3, 4, 5],
            default=3
        ).astype(float)

        # 翻新效果衰减
        temporal["RenovationEffectiveness"] = np.exp(-reno_age / 20.0)

        # 季节性强度（离 6 月越远越强）
        month = self._to_num(df, "MonthListed")
        temporal["SeasonalStrength"] = np.abs(6 - np.abs(month - 6)) / 6.0

        return temporal

    def add_advanced_interactions(self, df: pd.DataFrame, out: Dict[str, Any]) -> Dict[str, Any]:
        inter: Dict[str, Any] = {}
        quality = self._to_num(df, "BuildingGrade")
        area = pd.to_numeric(pd.Series(out.get("TotalLivingArea")), errors="coerce")
        proximity = pd.to_numeric(pd.Series(out.get("ProximityScore")), errors="coerce")

        inter["QualityAreaProximity"] = quality * np.log1p(area) * (1.0 / (1.0 + np.where(np.isnan(proximity), 0.0, proximity)))

        basement_finished = (self._to_num(df, "FinishedBasementArea1").fillna(0) +
                             self._to_num(df, "FinishedBasementArea2").fillna(0))
        basement_total = self._to_num(df, "BasementArea")
        basement_quality = self._to_num(df, "BasementQuality")

        inter["BasementEfficiency"] = (
            self._safe_ratio(basement_finished.to_numpy(), basement_total.to_numpy()) *
            basement_quality *
            self._to_num(df, "BasementExposure")
        )

        inter["ValueDensity"] = (
            quality * self._to_num(df, "BuildingCondition") *
            np.log1p(area) / np.log1p(self._to_num(df, "PlotSize"))
        )

        return inter

    def add_domain_knowledge_features(self, df: pd.DataFrame, out: Dict[str, Any]) -> Dict[str, Any]:
        dom: Dict[str, Any] = {}
        living_area = pd.to_numeric(pd.Series(out.get("TotalLivingArea")), errors="coerce")
        total_rooms = self._to_num(df, "TotalRooms")

        dom["RoomSizeAdequacy"] = self._safe_ratio(living_area.to_numpy(), (total_rooms + 1e-6).to_numpy())

        parking_spots = self._to_num(df, "ParkingSpots")
        dom["ParkingAdequacy"] = self._safe_ratio(parking_spots.to_numpy(), (total_rooms / 3.0 + 1e-6).to_numpy())

        building_age = pd.to_numeric(pd.Series(out.get("BuildingAge")), errors="coerce")
        last_reno_age = pd.to_numeric(pd.Series(out.get("RenovationAge")), errors="coerce")
        condition = self._to_num(df, "BuildingCondition")
        dom["RenovationNeed"] = np.maximum(0, building_age - last_reno_age - 20.0) * (6.0 - condition)

        total_basement = pd.to_numeric(pd.Series(out.get("TotalBasementArea")), errors="coerce")
        plot_size = self._to_num(df, "PlotSize")
        total_area = living_area + total_basement
        dom["LandUtilization"] = self._safe_ratio(total_area.to_numpy(), (plot_size + 1e-6).to_numpy())

        return dom

    # ---------- sklearn API ----------
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        df = X.copy()
        out: Dict[str, Any] = {}

        to_num = self._to_num
        safe_ratio = self._safe_ratio

        # 年龄
        y_listed = to_num(df, "YearListed")
        y_const  = to_num(df, "ConstructionYear")
        y_reno   = to_num(df, "RenovationYear")

        building_age = (y_listed - y_const)
        building_age = building_age.mask(building_age < 0, np.nan)
        out["BuildingAge"] = building_age

        years_since_reno = (y_listed - y_reno)
        years_since_reno = years_since_reno.mask(years_since_reno < 0, np.nan)
        out["YearsSinceRenovation"] = years_since_reno
        out["IsRenovated"] = (y_reno > y_const).astype(float).where(~(y_reno.isna() | y_const.isna()), np.nan)
        out["RenovationAge"] = years_since_reno

        # 面积
        ground = to_num(df, "GroundFloorArea")
        upper  = to_num(df, "UpperFloorArea")
        basement_area = to_num(df, "BasementArea")
        fin_b1 = to_num(df, "FinishedBasementArea1")
        fin_b2 = to_num(df, "FinishedBasementArea2")
        unfinished = to_num(df, "UnfinishedBasementArea")

        total_living = ground + upper
        out["TotalLivingArea"] = total_living

        total_basement = basement_area.copy()
        fallback = fin_b1.fillna(0) + fin_b2.fillna(0) + unfinished.fillna(0)
        total_basement = total_basement.where(~total_basement.isna(), fallback.where(fallback > 0, np.nan))
        out["TotalBasementArea"] = total_basement

        # 比率
        office = to_num(df, "OfficeSpace")
        plot_size = to_num(df, "PlotSize")
        total_rooms = to_num(df, "TotalRooms")
        parking = to_num(df, "ParkingSpots")

        out["BasementFinishRatio"] = safe_ratio((fin_b1.fillna(0) + fin_b2.fillna(0)).to_numpy(),
                                                basement_area.to_numpy())
        out["OfficeSpaceRatio"] = safe_ratio(office.to_numpy(), total_living.to_numpy())
        out["PlotCoverage"] = safe_ratio(total_living.to_numpy(), plot_size.to_numpy())
        out["RoomDensity"] = safe_ratio(total_rooms.to_numpy(), total_living.to_numpy())
        out["ParkingPerArea"] = safe_ratio(parking.to_numpy(), total_living.to_numpy())
        out["HasParking"] = np.where(parking.fillna(0) > 0, 1.0, 0.0)
        out["BasementUtilization"] = safe_ratio((fin_b1.fillna(0) + fin_b2.fillna(0)).to_numpy(),
                                                (basement_area.fillna(0).to_numpy() + 1))

        # 质量组合
        def prod(a, b, c=None):
            def num(x): return to_num(df, x)
            if c is None:
                x = num(a) * num(b)
            else:
                x = num(a) * num(b) * num(c)
            return x.replace([np.inf, -np.inf], np.nan)

        out["OverallQuality"] = prod("BuildingGrade", "BuildingCondition")
        out["ExteriorScore"]  = prod("ExteriorQuality", "ExteriorCondition")
        out["BasementOverallScore"] = prod("BasementQuality", "BasementCondition", "BasementExposure")

        # 组合/季节/年代
        prox1 = to_num(df, "Proximity1")
        prox2 = to_num(df, "Proximity2")
        out["ProximityScore"] = prox1.add(prox2, fill_value=np.nan)

        month = to_num(df, "MonthListed")
        out["SeasonListed"] = ((month % 12) // 3 + 1).where(~month.isna(), np.nan)

        out["ConstructionDecade"] = (y_const // 10 * 10).where(~y_const.isna(), np.nan)

        # 分箱
        try:
            out["PlotSize_binned"] = pd.cut(plot_size, bins=5, labels=False).astype("float")
        except Exception:
            out["PlotSize_binned"] = np.nan
        try:
            out["BuildingAge_binned"] = pd.cut(out["BuildingAge"],
                                               bins=[0, 10, 25, 50, 100, 200],
                                               labels=False, include_lowest=True).astype("float")
        except Exception:
            out["BuildingAge_binned"] = np.nan

        # === 新增：高级时间 / 交互 / 领域知识 ===
        out.update(self.add_advanced_temporal_features(df, out))
        out.update(self.add_advanced_interactions(df, out))
        out.update(self.add_domain_knowledge_features(df, out))

        out_df = pd.DataFrame(out)
        self.feature_names_ = list(out_df.columns)
        return out_df.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_ or [])


# ------------------------------------------------------------
# 统计聚合特征（可与 WideFeatureBuilder 并行）
# ------------------------------------------------------------

class StatisticalAggregator(BaseEstimator, TransformerMixin):
    """
    对 groupby 列做聚合，生成组内 z-score、相对偏移等。
    ✅ 关键：__init__ 不要改写入参；仅做原样赋值。
    """
    def __init__(self, groupby_cols=None, agg_cols=None):
        # 不要 list(...) / set(...) / tuple(...)；原样保存
        self.groupby_cols = groupby_cols
        self.agg_cols = agg_cols
        # 下面这些是拟合态
        self.agg_stats_ = None
        self._transform_output = None
        self.feature_names_ = None

    def set_output(self, *, transform=None):
        self._transform_output = transform
        return self

    @staticmethod
    def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        def to_num(col):
            return pd.to_numeric(out[col], errors='coerce') if col in out.columns else pd.Series([np.nan]*len(out))
        if "TotalLivingArea" not in out.columns:
            out["TotalLivingArea"] = to_num("GroundFloorArea") + to_num("UpperFloorArea")
        if "BuildingAge" not in out.columns:
            out["BuildingAge"] = to_num("YearListed") - to_num("ConstructionYear")
            out["BuildingAge"] = out["BuildingAge"].mask(out["BuildingAge"] < 0, np.nan)
        if "OverallQuality" not in out.columns:
            out["OverallQuality"] = to_num("BuildingGrade") * to_num("BuildingCondition")
        return out

    def fit(self, X: pd.DataFrame, y=None):
        Xc = self._ensure_cols(X)
        # 在 fit 里使用默认值回退（此处才可以构造 list/tuple）
        groupby_cols = self.groupby_cols if self.groupby_cols is not None else ('ZoningClassification','BuildingType')
        agg_cols = self.agg_cols if self.agg_cols is not None else ('TotalLivingArea','BuildingAge','OverallQuality')

        self.agg_stats_ = {}
        for gcol in groupby_cols:
            if gcol in Xc.columns:
                for acol in agg_cols:
                    if acol in Xc.columns:
                        stats = Xc.groupby(gcol)[acol].agg(['mean', 'std', 'median'])
                        self.agg_stats_[(gcol, acol)] = stats
        return self

    def transform(self, X: pd.DataFrame):
        Xc = self._ensure_cols(X)
        feats: Dict[str, np.ndarray] = {}
        for (gcol, acol), stats in (self.agg_stats_ or {}).items():
            if gcol in Xc.columns and acol in Xc.columns:
                gmean = Xc[gcol].map(stats['mean'])
                gstd  = Xc[gcol].map(stats['std'])
                cur   = Xc[acol]
                feats[f"{acol}_ZScore__{gcol}"]   = (cur - gmean) / (gstd + 1e-6)
                feats[f"{acol}_RelShift__{gcol}"] = (cur - gmean) / (np.abs(gmean) + 1e-6)
        out_df = pd.DataFrame(feats)
        self.feature_names_ = list(out_df.columns)
        return out_df.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_ or [])



# ------------------------------------------------------------
# 对数化某列（不增删列）
# ------------------------------------------------------------
class Log1pOnColumn(BaseEstimator, TransformerMixin):
    """
    在数值矩阵中对指定列做 log1p 原地替换；不增删列，不改变索引顺序。
    通过 `col_names`（ColumnTransformer 传入的列名顺序）定位具体列索引。
    """
    def __init__(self, col_names: Sequence[str], target_col: str = "PlotSize", clip_min: float = 0.0):
        # 用 tuple 保持不可变（方便 sklearn clone）
        self.col_names = tuple(col_names)
        self.target_col = target_col
        self.clip_min = clip_min
        self._idx = None

    def fit(self, X, y=None):
        self._idx = self.col_names.index(self.target_col) if self.target_col in self.col_names else None
        return self

    def transform(self, X):
        A = np.asarray(X, dtype=float)
        if self._idx is not None and self._idx < A.shape[1]:
            A = A.copy()
            v = A[:, self._idx]
            v = np.log1p(np.clip(v, self.clip_min, None))
            A[:, self._idx] = v
        return A


# ------------------------------------------------------------
# 组装 ColumnTransformer（避免重复编码） + 新增宽特征 & 聚合特征分支
# ------------------------------------------------------------
def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    - 删除: AlleyAccess, ExteriorFinishType, RecreationQuality, MiscellaneousFeature
    - ConferenceRoomQuality 的“有业务含义缺失”：
        * 类别支线: 缺失填 "__MISSING__" + One-Hot
        * 分支: 生成 ConferenceRoomQuality_missing (0/1)
    - 新增：
        * 宽特征分支 WideFeatureBuilder（含高级时间/交互/领域知识）
        * 统计聚合分支 StatisticalAggregator（基于 zoning/building type 的组内标准化）
        * 数值分支对 PlotSize 做 log1p
    """
    policy_drop = ["AlleyAccess", "ExteriorFinishType", "RecreationQuality", "MiscellaneousFeature"]
    business_missing_col = "ConferenceRoomQuality"

    # 清理列清单
    num_cols = [c for c in num_cols if c not in policy_drop]
    cat_cols = [c for c in cat_cols if c not in policy_drop]

    if (business_missing_col not in num_cols) and (business_missing_col not in cat_cols):
        cat_cols.append(business_missing_col)

    # 候选：频率编码 / 目标编码
    freq_cols = ['RoofType', 'ExteriorCovering1', 'FoundationType']
    freq_cols = [c for c in freq_cols if c in (num_cols + cat_cols)]

    te_cols = ['ZoningClassification', 'BuildingType', 'BuildingStyle', 'HeatingType']
    te_cols = [c for c in te_cols if c in (num_cols + cat_cols)]

    # 避免与 OHE 重复
    cat_cols_oh = [c for c in cat_cols if c not in set(freq_cols + te_cols)]

    # 宽特征需要的输入（原始列）
    wide_needed = [
        "YearListed", "ConstructionYear", "RenovationYear",
        "GroundFloorArea", "UpperFloorArea", "BasementArea",
        "FinishedBasementArea1", "FinishedBasementArea2", "UnfinishedBasementArea",
        "OfficeSpace", "PlotSize", "TotalRooms", "ParkingSpots",
        "BuildingGrade", "BuildingCondition", "ExteriorQuality", "ExteriorCondition",
        "BasementQuality", "BasementCondition", "BasementExposure",
        "Proximity1", "Proximity2", "MonthListed"
    ]
    wide_input_cols = [c for c in wide_needed if c in (num_cols + cat_cols)]

    # 统计聚合分支所需最小输入：groupby+原始列（内部会补算派生）
    agg_input_cols = list(set(
        ['ZoningClassification', 'BuildingType',
         'GroundFloorArea', 'UpperFloorArea',
         'YearListed', 'ConstructionYear',
         'BuildingGrade', 'BuildingCondition',
         'PlotSize']
    ) & set(num_cols + cat_cols))

    # 数值分支
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("log_plot", Log1pOnColumn(col_names=num_cols, target_col="PlotSize")),
    ])

    # 类别分支
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # CRQ 缺失指示
    crq_missing_tf = FunctionTransformer(crq_missing_indicator, validate=False)

    # 宽特征分支
    wide_pipe = Pipeline(steps=[
        ("builder", WideFeatureBuilder()),
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # 频率编码分支
    freq_pipe = Pipeline([("freq", FrequencyEncoder(freq_cols))]) if freq_cols else None

    # 多分类目标编码
    te_pipe = Pipeline([("te", MultiClassTargetEncoder(te_cols, alpha=10.0))]) if te_cols else None

    # 统计聚合分支
    agg_pipe = Pipeline(steps=[
        ("agg", StatisticalAggregator(
            groupby_cols=['ZoningClassification', 'BuildingType'],
            agg_cols=['TotalLivingArea', 'BuildingAge', 'OverallQuality']
        )),
        ("imputer", SimpleImputer(strategy="median"))
    ]) if agg_input_cols else None

    transformers = [
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols_oh),
        ("crq_missing", crq_missing_tf, [business_missing_col]),
    ]
    if wide_input_cols:
        transformers.append(("wide_feats", wide_pipe, wide_input_cols))
    if freq_pipe is not None:
        transformers.append(("freq_enc", freq_pipe, freq_cols))
    if te_pipe is not None:
        transformers.append(("target_enc", te_pipe, te_cols))
    if agg_pipe is not None:
        transformers.append(("agg_feats", agg_pipe, agg_input_cols))

    # ✅ 末端列选择器：只保留你指定的输出列索引 SELECTED_IDX = [3, 138, 15, 271, 4, 273, 260, 269, 5, 18, 25, 20, 256, 144, 22, 261, 11, 148, 267, 137, 263, 188, 26, 266, 146, 12, 182, 218, 91, 215, 70, 19, 280, 209, 24, 255, 284, 187, 221, 90]
    ''' preprocessor = Pipeline(steps=[ ("ct", column_tf), ("select", ColumnIndexSelector(SELECTED_IDX)), ]) '''
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    return preprocessor
