"""
Wide feature builder for creating derived features.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin


class WideFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Generate a comprehensive set of derived features:
    
    1. Age features: BuildingAge, YearsSinceRenovation, IsRenovated, etc.
    2. Area features: TotalLivingArea, TotalBasementArea, ratios
    3. Quality combinations: OverallQuality, ExteriorScore, BasementScore
    4. Proximity and location features
    5. Seasonal/temporal features
    6. Binned features
    7. Advanced temporal features: BuildingLifeStage, RenovationEffectiveness
    8. Interaction features: QualityAreaProximity, BasementEfficiency
    9. Domain knowledge features: RoomSizeAdequacy, ParkingAdequacy
    """
    
    def __init__(self):
        self.feature_names_: List[str] = []
    
    @staticmethod
    def _to_num(df: pd.DataFrame, col: str) -> pd.Series:
        """Convert column to numeric, handling missing columns."""
        if col in df.columns:
            return pd.to_numeric(df[col], errors='coerce')
        return pd.Series([np.nan] * len(df), index=df.index)
    
    @staticmethod
    def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        """Compute ratio with safe division."""
        den = np.asarray(den, dtype=float)
        num = np.asarray(num, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=(den > 0))
    
    @staticmethod
    def _clip_series(arr: np.ndarray, lower: float = 0.0, upper: float = 20.0) -> np.ndarray:
        """Clip array values while preserving NaNs."""
        clipped = np.clip(arr, lower, upper)
        clipped[np.isnan(arr)] = np.nan
        return clipped
    
    def _add_age_features(self, df: pd.DataFrame, out: Dict[str, Any]):
        """Add age-related features."""
        y_listed = self._to_num(df, "YearListed")
        y_const = self._to_num(df, "ConstructionYear")
        y_reno = self._to_num(df, "RenovationYear")
        
        # Building age
        building_age = (y_listed - y_const)
        building_age = building_age.mask(building_age < 0, np.nan)
        out["BuildingAge"] = building_age
        
        # Years since renovation
        years_since_reno = (y_listed - y_reno)
        years_since_reno = years_since_reno.mask(years_since_reno < 0, np.nan)
        out["YearsSinceRenovation"] = years_since_reno
        out["RenovationAge"] = years_since_reno
        
        # Is renovated flag
        out["IsRenovated"] = (y_reno > y_const).astype(float).where(
            ~(y_reno.isna() | y_const.isna()), np.nan
        )
    
    def _add_area_features(self, df: pd.DataFrame, out: Dict[str, Any]):
        """Add area-related features."""
        ground = self._to_num(df, "GroundFloorArea")
        upper = self._to_num(df, "UpperFloorArea")
        basement_area = self._to_num(df, "BasementArea")
        fin_b1 = self._to_num(df, "FinishedBasementArea1")
        fin_b2 = self._to_num(df, "FinishedBasementArea2")
        unfinished = self._to_num(df, "UnfinishedBasementArea")
        
        # Total living area
        total_living = ground + upper
        out["TotalLivingArea"] = total_living
        
        # Total basement area
        total_basement = basement_area.copy()
        fallback = fin_b1.fillna(0) + fin_b2.fillna(0) + unfinished.fillna(0)
        total_basement = total_basement.where(
            ~total_basement.isna(),
            fallback.where(fallback > 0, np.nan)
        )
        out["TotalBasementArea"] = total_basement
    
    def _add_ratio_features(self, df: pd.DataFrame, out: Dict[str, Any]):
        """Add ratio features."""
        fin_b1 = self._to_num(df, "FinishedBasementArea1")
        fin_b2 = self._to_num(df, "FinishedBasementArea2")
        basement_area = self._to_num(df, "BasementArea")
        office = self._to_num(df, "OfficeSpace")
        plot_size = self._to_num(df, "PlotSize")
        total_rooms = self._to_num(df, "TotalRooms")
        parking = self._to_num(df, "ParkingSpots")
        meeting_rooms = self._to_num(df, "MeetingRooms")
        restrooms = self._to_num(df, "Restrooms")
        
        total_living = out.get("TotalLivingArea", pd.Series([np.nan] * len(df)))
        
        out["BasementFinishRatio"] = self._clip_series(self._safe_ratio(
            (fin_b1.fillna(0) + fin_b2.fillna(0)).to_numpy(),
            basement_area.to_numpy()
        ), upper=5.0)
        out["OfficeSpaceRatio"] = self._clip_series(self._safe_ratio(
            office.to_numpy(),
            total_living.to_numpy()
        ), upper=5.0)
        out["PlotCoverage"] = self._clip_series(self._safe_ratio(
            total_living.to_numpy(),
            plot_size.to_numpy()
        ), upper=10.0)
        out["RoomDensity"] = self._clip_series(self._safe_ratio(
            total_rooms.to_numpy(),
            total_living.to_numpy()
        ), upper=5.0)
        out["ParkingPerArea"] = self._clip_series(self._safe_ratio(
            parking.to_numpy(),
            total_living.to_numpy()
        ), upper=5.0)
        out["HasParking"] = np.where(parking.fillna(0) > 0, 1.0, 0.0)
        out["BasementUtilization"] = self._clip_series(self._safe_ratio(
            (fin_b1.fillna(0) + fin_b2.fillna(0)).to_numpy(),
            (basement_area.fillna(0).to_numpy() + 1)
        ), upper=5.0)
        amenities = meeting_rooms.fillna(0) + restrooms.fillna(0) + parking.fillna(0)
        out["AmenityDensity"] = self._clip_series(self._safe_ratio(
            amenities.to_numpy(),
            total_living.to_numpy()
        ), upper=5.0)
        out["RestroomPerRoom"] = self._clip_series(self._safe_ratio(
            restrooms.to_numpy(),
            total_rooms.to_numpy()
        ), upper=3.0)
    
    def _add_quality_features(self, df: pd.DataFrame, out: Dict[str, Any]):
        """Add quality combination features."""
        def prod(*cols):
            result = self._to_num(df, cols[0])
            for col in cols[1:]:
                result = result * self._to_num(df, col)
            return result.replace([np.inf, -np.inf], np.nan)
        
        out["OverallQuality"] = prod("BuildingGrade", "BuildingCondition")
        out["ExteriorScore"] = prod("ExteriorQuality", "ExteriorCondition")
        out["BasementOverallScore"] = prod(
            "BasementQuality", "BasementCondition", "BasementExposure"
        )
    
    def _add_location_features(self, df: pd.DataFrame, out: Dict[str, Any]):
        """Add location/proximity features."""
        prox1 = self._to_num(df, "Proximity1")
        prox2 = self._to_num(df, "Proximity2")
        out["ProximityScore"] = prox1.add(prox2, fill_value=np.nan)
    
    def _add_temporal_features(self, df: pd.DataFrame, out: Dict[str, Any]):
        """Add temporal/seasonal features."""
        y_const = self._to_num(df, "ConstructionYear")
        month = self._to_num(df, "MonthListed")
        
        # Season (1-4)
        out["SeasonListed"] = ((month % 12) // 3 + 1).where(~month.isna(), np.nan)
        
        # Construction decade
        out["ConstructionDecade"] = (y_const // 10 * 10).where(~y_const.isna(), np.nan)
    
    def _add_binned_features(self, df: pd.DataFrame, out: Dict[str, Any]):
        """Add binned features."""
        plot_size = self._to_num(df, "PlotSize")
        building_age = out.get("BuildingAge", pd.Series([np.nan] * len(df)))
        
        # Plot size bins
        try:
            out["PlotSize_binned"] = pd.cut(
                plot_size, bins=5, labels=False
            ).astype("float")
        except Exception:
            out["PlotSize_binned"] = np.nan
        
        # Building age bins
        try:
            out["BuildingAge_binned"] = pd.cut(
                building_age,
                bins=[0, 10, 25, 50, 100, 200],
                labels=False,
                include_lowest=True
            ).astype("float")
        except Exception:
            out["BuildingAge_binned"] = np.nan
    
    def _add_advanced_temporal_features(self, df: pd.DataFrame, out: Dict[str, Any]):
        """Add advanced temporal features."""
        building_age = pd.to_numeric(pd.Series(out.get("BuildingAge")), errors="coerce")
        reno_age = pd.to_numeric(pd.Series(out.get("RenovationAge")), errors="coerce")
        
        # Building life stage (1-5)
        out["BuildingLifeStage"] = np.select(
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
        
        # Renovation effectiveness (exponential decay)
        out["RenovationEffectiveness"] = np.exp(-reno_age / 20.0)
        
        # Seasonal strength
        month = self._to_num(df, "MonthListed")
        out["SeasonalStrength"] = np.abs(6 - np.abs(month - 6)) / 6.0
    
    def _add_interaction_features(self, df: pd.DataFrame, out: Dict[str, Any]):
        """Add interaction features."""
        quality = self._to_num(df, "BuildingGrade")
        area = pd.to_numeric(pd.Series(out.get("TotalLivingArea")), errors="coerce")
        proximity = pd.to_numeric(pd.Series(out.get("ProximityScore")), errors="coerce")
        
        # Quality × Area × Proximity
        out["QualityAreaProximity"] = quality * np.log1p(area) * (
            1.0 / (1.0 + np.where(np.isnan(proximity), 0.0, proximity))
        )
        
        # Basement efficiency
        basement_finished = (
            self._to_num(df, "FinishedBasementArea1").fillna(0) +
            self._to_num(df, "FinishedBasementArea2").fillna(0)
        )
        basement_total = self._to_num(df, "BasementArea")
        basement_quality = self._to_num(df, "BasementQuality")
        basement_exposure = self._to_num(df, "BasementExposure")
        
        out["BasementEfficiency"] = (
            self._safe_ratio(basement_finished.to_numpy(), basement_total.to_numpy()) *
            basement_quality *
            basement_exposure
        )
        
        # Value density
        condition = self._to_num(df, "BuildingCondition")
        plot_size = self._to_num(df, "PlotSize")
        
        out["ValueDensity"] = (
            quality * condition * np.log1p(area) / np.log1p(plot_size + 1e-3)
        )
        
        # Age-adjusted quality/area interactions
        building_age = pd.to_numeric(pd.Series(out.get("BuildingAge")), errors="coerce")
        overall_quality = pd.to_numeric(pd.Series(out.get("OverallQuality")), errors="coerce")
        out["AgeQualityInteraction"] = self._clip_series(
            np.log1p(overall_quality) / (1.0 + building_age),
            upper=10.0
        )
        out["LogQualityArea"] = self._clip_series(
            np.log1p(overall_quality) * np.log1p(area + 1.0),
            upper=50.0
        )
    
    def _add_domain_knowledge_features(self, df: pd.DataFrame, out: Dict[str, Any]):
        """Add domain knowledge features."""
        living_area = pd.to_numeric(pd.Series(out.get("TotalLivingArea")), errors="coerce")
        total_rooms = self._to_num(df, "TotalRooms")
        parking_spots = self._to_num(df, "ParkingSpots")
        building_age = pd.to_numeric(pd.Series(out.get("BuildingAge")), errors="coerce")
        last_reno_age = pd.to_numeric(pd.Series(out.get("RenovationAge")), errors="coerce")
        condition = self._to_num(df, "BuildingCondition")
        total_basement = pd.to_numeric(pd.Series(out.get("TotalBasementArea")), errors="coerce")
        plot_size = self._to_num(df, "PlotSize")
        
        # Room size adequacy
        out["RoomSizeAdequacy"] = self._safe_ratio(
            living_area.to_numpy(),
            (total_rooms + 1e-6).to_numpy()
        )
        
        # Parking adequacy
        out["ParkingAdequacy"] = self._safe_ratio(
            parking_spots.to_numpy(),
            (total_rooms / 3.0 + 1e-6).to_numpy()
        )
        
        # Renovation need
        out["RenovationNeed"] = (
            np.maximum(0, building_age - last_reno_age - 20.0) * (6.0 - condition)
        )
        
        # Land utilization
        total_area = living_area + total_basement
        out["LandUtilization"] = self._safe_ratio(
            total_area.to_numpy(),
            (plot_size + 1e-6).to_numpy()
        )
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit is a no-op for this transformer."""
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform by creating all derived features.
        
        Args:
            X: Input DataFrame
        
        Returns:
            Numpy array of derived features
        """
        df = X.copy()
        out: Dict[str, Any] = {}
        
        # Build features in order (some depend on others)
        self._add_age_features(df, out)
        self._add_area_features(df, out)
        self._add_ratio_features(df, out)
        self._add_quality_features(df, out)
        self._add_location_features(df, out)
        self._add_temporal_features(df, out)
        self._add_binned_features(df, out)
        self._add_advanced_temporal_features(df, out)
        self._add_interaction_features(df, out)
        self._add_domain_knowledge_features(df, out)
        
        # Convert to DataFrame then numpy
        out_df = pd.DataFrame(out, index=df.index)
        self.feature_names_ = list(out_df.columns)
        
        return out_df.to_numpy(dtype=float)
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return np.array(self.feature_names_ if self.feature_names_ else [])
