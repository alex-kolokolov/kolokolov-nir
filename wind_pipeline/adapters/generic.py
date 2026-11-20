from __future__ import annotations

import pandas as pd

from wind_pipeline.features import add_universal_features
from wind_pipeline.schema import complete_feature_space, validate_canonical

from .base import DatasetAdapter
from .legacy_scada import _rated_power


class GenericAdapter(DatasetAdapter):
    """Config-driven adapter for Engie, GEFCom, EDP and future domains.

    The columns mapping is source-name -> canonical/raw semantic name.
    """

    def load(self) -> pd.DataFrame:
        raw = self.read_source()
        if not self.config.columns:
            raise ValueError("The generic adapter requires a columns mapping")
        missing_sources = [c for c in self.config.columns if c not in raw]
        if missing_sources:
            raise ValueError(f"Configured source columns are absent: {missing_sources}")
        df = raw.rename(columns=self.config.columns).copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if "farm_id" not in df:
            if not self.config.farm_id:
                raise ValueError("Map farm_id or set farm_id in config")
            df["farm_id"] = self.config.farm_id
        if "turbine_id" not in df:
            df["turbine_id"] = df["farm_id"].astype(str) + "_aggregate"
        df["dataset_id"] = self.config.dataset_id
        df["rated_power_kw"] = _rated_power(df["farm_id"], self.config.rated_power_kw)
        if "power_pu" not in df:
            if "power_kw" not in df:
                raise ValueError("Map either power_kw or power_pu")
            df["power_pu"] = pd.to_numeric(df["power_kw"], errors="coerce") / df["rated_power_kw"]
        df = _derive_nwp_speed_direction(df)
        df = df.dropna(subset=["timestamp", "farm_id", "turbine_id", "power_pu"])
        df = df.sort_values(["turbine_id", "timestamp"]).drop_duplicates(
            ["turbine_id", "timestamp"], keep="last"
        )
        df = add_universal_features(df, sample_minutes=self.config.sample_minutes)
        df = complete_feature_space(df)
        validate_canonical(df)
        return df.reset_index(drop=True)


def _derive_nwp_speed_direction(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np

    out = df.copy()
    for height in (10, 100):
        u, v = f"wind_u_{height}_ms", f"wind_v_{height}_ms"
        speed, direction = f"wind_speed_{height}_ms", f"wind_direction_{height}_deg"
        if {u, v}.issubset(out.columns):
            out[speed] = np.hypot(out[u], out[v])
            out[direction] = np.rad2deg(np.arctan2(-out[u], -out[v])) % 360.0
    if "wind_speed_ms" not in out and "wind_speed_100_ms" in out:
        out["wind_speed_ms"] = out["wind_speed_100_ms"]
    if "wind_direction_deg" not in out and "wind_direction_100_deg" in out:
        out["wind_direction_deg"] = out["wind_direction_100_deg"]
    return out
