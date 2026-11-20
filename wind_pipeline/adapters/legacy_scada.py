from __future__ import annotations

import pandas as pd

from wind_pipeline.features import add_universal_features
from wind_pipeline.schema import complete_feature_space, validate_canonical

from .base import DatasetAdapter


DEFAULT_COLUMNS = {
    "timestamp": "timestamp",
    "farm": "farm_id",
    "turbine_id": "turbine_id",
    "power_(kw)": "power_kw",
    "wind_speed_(m/s)": "wind_speed_ms",
    "wind_direction_(°)": "wind_direction_deg",
    "yaw_error_(°)": "yaw_error_deg",
    "nacelle_position_(°)": "nacelle_direction_deg",
    "nacelle_temperature_(°c)": "nacelle_temp_c",
    "ambient_temperature_(converter)_(°c)": "ambient_temp_c",
    "rotor_speed_(rpm)": "rotor_speed_rpm",
}


class LegacyScadaAdapter(DatasetAdapter):
    """Adapter for the existing Kelmarsh/Penmanshiel/Dundalk artifacts."""

    def load(self) -> pd.DataFrame:
        raw = self.read_source()
        mapping = {**DEFAULT_COLUMNS, **self.config.columns}
        selected = {src: dst for src, dst in mapping.items() if src in raw.columns}
        df = raw.rename(columns=selected).copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if "farm_id" not in df:
            if not self.config.farm_id:
                raise ValueError("farm_id is absent; set it in the data config")
            df["farm_id"] = self.config.farm_id
        if "turbine_id" not in df:
            df["turbine_id"] = df["farm_id"].astype(str) + "_T00"

        df["dataset_id"] = self.config.dataset_id
        df["rated_power_kw"] = _rated_power(df["farm_id"], self.config.rated_power_kw)
        if "power_kw" not in df:
            raise ValueError("Legacy SCADA source has no mapped power column")
        df["power_pu"] = pd.to_numeric(df["power_kw"], errors="coerce") / df["rated_power_kw"]
        df = df.dropna(subset=["timestamp", "farm_id", "turbine_id", "power_pu"])
        df = df.sort_values(["turbine_id", "timestamp"]).drop_duplicates(
            ["turbine_id", "timestamp"], keep="last"
        )
        df = add_universal_features(df, sample_minutes=self.config.sample_minutes)
        df = complete_feature_space(df)
        validate_canonical(df)
        return df.reset_index(drop=True)


def _rated_power(
    farms: pd.Series, value: float | dict[str, float] | None
) -> pd.Series:
    if isinstance(value, dict):
        result = farms.astype(str).map(value)
    elif value is not None:
        result = pd.Series(float(value), index=farms.index)
    else:
        raise ValueError(
            "rated_power_kw is required. Do not infer it on the full dataset: "
            "that leaks test-domain information."
        )
    if result.isna().any():
        missing = sorted(farms[result.isna()].astype(str).unique())
        raise ValueError(f"No rated power configured for farms: {missing}")
    return result.astype(float)

