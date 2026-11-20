from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CanonicalColumns:
    timestamp: str = "timestamp"
    dataset: str = "dataset_id"
    farm: str = "farm_id"
    turbine: str = "turbine_id"
    rated_power: str = "rated_power_kw"
    power: str = "power_pu"


FEATURE_TIERS: dict[str, tuple[str, ...]] = {
    "core": (
        "power_pu", "wind_speed_ms", "wind_speed_sq", "wind_speed_cu",
        "wind_dir_sin", "wind_dir_cos", "hour_sin", "hour_cos",
        "year_sin", "year_cos",
    ),
    "scada": (
        "pitch_max_deg", "yaw_error_sin", "yaw_error_cos",
        "turbulence_intensity", "temp_delta_c", "rotor_speed_rpm",
    ),
    "meteo_nwp": (
        "wind_shear_alpha", "air_density_kgm3", "wind_u_10_ms",
        "wind_v_10_ms", "wind_u_100_ms", "wind_v_100_ms",
    ),
}
MODEL_FEATURES: tuple[str, ...] = sum(FEATURE_TIERS.values(), ())
IDENTITY_COLUMNS = (
    "timestamp", "dataset_id", "farm_id", "turbine_id", "rated_power_kw",
)


def validate_canonical(df: pd.DataFrame) -> None:
    missing = [c for c in IDENTITY_COLUMNS + ("power_pu",) if c not in df.columns]
    if missing:
        raise ValueError(f"Canonical frame is missing columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise TypeError("timestamp must be datetime64")
    bad_power = df["rated_power_kw"].notna() & (df["rated_power_kw"] <= 0)
    if bad_power.any():
        raise ValueError("rated_power_kw must be positive")


def complete_feature_space(df: pd.DataFrame) -> pd.DataFrame:
    """Add absent channels as NaN and an explicit availability mask."""
    out = df.copy()
    for feature in MODEL_FEATURES:
        if feature not in out:
            out[feature] = np.nan
        out[f"{feature}__available"] = out[feature].notna().astype(np.uint8)
    return out

