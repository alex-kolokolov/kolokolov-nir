from __future__ import annotations

import numpy as np
import pandas as pd


def circular_difference_deg(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b + 180.0) % 360.0 - 180.0


def add_universal_features(
    df: pd.DataFrame,
    *,
    sample_minutes: int = 10,
    wind_direction_col: str = "wind_direction_deg",
) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], errors="coerce")
    out["timestamp"] = ts
    if "wind_speed_ms" in out:
        ws = pd.to_numeric(out["wind_speed_ms"], errors="coerce")
        out["wind_speed_sq"] = ws.pow(2)
        out["wind_speed_cu"] = ws.pow(3)
    else:
        out["wind_speed_sq"] = np.nan
        out["wind_speed_cu"] = np.nan

    if wind_direction_col in out:
        direction = pd.to_numeric(out[wind_direction_col], errors="coerce")
        rad = np.deg2rad(direction)
        out["wind_dir_sin"] = np.sin(rad)
        out["wind_dir_cos"] = np.cos(rad)
    else:
        out["wind_dir_sin"] = np.nan
        out["wind_dir_cos"] = np.nan

    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    out["hour_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    out["hour_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    year_phase = (ts.dt.dayofyear - 1 + minute_of_day / 1440.0) / 365.2425
    out["year_sin"] = np.sin(2 * np.pi * year_phase)
    out["year_cos"] = np.cos(2 * np.pi * year_phase)

    if {"pitch_1_deg", "pitch_2_deg", "pitch_3_deg"}.issubset(out.columns):
        out["pitch_max_deg"] = out[["pitch_1_deg", "pitch_2_deg", "pitch_3_deg"]].max(axis=1)
    if {wind_direction_col, "nacelle_direction_deg"}.issubset(out.columns):
        yaw = circular_difference_deg(out[wind_direction_col], out["nacelle_direction_deg"])
        out["yaw_error_sin"] = np.sin(np.deg2rad(yaw))
        out["yaw_error_cos"] = np.cos(np.deg2rad(yaw))
    elif "yaw_error_deg" in out:
        yaw = pd.to_numeric(out["yaw_error_deg"], errors="coerce")
        out["yaw_error_sin"] = np.sin(np.deg2rad(yaw))
        out["yaw_error_cos"] = np.cos(np.deg2rad(yaw))
    if {"nacelle_temp_c", "ambient_temp_c"}.issubset(out.columns):
        out["temp_delta_c"] = out["nacelle_temp_c"] - out["ambient_temp_c"]

    if "wind_speed_std_ms" in out:
        out["turbulence_intensity"] = out["wind_speed_std_ms"] / ws.clip(lower=0.1)
    else:
        window = max(1, 60 // sample_minutes)
        grouped = out.groupby("turbine_id", sort=False)["wind_speed_ms"]
        mean = grouped.transform(lambda s: s.rolling(window, min_periods=window).mean())
        std = grouped.transform(lambda s: s.rolling(window, min_periods=window).std())
        out["turbulence_intensity"] = std / mean.clip(lower=0.1)
    if {"pressure_pa", "ambient_temp_c"}.issubset(out.columns):
        out["air_density_kgm3"] = out["pressure_pa"] / (
            287.05 * (out["ambient_temp_c"] + 273.15)
        )

    _add_wind_vectors(out, "wind_speed_10_ms", "wind_direction_10_deg", 10)
    _add_wind_vectors(out, "wind_speed_100_ms", "wind_direction_100_deg", 100)
    if {"wind_speed_10_ms", "wind_speed_100_ms"}.issubset(out.columns):
        v10 = out["wind_speed_10_ms"].clip(lower=1e-3)
        v100 = out["wind_speed_100_ms"].clip(lower=1e-3)
        out["wind_shear_alpha"] = np.log(v100 / v10) / np.log(10.0)
    return out


def _add_wind_vectors(
    df: pd.DataFrame, speed_col: str, direction_col: str, height: int
) -> None:
    if {speed_col, direction_col}.issubset(df.columns):
        angle = np.deg2rad(df[direction_col])
        df[f"wind_u_{height}_ms"] = -df[speed_col] * np.sin(angle)
        df[f"wind_v_{height}_ms"] = -df[speed_col] * np.cos(angle)
