from __future__ import annotations

import pandas as pd

from wind_pipeline.features import add_universal_features
from wind_pipeline.schema import complete_feature_space, validate_canonical

from .base import DatasetAdapter


DEFAULT_COLUMNS = {
    "TurbID": "raw_turbine_id",
    "Patv": "power_kw",
    "Wspd": "wind_speed_ms",
    "Wdir": "wind_direction_deg",
    "Ndir": "nacelle_direction_deg",
    "Pab1": "pitch_1_deg",
    "Pab2": "pitch_2_deg",
    "Pab3": "pitch_3_deg",
    "Etmp": "ambient_temp_c",
    "Itmp": "nacelle_temp_c",
}


class SdwpfAdapter(DatasetAdapter):
    def load(self) -> pd.DataFrame:
        raw = self.read_source()
        mapping = {**DEFAULT_COLUMNS, **self.config.columns}
        selected = {src: dst for src, dst in mapping.items() if src in raw.columns}
        df = raw.rename(columns=selected).copy()
        relative_time = "timestamp" not in df
        df["timestamp"] = _timestamp(df)
        farm = self.config.farm_id or "SDWPF"
        df["farm_id"] = farm
        if "raw_turbine_id" not in df:
            raise ValueError("SDWPF source has no mapped turbine id")
        df["turbine_id"] = farm + "_T" + df["raw_turbine_id"].astype(str)
        df["dataset_id"] = self.config.dataset_id

        rated = self.config.rated_power_kw
        if not isinstance(rated, (int, float)):
            raise ValueError("SDWPF adapter requires numeric rated_power_kw")
        df["rated_power_kw"] = float(rated)
        df["power_kw"] = pd.to_numeric(df["power_kw"], errors="coerce")
        df["wind_speed_ms"] = pd.to_numeric(df["wind_speed_ms"], errors="coerce")

        # Official KDD-style invalid/stoppage rules. Keep the raw signal for audit.
        df["quality_excluded"] = (
            (df["wind_speed_ms"] > 2.5) & (df["power_kw"] <= 0)
        )
        df.loc[df["power_kw"] < 0, "power_kw"] = 0.0
        df = df[~df["quality_excluded"]].copy()
        df["power_pu"] = df["power_kw"] / df["rated_power_kw"]
        df = df.dropna(subset=["timestamp", "turbine_id", "power_pu", "wind_speed_ms"])
        df = df.sort_values(["turbine_id", "timestamp"]).drop_duplicates(
            ["turbine_id", "timestamp"], keep="last"
        )
        df = add_universal_features(df, sample_minutes=self.config.sample_minutes)
        if relative_time:
            # Day is only a relative competition index; a fake calendar season
            # would create a misleading China-to-Europe relationship.
            df[["year_sin", "year_cos"]] = float("nan")
        df = complete_feature_space(df)
        validate_canonical(df)
        return df.reset_index(drop=True)


def _timestamp(df: pd.DataFrame) -> pd.Series:
    if "timestamp" in df:
        return pd.to_datetime(df["timestamp"], errors="coerce")
    if {"Day", "Tmstamp"}.issubset(df.columns):
        # SDWPF exposes a relative day index. The anchor is intentionally synthetic.
        day = pd.to_numeric(df["Day"], errors="coerce").fillna(1).astype(int)
        text = df["Tmstamp"].astype(str)
        clock = pd.to_timedelta(text, errors="coerce")
        missing = clock.isna()
        clock.loc[missing] = pd.to_timedelta(
            text.loc[missing] + ":00", errors="coerce"
        )
        return pd.Timestamp("2020-01-01") + pd.to_timedelta(day - 1, unit="D") + clock
    raise ValueError("Map a timestamp column, or provide SDWPF Day and Tmstamp")
