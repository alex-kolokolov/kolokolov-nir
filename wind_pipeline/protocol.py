from __future__ import annotations

import pandas as pd


def select_farms(frame: pd.DataFrame, farms: tuple[str, ...]) -> pd.DataFrame:
    if not farms:
        return frame.copy()
    return frame[frame["farm_id"].isin(farms)].copy()


def temporal_train_val_split(
    frame: pd.DataFrame, val_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split every turbine chronologically; never fit preprocessing on validation."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    ordered = frame.sort_values(["turbine_id", "timestamp"]).copy()
    rank = ordered.groupby("turbine_id", sort=False).cumcount()
    size = ordered.groupby("turbine_id", sort=False)["timestamp"].transform("size")
    is_val = rank >= (size * (1.0 - val_fraction)).astype(int)
    return ordered[~is_val].copy(), ordered[is_val].copy()

