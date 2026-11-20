from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Allows data validation without the neural extra installed.
    torch = None
    Dataset = object

from .schema import MODEL_FEATURES, validate_canonical


@dataclass(frozen=True)
class WindowIndex:
    start: int
    target_indices: tuple[int, ...]


class UniversalWindDataset(Dataset):
    """Gap-safe windows with explicit masks for dynamically available channels."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        lookback: int = 96,
        horizons: Sequence[int] = (1, 3, 6, 12, 24),
        features: Sequence[str] = MODEL_FEATURES,
        sample_minutes: int = 10,
        stride: int = 1,
        as_torch: bool = True,
    ):
        validate_canonical(frame)
        if lookback < 2 or not horizons or min(horizons) < 1:
            raise ValueError("lookback >= 2 and positive horizons are required")
        self.lookback = int(lookback)
        self.horizons = tuple(int(h) for h in horizons)
        self.features = tuple(features)
        self.as_torch = as_torch
        if as_torch and torch is None:
            raise ImportError("Install torch to receive tensors, or set as_torch=False")

        missing = [c for c in self.features if c not in frame]
        if missing:
            raise ValueError(f"Frame is missing selected features: {missing}")
        self.frame = frame.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)
        self.values = self.frame.loc[:, self.features].to_numpy(dtype=np.float32)
        self.mask = np.isfinite(self.values)
        self.values = np.nan_to_num(self.values, nan=0.0, posinf=0.0, neginf=0.0)
        self.targets = self.frame["power_pu"].to_numpy(dtype=np.float32)
        self.indices = _build_indices(
            self.frame,
            lookback=self.lookback,
            horizons=self.horizons,
            sample_minutes=sample_minutes,
            stride=stride,
        )
        anchors = [item.start + self.lookback - 1 for item in self.indices]
        self.window_farms = self.frame.loc[anchors, "farm_id"].astype(str).to_numpy()
        self.window_turbines = self.frame.loc[anchors, "turbine_id"].astype(str).to_numpy()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        spec = self.indices[item]
        stop = spec.start + self.lookback
        x = self.values[spec.start:stop]
        x_mask = self.mask[spec.start:stop]
        y = self.targets[list(spec.target_indices)]
        row = self.frame.iloc[stop - 1]
        p_last = np.float32(self.targets[stop - 1])
        sample: dict[str, Any] = {
            "x": x,
            "x_mask": x_mask,
            "y": y,
            "p_last": p_last,
            "rated_power_kw": np.float32(row["rated_power_kw"]),
            "turbine_id": str(row["turbine_id"]),
            "farm_id": str(row["farm_id"]),
            "timestamp": str(row["timestamp"]),
        }
        if self.as_torch:
            sample["x"] = torch.from_numpy(x)
            sample["x_mask"] = torch.from_numpy(x_mask)
            sample["y"] = torch.from_numpy(y)
            sample["p_last"] = torch.as_tensor(p_last)
            sample["rated_power_kw"] = torch.as_tensor(sample["rated_power_kw"])
        return sample


def _build_indices(
    frame: pd.DataFrame,
    *,
    lookback: int,
    horizons: tuple[int, ...],
    sample_minutes: int,
    stride: int,
) -> list[WindowIndex]:
    result: list[WindowIndex] = []
    max_horizon = max(horizons)
    expected = pd.Timedelta(minutes=sample_minutes)
    targets_finite = np.isfinite(frame["power_pu"].to_numpy())
    for _, group in frame.groupby("turbine_id", sort=False):
        positions = group.index.to_numpy()
        times = group["timestamp"]
        breaks = np.flatnonzero(times.diff().ne(expected).to_numpy())
        starts = breaks
        ends = np.r_[breaks[1:], len(group)]
        for local_start, local_end in zip(starts, ends):
            length = int(local_end - local_start)
            count = length - lookback - max_horizon + 1
            for offset in range(0, max(0, count), stride):
                start = int(positions[local_start + offset])
                anchor = start + lookback - 1
                target_indices = tuple(anchor + h for h in horizons)
                if targets_finite[list(target_indices)].all():
                    result.append(WindowIndex(start, target_indices))
    return result
