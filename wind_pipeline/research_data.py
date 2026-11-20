"""Independent, auditable preparation for the UK cross-farm experiment.

Old row-shifted labels and full-series normalization artifacts are never read.
All numerical transforms use fixed physical scales, with no fitted statistics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


FEATURES = ('power_pu', 'wind_scaled', 'direction_sin', 'direction_cos',
            'rotor_scaled', 'ambient_scaled', 'nacelle_scaled', 'hour_sin', 'hour_cos')
RAW_COLUMNS = ('timestamp', 'farm', 'turbine_id', 'power_(kw)', 'wind_speed_(m/s)',
               'wind_direction_(°)', 'rotor_speed_(rpm)',
               'ambient_temperature_(converter)_(°c)', 'nacelle_temperature_(°c)')


@dataclass
class FarmData:
    farm: str
    frame: pd.DataFrame
    values: np.ndarray
    anchors: np.ndarray
    horizons: tuple[int, ...]
    lookback: int
    audit: dict

    def select(self, start: str, end: str, stride: int) -> 'WindowSet':
        """Entire context and all future labels must lie inside [start, end)."""
        if stride < 1:
            raise ValueError('stride must be positive')
        a = self.anchors
        ts = self.frame.timestamp.to_numpy()
        mask = ((ts[a - self.lookback + 1] >= np.datetime64(start)) &
                (ts[a + max(self.horizons)] < np.datetime64(end)))
        # Fixed global ten-minute clock, rather than a row index after filtering.
        ticks = ts[a].astype('datetime64[m]').astype(np.int64) // 10
        return WindowSet(self, a[mask & (ticks % stride == 0)])


class WindowSet(Dataset):
    def __init__(self, data: FarmData, anchors: np.ndarray):
        self.data = data
        self.anchors = np.asarray(anchors, dtype=np.int64)
        self.offsets = np.arange(1 - data.lookback, 1)
        self.targets = data.values[self.anchors[:, None] + np.array(data.horizons), 0].copy()
        self.persistence = data.values[self.anchors, 0].copy()
        self.times = pd.DatetimeIndex(data.frame.timestamp.iloc[self.anchors])
        self.turbines = data.frame.turbine_id.iloc[self.anchors].to_numpy()

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, i):
        x = self.data.values[self.anchors[i] + self.offsets]
        return torch.from_numpy(x), torch.from_numpy(self.targets[i])

    def tabular(self) -> np.ndarray:
        # CatBoost receives every raw time step available to GRU, plus summaries.
        x = self.data.values[self.anchors[:, None] + self.offsets]
        return np.concatenate([x.reshape(len(x), -1), x.mean(1), x.std(1),
                               x[:, -1] - x[:, 0]], axis=1)


def prepare_farm(raw: pd.DataFrame, farm: str, *, rated_kw: float = 2050,
                 lookback: int = 36, horizons=(1, 3, 6, 12),
                 features=FEATURES, preprocessing: dict | None = None) -> FarmData:
    if rated_kw <= 0 or lookback < 2 or min(horizons) <= 0:
        raise ValueError('Invalid rated power, lookback, or horizons')
    frame = raw.loc[raw.farm == farm, list(RAW_COLUMNS)].copy()
    if frame.empty:
        raise ValueError(f'No rows for {farm}')
    frame['timestamp'] = pd.to_datetime(frame.timestamp, errors='raise')
    frame = frame.sort_values(['turbine_id', 'timestamp']).reset_index(drop=True)
    if frame.duplicated(['turbine_id', 'timestamp']).any():
        raise ValueError('Duplicate turbine timestamps require explicit resolution')
    for col in RAW_COLUMNS[3:]:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    p = frame['power_(kw)'].to_numpy()
    ws = frame['wind_speed_(m/s)'].to_numpy()
    direction = frame['wind_direction_(°)'].to_numpy()
    rotor = frame['rotor_speed_(rpm)'].to_numpy()
    ambient = frame['ambient_temperature_(converter)_(°c)'].to_numpy()
    nacelle = frame['nacelle_temperature_(°c)'].to_numpy()
    cfg = dict(minimum_power_pu=-0.05, maximum_power_pu=1.1, maximum_wind_ms=60,
               maximum_rotor_rpm=60, minimum_temperature_c=-60, maximum_temperature_c=100,
               wind_scale_ms=15, rotor_scale_rpm=20, temperature_scale_c=40)
    cfg.update(preprocessing or {})
    power_ok = np.isfinite(p) & (p >= cfg['minimum_power_pu'] * rated_kw) & (p <= cfg['maximum_power_pu'] * rated_kw)
    ws = np.where((ws >= 0) & (ws <= cfg['maximum_wind_ms']), ws, np.nan)
    rotor = np.where((rotor >= 0) & (rotor <= cfg['maximum_rotor_rpm']), rotor, np.nan)
    ambient = np.where((ambient >= cfg['minimum_temperature_c']) & (ambient <= cfg['maximum_temperature_c']), ambient, np.nan)
    nacelle = np.where((nacelle >= cfg['minimum_temperature_c']) & (nacelle <= cfg['maximum_temperature_c']), nacelle, np.nan)
    minute = frame.timestamp.dt.hour.to_numpy() * 60 + frame.timestamp.dt.minute.to_numpy()
    radians = np.deg2rad(direction)
    values = np.column_stack([
        np.clip(p / rated_kw, 0, cfg['maximum_power_pu']), ws / cfg['wind_scale_ms'],
        np.sin(radians), np.cos(radians), rotor / cfg['rotor_scale_rpm'],
        ambient / cfg['temperature_scale_c'], nacelle / cfg['temperature_scale_c'],
        np.sin(2 * np.pi * minute / 1440), np.cos(2 * np.pi * minute / 1440),
    ]).astype(np.float32)
    values[~power_ok, 0] = np.nan
    if not features or features[0] != 'power_pu' or len(set(features)) != len(features):
        raise ValueError('Features must be unique and begin with power_pu')
    values = values[:, [FEATURES.index(f) for f in features]]
    input_ok = np.isfinite(values).all(axis=1)
    anchors, gap_count, candidates = [], 0, 0
    for _, group in frame.groupby('turbine_id', sort=False):
        positions = group.index.to_numpy()
        first, stop = int(positions[0]), int(positions[-1]) + 1
        a = np.arange(first + lookback - 1, stop - max(horizons))
        candidates += len(a)
        dt = group.timestamp.diff().to_numpy()
        # Prefix sums detect gaps inside context AND all steps up to the last label.
        breaks = np.r_[False, dt[1:] != np.timedelta64(10, 'm')]
        gap_count += int(breaks.sum())
        gap_prefix = np.r_[0, np.cumsum(breaks)]
        bad_prefix = np.r_[0, np.cumsum(~input_ok[first:stop])]
        lo, hi = a - first - lookback + 1, a - first
        contiguous = gap_prefix[hi + max(horizons) + 1] == gap_prefix[lo + 1]
        valid_context = bad_prefix[hi + 1] == bad_prefix[lo]
        valid_targets = power_ok[a[:, None] + np.array(horizons)].all(1)
        anchors.append(a[contiguous & valid_context & valid_targets])
    joined = np.concatenate(anchors)
    audit = dict(farm=farm, rows=len(frame), turbines=frame.turbine_id.nunique(),
                 start=str(frame.timestamp.min()), end=str(frame.timestamp.max()),
                 negative_power_rows=int((p < 0).sum()),
                 invalid_power_rows=int((~power_ok).sum()),
                 invalid_input_rows=int((~input_ok).sum()), time_gaps=gap_count,
                 possible_row_windows=candidates, valid_windows_stride1=len(joined),
                 feature_scales='fixed physical constants; no full-series fitting')
    return FarmData(farm, frame, values, joined, tuple(horizons), lookback, audit)
