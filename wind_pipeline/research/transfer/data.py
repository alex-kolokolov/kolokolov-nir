"""Regular time grids, causal covariates, and disjoint replay windows."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .scada import TIME, TURBINE
from .weather import WEATHER_FEATURES, align_forecasts, time_minutes

CORE = ('power_pu', 'wind', 'direction_sin', 'direction_cos', 'rotor', 'ambient', 'nacelle', 'hour_sin', 'hour_cos')
EXTRA = ('wind_sd', 'yaw_error_sin', 'yaw_error_cos', 'pitch', 'setpoint_pu', 'rotor_active',
         'neighbour_wind', 'neighbour_power')
FEATURES = CORE + EXTRA


def bounded(values, lower, upper):
    return np.where((values >= lower) & (values <= upper), values, np.nan)


def neighbour_means(values, coordinates, depth=3):
    """Summarize other turbines using only their samples at the same timestamp."""
    result = np.full_like(values, np.nan)
    if len(coordinates) < 2:
        return result
    points = coordinates.copy()
    points[:, 1] *= np.cos(np.deg2rad(points[:, 0].mean()))
    distance = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argsort(distance, axis=1, kind='stable')[:, :min(depth, len(points) - 1)]
    for i, neighbours in enumerate(nearest):
        other = values[:, neighbours]
        count = np.isfinite(other).sum(1)
        np.divide(np.nansum(other, axis=1), count, out=result[:, i], where=count > 0)
    return result


def numeric_grid(frame, inventory):
    turbines = inventory[TURBINE].tolist()
    rated = inventory.rated_kw.to_numpy(np.float32)
    times = pd.date_range(frame[TIME].min(), frame[TIME].max(), freq='10min')
    grid = frame.set_index([TIME, TURBINE]).reindex(pd.MultiIndex.from_product([times, turbines]))
    shape = (len(times), len(turbines))

    def raw(name):
        return grid[name].to_numpy(dtype=np.float32).reshape(shape)

    power = bounded(raw('power_kw') / rated, -0.05, 1.1).clip(min=0)
    wind = bounded(raw('wind_ms'), 0, 60) / 15
    direction = np.deg2rad(bounded(raw('wind_direction_deg'), -360, 360))
    rotor = bounded(raw('rotor_rpm'), 0, 60)
    hour = np.broadcast_to((time_minutes(times) % 1440 / 1440 * 2 * np.pi)[:, None], shape)
    yaw = np.deg2rad(bounded(raw('yaw_deg'), -360, 360))
    pitch = bounded(raw('pitch_deg'), -360, 360)
    pitch = bounded((pitch + 180) % 360 - 180, -30, 100)
    active = np.where(np.isfinite(rotor), (rotor >= 1).astype('float32'), np.nan)
    coordinates = inventory[['latitude', 'longitude']].to_numpy(dtype=float)
    arrays = [power, wind, np.sin(direction), np.cos(direction), rotor / 20,
              bounded(raw('ambient_c'), -60, 100) / 40, bounded(raw('nacelle_c'), -60, 100) / 40,
              np.sin(hour), np.cos(hour), bounded(raw('wind_sd_ms'), 0, 20) / 5,
              np.sin(direction - yaw), np.cos(direction - yaw), pitch / 45,
              bounded(raw('setpoint_kw') / rated, -0.05, 1.1).clip(min=0), active,
              neighbour_means(wind, coordinates), neighbour_means(power, coordinates)]
    return time_minutes(times), np.stack(arrays, axis=-1, dtype=np.float32)


def write_grid(config, farm, frame, inventory, forecasts, path):
    times, values = numeric_grid(frame, inventory)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / 'times.npy', times, allow_pickle=False)
    np.save(path / 'scada.npy', values, allow_pickle=False)
    inventory.to_parquet(path / 'turbines.parquet', index=False)
    observed = np.isfinite(values)
    audit = {'farm': farm, 'observations': len(frame), 'grid_rows': int(np.prod(values.shape[:2])),
             'turbines': len(inventory), 'start': str(np.datetime64(int(times[0]), 'm')),
             'end_inclusive': str(np.datetime64(int(times[-1]), 'm')),
             **{name: float(observed[:, :, index].mean()) for index, name in enumerate(FEATURES)}}
    if forecasts is not None:
        weather, runs = align_forecasts(forecasts.loc[forecasts.farm == farm],
                                        times.astype('datetime64[m]'), config.horizons,
                                        config['weather']['publication_delay_hours'])
        coverage = float(np.isfinite(weather).all(axis=(1, 2)).mean())
        if coverage < config['weather']['minimum_coverage']:
            raise ValueError(f'Weather coverage for {farm} is {coverage:.1%}; review the archive before training')
        np.save(path / 'weather.npy', weather, allow_pickle=False)
        np.save(path / 'weather_runs.npy', runs, allow_pickle=False)
        audit['weather_coverage'] = coverage
    return audit


@dataclass(frozen=True)
class FarmSeries:
    name: str
    times: np.ndarray
    scada: np.ndarray
    weather: np.ndarray | None
    weather_runs: np.ndarray | None
    turbines: pd.DataFrame

    @classmethod
    def load(cls, config, name):
        path = config.output / 'prepared' / name
        weather = np.load(path / 'weather.npy', mmap_mode='r') if config['weather'] else None
        runs = np.load(path / 'weather_runs.npy', mmap_mode='r') if config['weather'] else None
        return cls(name, np.load(path / 'times.npy', mmap_mode='r'), np.load(path / 'scada.npy', mmap_mode='r'),
                   weather, runs, pd.read_parquet(path / 'turbines.parquet'))

    def select(self, config, start, end, stride):
        """Require observed power at the origin and at least one exact future horizon."""
        offsets = np.array(config.horizons, dtype=np.int64) // 10
        lookback = config['history_minutes'] // 10
        start_min, end_min = time_minutes([pd.Timestamp(start), pd.Timestamp(end)])
        lo = int(np.searchsorted(self.times, start_min)) + lookback - 1
        hi = int(np.searchsorted(self.times, end_min))
        indices = np.arange(lo, hi)
        indices = indices[self.times[indices] % stride == 0]
        future = indices[:, None] + offsets
        within = future < hi
        targets = self.scada[np.minimum(future, len(self.times) - 1), :, 0].copy()
        targets[~within] = np.nan
        allowed = np.isfinite(self.scada[indices, :, 0]) & np.isfinite(targets).any(axis=1)
        local_time, turbine = np.nonzero(allowed)
        if not len(turbine):
            raise ValueError(f'No windows at {self.name} between {start} and {end}')
        return Windows(self, lookback, indices[local_time], turbine, targets[local_time, :, turbine])


@dataclass(frozen=True)
class Windows:
    farm: FarmSeries
    lookback: int
    time_index: np.ndarray
    turbine_index: np.ndarray
    targets: np.ndarray

    def __len__(self):
        return len(self.time_index)

    @property
    def times(self):
        return self.farm.times[self.time_index].astype('datetime64[m]')

    @property
    def ratings(self):
        return self.farm.turbines.rated_kw.to_numpy()[self.turbine_index]

    @property
    def persistence(self):
        return self.farm.scada[self.time_index, self.turbine_index, 0]

    def inputs(self, ids, variant):
        ncols = len(CORE) if variant == 'core' else len(FEATURES)
        offsets = np.arange(1 - self.lookback, 1)
        time = self.time_index[ids]
        history = self.farm.scada[time[:, None] + offsets, self.turbine_index[ids, None], :ncols]
        future = self.farm.weather[time] if variant == 'weather' else None
        return history, future

    def ses(self, alpha):
        power = self.farm.scada[self.time_index[:, None] + np.arange(-11, 1), self.turbine_index[:, None], 0]
        level = np.full(len(self), np.nan, dtype=np.float32)
        for point in power.T:
            level = np.where(np.isfinite(point), np.where(np.isfinite(level), alpha * point + (1 - alpha) * level, point), level)
        return level


def summaries(history, variant):
    """Multi-scale statistics on measured power and wind with no look-ahead."""
    history = history[:, :, :2]
    sizes = (history.shape[1],) if variant == 'core' else (3, 6, 12, history.shape[1])
    result = []
    for size in sizes:
        local = history[:, -size:]
        count = np.isfinite(local).sum(axis=1)
        denom = np.maximum(count, 1)
        mean = np.nansum(local, axis=1) / denom
        std = np.sqrt(np.nansum((local - mean[:, None, :]) ** 2, axis=1) / denom)
        mean[count == 0] = np.nan
        std[count == 0] = np.nan
        result.extend([mean, std, local[:, -1] - local[:, 0]])
    return np.concatenate(result, axis=1).astype(np.float32)


def tabular(windows, ids, variant, horizon_index):
    """Direct regression inputs. Numeric NaNs are handled by CatBoost."""
    lag_offsets = sorted({1, 2, 3, 6, 12, 24, windows.lookback} & set(range(1, windows.lookback + 1)))
    output = []
    for start in range(0, len(ids), 4096):
        history, forecast = windows.inputs(ids[start:start + 4096], variant)
        lagged = history[:, -np.array(lag_offsets)].reshape(len(history), -1)
        parts = [lagged, summaries(history, variant)]
        if forecast is not None:
            parts.append(forecast[:, horizon_index, :len(WEATHER_FEATURES)])
        output.append(np.concatenate(parts, axis=1))
    return np.concatenate(output)
