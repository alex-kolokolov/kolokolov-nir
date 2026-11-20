"""Extract point forecasts from the operational IFS HRES archive in WeatherBench 2."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from numcodecs import get_codec
from urllib3.util.retry import Retry

from ..artifacts import object_hash, write_json

COMPONENTS = ('10m_u_component_of_wind', '10m_v_component_of_wind')
WEATHER_FEATURES = ('weather_u', 'weather_v', 'weather_speed', 'weather_change', 'weather_age')
LEADS = (6, 12, 18, 24, 30, 36, 42)
ARCHIVE_COLUMNS = ('farm', 'run_time', 'valid_time', 'u_ms', 'v_ms')


def time_minutes(values):
    return np.asarray(values, dtype='datetime64[m]').astype('int64')


class HresArchive:
    def __init__(self, settings, cache, coordinates):
        self.url = settings['url'].rstrip('/')
        self.workers = settings['download_workers']
        self.thread = threading.local()
        self.cache = cache / object_hash({'url': self.url, 'schema': settings['metadata_sha256'],
                                          'coordinates': coordinates, 'leads': LEADS,
                                          'interpolation': 'bilinear'})[:20]
        self.cache.mkdir(parents=True, exist_ok=True)
        raw = self.small_object('.zmetadata')
        if hashlib.sha256(raw).hexdigest() != settings['metadata_sha256']:
            raise ValueError('IFS Zarr metadata hash does not match the pinned archive')
        self.metadata = json.loads(raw)['metadata']
        time_unit = self.metadata['time/.zattrs']['units']
        if not time_unit.startswith('hours since ') or self.metadata['prediction_timedelta/.zattrs']['units'] != 'hours':
            raise ValueError('Unexpected IFS time units')
        self.times = pd.Timestamp(time_unit.removeprefix('hours since ')) + pd.to_timedelta(self.coordinate('time'), unit='h')
        leads = self.coordinate('prediction_timedelta')
        self.lead_indices = [int(np.flatnonzero(leads == lead).item()) for lead in LEADS]
        self.lat = self.coordinate('latitude')
        self.lon = self.coordinate('longitude')
        self.locations = coordinates
        self.grids = {name: self.locate(lat, lon % 360) for name, (lat, lon) in coordinates.items()}
        for variable in COMPONENTS:
            spec = self.spec(variable)
            axes = self.metadata[f'{variable}/.zattrs']['_ARRAY_DIMENSIONS']
            if axes != ['time', 'prediction_timedelta', 'longitude', 'latitude']:
                raise ValueError(f'Unexpected IFS axis order for {variable}: {axes}')
            if spec['chunks'] != [1, 8, len(self.lon), len(self.lat)] or max(self.lead_indices) >= 8:
                raise ValueError(f'Unexpected IFS chunk layout for {variable}')

    def request(self, key):
        if not hasattr(self.thread, 'session'):
            session = requests.Session()
            session.mount('https://', requests.adapters.HTTPAdapter(max_retries=Retry(
                total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))))
            self.thread.session = session
        response = self.thread.session.get(f'{self.url}/{key}', timeout=(15, 60))
        response.raise_for_status()
        return response.content, response.headers.get('x-goog-generation')

    def small_object(self, key):
        path = self.cache / key.replace('/', '_')
        if path.is_file():
            return path.read_bytes()
        raw, _ = self.request(key)
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_bytes(raw)
        temporary.replace(path)
        return raw

    def spec(self, variable):
        spec = self.metadata[f'{variable}/.zarray']
        if spec['zarr_format'] != 2 or spec['filters'] is not None or spec['order'] != 'C':
            raise ValueError(f'Unsupported IFS encoding for {variable}')
        return spec

    def decode(self, variable, raw):
        spec = self.spec(variable)
        unpacked = get_codec(spec['compressor']).decode(raw)
        return np.frombuffer(unpacked, dtype=spec['dtype']).reshape(spec['chunks'])

    def coordinate(self, variable):
        spec = self.spec(variable)
        if spec['shape'] != spec['chunks']:
            raise ValueError(f'Expected one coordinate chunk for {variable}')
        return self.decode(variable, self.small_object(f'{variable}/0'))

    def locate(self, lat, lon):
        def bracket(grid, value):
            order = np.argsort(grid)
            sorted_grid = grid[order]
            high = int(np.searchsorted(sorted_grid, value, side='right'))
            if not 0 < high < len(grid):
                raise ValueError(f'Farm coordinate {value} lies outside the interpolation grid')
            low = high - 1
            upper_weight = (value - sorted_grid[low]) / (sorted_grid[high] - sorted_grid[low])
            return [(int(order[low]), 1 - upper_weight), (int(order[high]), upper_weight)]

        longitude = ((self.lon - lon + 180) % 360) - 180
        return [(j, i, w_lat * w_lon) for i, w_lat in bracket(self.lat, lat)
                for j, w_lon in bracket(longitude, 0)]

    def cycle(self, index):
        run = self.times[index]
        path = self.cache / f'{run:%Y%m%d%H}.json'
        if path.is_file():
            saved = json.loads(path.read_text(encoding='utf-8'))
            if saved['sha256'] != object_hash(saved['sample']) or saved['sample']['run_time'] != str(run):
                raise ValueError(f'IFS cache integrity check failed for {path}')
            return saved['sample']
        sample = {'run_time': str(run), 'source_chunks': {}, 'sites': {}}
        values = {}
        for variable in COMPONENTS:
            key = f'{variable}/{index}.0.0.0'
            raw, generation = self.request(key)
            sample['source_chunks'][key] = {'sha256': hashlib.sha256(raw).hexdigest(), 'generation': generation}
            values[variable] = self.decode(variable, raw)[0, self.lead_indices]
        for name, points in self.grids.items():
            uv = [sum(weight * values[v][:, j, i] for j, i, weight in points) for v in COMPONENTS]
            if not np.isfinite(uv).all():
                raise ValueError(f'Non-finite IFS wind at {name}, run {run}')
            sample['sites'][name] = np.stack(uv, axis=-1).tolist()
        write_json(path, {'sha256': object_hash(sample), 'sample': sample})
        return sample

    def indices(self, start, end):
        mask = (self.times >= pd.Timestamp(start) - pd.Timedelta(days=2)) & (self.times < pd.Timestamp(end))
        indices = np.flatnonzero(mask)
        if not len(indices):
            raise ValueError('IFS archive contains no runs for the configured interval')
        return indices

    def fetch(self, start, end):
        indices = self.indices(start, end)
        print(f'IFS archive: {len(indices)} cycles for {len(self.locations)} farms', flush=True)
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for count, _ in enumerate(pool.map(self.cycle, indices), 1):
                if count % 100 == 0 or count == len(indices):
                    print(f'IFS point forecasts: {count}/{len(indices)} cycles ({time.monotonic() - started:.0f} s)', flush=True)

    def station_table(self, start, end):
        samples = []
        for index in self.indices(start, end):
            path = self.cache / f'{self.times[index]:%Y%m%d%H}.json'
            if not path.is_file():
                raise FileNotFoundError(f'IFS point cache is incomplete at {path.name}; run research --step fetch')
            samples.append(self.cycle(index))
        rows = []
        for sample in samples:
            run = pd.Timestamp(sample['run_time'])
            for farm, forecasts in sample['sites'].items():
                for lead, (u, v) in zip(LEADS, forecasts, strict=True):
                    rows.append((farm, run, run + pd.Timedelta(hours=lead), u, v))
        record = {sample['run_time']: sample['source_chunks'] for sample in samples}
        return pd.DataFrame(rows, columns=ARCHIVE_COLUMNS), record


def align_forecasts(forecasts, origins, horizons, delay_hours):
    """Use one observable run per origin. Interpolate only inside that run."""
    table = forecasts.sort_values(['run_time', 'valid_time'])
    if table.duplicated(['run_time', 'valid_time']).any() or (table.valid_time <= table.run_time).any():
        raise ValueError('IFS table requires unique, positive forecast lead times')
    origin = time_minutes(origins)
    groups = table.groupby('run_time', sort=True)
    cycles = time_minutes(list(groups.groups))
    if not len(cycles):
        raise ValueError('Weather archive has no forecast cycles for this farm')
    selected = np.searchsorted(cycles, origin - 60 * delay_hours, side='right') - 1
    features = np.full((len(origin), len(horizons), len(WEATHER_FEATURES)), np.nan, np.float32)
    run_used = np.full(len(origin), np.datetime64('NaT', 'm'), dtype='datetime64[m]')
    for index, (_, run) in enumerate(groups):
        ids = np.flatnonzero((selected == index) & (origin - cycles[index] <= 24 * 60 + 60 * delay_hours))
        if not len(ids):
            continue
        valid = time_minutes(run.valid_time)
        valid_future = origin[ids, None] + np.asarray(horizons)[None, :]
        u = np.interp(valid_future, valid, run.u_ms, left=np.nan, right=np.nan)
        v = np.interp(valid_future, valid, run.v_ms, left=np.nan, right=np.nan)
        base_u = np.interp(origin[ids], valid, run.u_ms, left=np.nan, right=np.nan)
        base_v = np.interp(origin[ids], valid, run.v_ms, left=np.nan, right=np.nan)
        speed = np.hypot(u, v)
        age = np.broadcast_to(((origin[ids] - cycles[index]) / (24 * 60))[:, None], u.shape)
        features[ids] = np.stack([u / 15, v / 15, speed / 15,
                                  (speed - np.hypot(base_u, base_v)[:, None]) / 15, age], axis=-1)
        run_used[ids] = np.datetime64(int(cycles[index]), 'm')
    return features, run_used
