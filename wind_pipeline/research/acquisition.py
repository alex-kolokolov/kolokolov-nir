"""Version-pinned acquisition and ingestion of original Cubico SCADA archives."""
from __future__ import annotations

import csv
import hashlib
import io
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from ..research_data import RAW_COLUMNS
from .artifacts import file_hash, write_json


def md5(path):
    h = hashlib.md5()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download_one(config, record, item):
    directory = config.root / config['raw_sources']['directory'] / record['farm']
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory/item['name']
    url = f"https://zenodo.org/api/records/{record['record_id']}/files/{item['name']}/content"
    if destination.exists():
        if md5(destination) != item['md5']:
            raise ValueError(f'Existing raw file checksum mismatch: {destination}')
        print(f'Raw file verified: {item["name"]}', flush=True)
        return destination
    partial = destination.with_suffix(destination.suffix+'.part')
    headers = {'User-Agent': 'NIR-wind-research/1.0 (academic dataset download)', 'Range': 'bytes=0-0'}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
        content_range = response.headers.get('Content-Range')
        if not content_range:
            raise ValueError('Raw-data server must support byte ranges for resumable downloads')
        total_size = int(content_range.split('/')[-1])
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > total_size:
                raise ValueError('Partial download is larger than the pinned remote file')
            print(f'Downloading {item["name"]} from {offset//1024//1024}/{total_size//1024//1024} MiB', flush=True)
            while offset < total_size:
                end = min(offset + 32*1024*1024, total_size)-1
                headers['Range'] = f'bytes={offset}-{end}'
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Unexpected download resume range')
                    received = 0
                    with partial.open('ab') as stream:
                        while block := response.read(1024*1024):
                            stream.write(block)
                            received += len(block)
                    if received != end-offset+1:
                        raise ValueError('Truncated download range')
                offset = partial.stat().st_size
                print(f'{item["name"]}: {offset//1024//1024}/{total_size//1024//1024} MiB', flush=True)
            if md5(partial) != item['md5']:
                raise ValueError(f'Download checksum mismatch: {partial}')
            partial.replace(destination)
            print(f'Download verified: {item["name"]}', flush=True)
            return destination
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3 * (attempt+1))
    raise RuntimeError('Unreachable download state')


def fetch(config):
    items = [(record, item) for record in config['raw_sources']['records'] for item in record['files']]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(download_one, config, record, item) for record,item in items]
        for future in futures:
            future.result()


def normalize_column(name):
    return re.sub(r'[^a-z0-9]+', '_', name.lstrip('\ufeff# ').lower()).strip('_')


def read_member(stream, farm, name):
    """Read named fields after exporter comments; reject positional guesses."""
    text = io.TextIOWrapper(stream, encoding='utf-8-sig')
    for _ in range(1000):
        line = text.readline()
        if not line:
            raise ValueError(f'No SCADA header in {name}')
        if line.lower().startswith('# time zone:') and line.split(':',1)[1].strip() != 'UTC':
            raise ValueError(f'Unsupported source time zone in {name}')
        header = next(csv.reader([line]))
        time_columns = [c for c in header if normalize_column(c) in ('date_and_time', 'timestamp', 'date_time')]
        if len(time_columns) == 1 and len(header) > 2:
            break
    else:
        raise ValueError(f'No recognized header in {name}')
    normalized = {normalize_column(c): c for c in header}
    mapping = {}
    for canonical in RAW_COLUMNS[3:]:
        key = normalize_column(canonical)
        if key not in normalized:
            raise ValueError(f'Missing named SCADA signal {canonical!r} in {name}; columns={header}')
        mapping[normalized[key]] = canonical
    availability = normalized.get('data_availability')
    if availability is None:
        raise ValueError(f'Missing Data Availability signal in {name}')
    time_column = time_columns[0]
    selected = [time_column, availability] + list(mapping)
    frame = pd.read_csv(text, names=header, usecols=selected, low_memory=False)
    frame = frame.rename(columns={time_column: 'timestamp', availability: 'data_availability', **mapping})
    match = re.search(rf'(?:Turbine_Data_)?{re.escape(farm)}[_ -]+(?:T|WT)?0*(\d+)', Path(name).name, re.I)
    if not match:
        match = re.search(r'(?:Turbine_Data|Turbine|WT|T)[_ -]*0*(\d+)', Path(name).name, re.I)
    if not match:
        raise ValueError(f'Cannot identify turbine from source filename {name}')
    frame['turbine_id'] = f'{farm}_T{int(match.group(1)):02d}'
    frame['farm'] = farm
    # Source exporter timestamps are interpreted in UTC, with no local DST ambiguity.
    frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True, errors='raise').dt.tz_localize(None)
    frame['source_file'] = name
    return frame


def ingest(config):
    """Rebuild the canonical table from original files, retaining missing timestamps."""
    config.output.mkdir(parents=True, exist_ok=True)
    frames, sources, coverage = [], [], []
    lower, upper = pd.Timestamp(config['train_start']), pd.Timestamp(config['test_end'])
    for record in config['raw_sources']['records']:
        farm = record['farm']
        for item in record['files']:
            path = config.root/config['raw_sources']['directory']/farm/item['name']
            if not path.exists():
                raise FileNotFoundError(f'Raw source absent: {path}; run research --step fetch')
            if md5(path) != item['md5']:
                raise ValueError(f'Raw source checksum mismatch: {path}')
            sources.append(dict(farm=farm, filename=item['name'], md5=item['md5'],
                                sha256=file_hash(path), record_id=record['record_id']))
            if path.suffix.lower() == '.csv':
                static = pd.read_csv(path, encoding='utf-8-sig')
                powers = pd.to_numeric(static['Rated power (kW)'], errors='coerce').dropna()
                if len(powers) == 0 or not np.allclose(powers, config['rated_power_kw']):
                    raise ValueError('Configured rated power disagrees with the original turbine passport')
                continue
            with ZipFile(path) as archive:
                for member in sorted(archive.namelist()):
                    if not member.lower().endswith('.csv') or 'turbine_data' not in member.lower():
                        continue
                    with archive.open(member) as stream:
                        frame = read_member(stream, farm, member)
                    frame = frame[(frame.timestamp >= lower) & (frame.timestamp < upper)].copy()
                    if frame.empty:
                        continue
                    for col in RAW_COLUMNS[3:] + ('data_availability',):
                        frame[col] = pd.to_numeric(frame[col], errors='coerce')
                    unavailable = frame.data_availability < config['raw_sources']['availability_minimum']
                    unavailable |= frame.data_availability.isna()
                    # Preserve the timeline. Invalid measurements become missing, not removed rows.
                    frame.loc[unavailable, list(RAW_COLUMNS[3:])] = np.nan
                    coverage.append(dict(farm=farm, member=member, rows=len(frame),
                                         unavailable_rows=int(unavailable.sum()),
                                         availability_min=frame.data_availability.min(),
                                         availability_max=frame.data_availability.max()))
                    frames.append(frame)
                    print(f'Ingested {farm}/{Path(member).name}: {len(frame):,} rows', flush=True)
    if not frames:
        raise ValueError('No source CSV members matched the configured date range')
    combined = pd.concat(frames, ignore_index=True).sort_values(['farm','turbine_id','timestamp'])
    keys = ['farm','turbine_id','timestamp']
    duplicates = combined.duplicated(keys, keep=False)
    # Year boundaries can repeat a record, but conflicting measurements must be resolved explicitly.
    if duplicates.any():
        conflicts = combined.loc[duplicates].groupby(keys)[list(RAW_COLUMNS[3:])].nunique(dropna=False)
        if (conflicts > 1).any().any():
            raise ValueError('Conflicting duplicate raw measurements; refusing silent deduplication')
        combined = combined.drop_duplicates(keys)
    missing = set(config['farms']) - set(combined.farm.unique())
    if missing:
        raise ValueError(f'Missing farms in raw sources: {missing}')
    temporary = config.source.with_suffix('.parquet.tmp')
    config.source.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(temporary, index=False)
    temporary.replace(config.source)
    pd.DataFrame(coverage).to_csv(config.output/'raw_coverage.csv', index=False)
    write_json(config.output/'raw_manifest.json', dict(sources=sources, rows=len(combined),
               duplicated_rows=int(duplicates.sum()), timezone='UTC', canonical_sha256=file_hash(config.source),
               availability_policy='mask measurements, preserve timestamps',
               source_versions={r['farm']: r['record_id'] for r in config['raw_sources']['records']}))
