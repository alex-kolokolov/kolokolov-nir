from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from wind_pipeline.config import DataConfig


class DatasetAdapter(ABC):
    def __init__(self, config: DataConfig):
        self.config = config

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return a canonical, feature-enriched frame."""

    def read_source(self) -> pd.DataFrame:
        path = Path(self.config.path)
        suffix = path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        raise ValueError(f"Unsupported data format: {path}")


def build_adapter(config: DataConfig) -> DatasetAdapter:
    if config.adapter == "generic":
        from .generic import GenericAdapter

        return GenericAdapter(config)
    if config.adapter == "legacy_scada":
        from .legacy_scada import LegacyScadaAdapter

        return LegacyScadaAdapter(config)
    if config.adapter == "sdwpf":
        from .sdwpf import SdwpfAdapter

        return SdwpfAdapter(config)
    raise ValueError(f"Unknown adapter: {config.adapter!r}")

