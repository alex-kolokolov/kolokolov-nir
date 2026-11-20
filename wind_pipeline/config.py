from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    adapter: str
    path: str
    dataset_id: str
    farm_id: str | None = None
    sample_minutes: int = 10
    rated_power_kw: float | dict[str, float] | None = None
    columns: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class WindowConfig:
    lookback: int = 96
    horizons: tuple[int, ...] = (1, 3, 6, 12, 24)
    stride: int = 1
    features: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelConfig:
    patch_length: int = 16
    patch_stride: int = 8
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 3
    dropout: float = 0.1
    sensor_dropout: float = 0.3


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    val_fraction: float = 0.2
    seed: int = 42
    balance_domains: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    data: tuple[DataConfig, ...]
    window: WindowConfig
    output_dir: str = "artifacts"
    train_farms: tuple[str, ...] = ()
    test_farms: tuple[str, ...] = ()
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    data = tuple(DataConfig(**item) for item in payload["data"])
    raw_window = dict(payload.get("window", {}))
    raw_window["horizons"] = tuple(raw_window.get("horizons", (1, 3, 6, 12, 24)))
    raw_window["features"] = tuple(raw_window.get("features", ()))
    return ExperimentConfig(
        name=payload["name"],
        data=data,
        window=WindowConfig(**raw_window),
        output_dir=payload.get("output_dir", "artifacts"),
        train_farms=tuple(payload.get("train_farms", ())),
        test_farms=tuple(payload.get("test_farms", ())),
        model=ModelConfig(**payload.get("model", {})),
        training=TrainingConfig(**payload.get("training", {})),
    )
