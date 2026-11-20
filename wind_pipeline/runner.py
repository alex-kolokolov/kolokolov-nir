from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .adapters import build_adapter
from .baselines import evaluate_persistence, evaluate_ridge
from .config import ExperimentConfig
from .protocol import select_farms, temporal_train_val_split
from .schema import MODEL_FEATURES
from .windows import UniversalWindDataset


def prepare(config: ExperimentConfig) -> Path:
    frames = [build_adapter(item).load() for item in config.data]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sort_values(["dataset_id", "turbine_id", "timestamp"])
    output = experiment_dir(config)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "canonical.parquet"
    combined.to_parquet(path, index=False)
    manifest = (
        combined.groupby(["dataset_id", "farm_id"])
        .agg(rows=("timestamp", "size"), turbines=("turbine_id", "nunique"),
             start=("timestamp", "min"), end=("timestamp", "max"))
        .reset_index()
    )
    manifest.to_csv(output / "data_manifest.csv", index=False)
    return path


def run_baselines(config: ExperimentConfig) -> Path:
    frame = load_prepared(config)
    train_frame, evaluation_frame = protocol_frames(frame, config)
    train_data = make_dataset(train_frame, config, as_torch=False)
    evaluation_data = make_dataset(evaluation_frame, config, as_torch=False)
    result = pd.concat(
        [
            evaluate_persistence(evaluation_data),
            evaluate_ridge(train_data, evaluation_data),
        ],
        ignore_index=True,
    )
    result["experiment"] = config.name
    result["evaluation_farms"] = ",".join(
        sorted(evaluation_frame["farm_id"].astype(str).unique())
    )
    output = experiment_dir(config) / "baseline_results.csv"
    result.to_csv(output, index=False)
    return output


def train(config: ExperimentConfig) -> Path:
    from .training import evaluate, fit_model

    frame = load_prepared(config)
    training_domains = select_farms(frame, config.train_farms)
    if config.test_farms and not config.train_farms:
        training_domains = frame[~frame["farm_id"].isin(config.test_farms)].copy()
    train_frame, val_frame = temporal_train_val_split(
        training_domains, config.training.val_fraction
    )
    train_data = make_dataset(train_frame, config)
    val_data = make_dataset(val_frame, config)
    output = experiment_dir(config)
    model, _ = fit_model(
        train_data, val_data, config.model, config.training, output
    )
    metrics = evaluate(model, val_data, batch_size=config.training.batch_size)
    metrics.to_csv(output / "validation_metrics.csv", index=False)
    return output / "best_patchtst.pt"


def evaluate_checkpoint(config: ExperimentConfig, checkpoint: str | Path) -> Path:
    from .training import evaluate, load_model

    frame = load_prepared(config)
    _, evaluation_frame = protocol_frames(frame, config)
    dataset = make_dataset(evaluation_frame, config)
    model = load_model(checkpoint)
    result = evaluate(model, dataset, batch_size=config.training.batch_size)
    result["evaluation_farms"] = ",".join(
        sorted(evaluation_frame["farm_id"].astype(str).unique())
    )
    output = experiment_dir(config) / "transfer_metrics.csv"
    result.to_csv(output, index=False)
    return output


def protocol_frames(
    frame: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.test_farms:
        evaluation = select_farms(frame, config.test_farms)
        training = select_farms(frame, config.train_farms)
        if not config.train_farms:
            training = frame[~frame["farm_id"].isin(config.test_farms)].copy()
        if evaluation.empty:
            raise ValueError(f"No rows for test_farms={config.test_farms}")
        return training, evaluation
    return temporal_train_val_split(frame, config.training.val_fraction)


def make_dataset(
    frame: pd.DataFrame, config: ExperimentConfig, *, as_torch: bool = True
) -> UniversalWindDataset:
    features = config.window.features or MODEL_FEATURES
    resolutions = {item.sample_minutes for item in config.data}
    if len(resolutions) != 1:
        raise ValueError(
            "One experiment config must use one sampling interval. "
            "Run hourly GEFCom in a separate config from 10-minute SCADA."
        )
    sample_minutes = resolutions.pop()
    return UniversalWindDataset(
        frame,
        lookback=config.window.lookback,
        horizons=config.window.horizons,
        stride=config.window.stride,
        features=features,
        sample_minutes=sample_minutes,
        as_torch=as_torch,
    )


def load_prepared(config: ExperimentConfig) -> pd.DataFrame:
    path = experiment_dir(config) / "canonical.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run the prepare stage first")
    return pd.read_parquet(path)


def experiment_dir(config: ExperimentConfig) -> Path:
    return Path(config.output_dir) / config.name
