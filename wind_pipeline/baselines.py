from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .metrics import grouped_metric_rows
from .windows import UniversalWindDataset


def evaluate_persistence(dataset: UniversalWindDataset) -> pd.DataFrame:
    power_idx = dataset.features.index("power_pu")
    truths, predictions, rated = _collect_persistence(dataset, power_idx)
    return pd.DataFrame(grouped_metric_rows(
        model="Persistence",
        horizons=dataset.horizons,
        y_true=truths,
        y_pred=predictions,
        rated_power_kw=rated,
        farms=dataset.window_farms,
    ))


def evaluate_ridge(
    train: UniversalWindDataset,
    evaluation: UniversalWindDataset,
    *,
    alpha: float = 1.0,
    max_train_windows: int = 200_000,
) -> pd.DataFrame:
    try:
        from sklearn.linear_model import Ridge
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("Ridge baseline requires scikit-learn") from exc

    train_ids = _even_indices(len(train), max_train_windows)
    x_train, y_train, _ = _summary_arrays(train, train_ids)
    x_eval, y_eval, rated = _summary_arrays(evaluation, range(len(evaluation)))
    predictions = []
    for column, horizon in enumerate(train.horizons):
        model = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0),
            StandardScaler(),
            Ridge(alpha=alpha),
        )
        model.fit(x_train, y_train[:, column])
        predictions.append(model.predict(x_eval))
    prediction_matrix = np.column_stack(predictions)
    return pd.DataFrame(grouped_metric_rows(
        model="Ridge",
        horizons=evaluation.horizons,
        y_true=y_eval,
        y_pred=prediction_matrix,
        rated_power_kw=rated,
        farms=evaluation.window_farms,
    ))


def _collect_persistence(
    dataset: UniversalWindDataset, power_idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth, prediction, rated = [], [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        value = float(sample["x"][-1, power_idx])
        truth.append(np.asarray(sample["y"]))
        prediction.append(np.repeat(value, len(dataset.horizons)))
        rated.append(float(sample["rated_power_kw"]))
    return np.asarray(truth), np.asarray(prediction), np.asarray(rated)


def _summary_arrays(
    dataset: UniversalWindDataset, indices: Iterable[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, targets, rated = [], [], []
    for i in indices:
        sample = dataset[i]
        x = np.asarray(sample["x"], dtype=float)
        mask = np.asarray(sample["x_mask"], dtype=bool)
        masked = np.where(mask, x, np.nan)
        with np.errstate(invalid="ignore"):
            summary = np.concatenate([
                masked[-1],
                np.nanmean(masked, axis=0),
                np.nanstd(masked, axis=0),
                masked[-1] - masked[0],
                mask.mean(axis=0),
            ])
        features.append(summary)
        targets.append(np.asarray(sample["y"]))
        rated.append(float(sample["rated_power_kw"]))
    return np.asarray(features), np.asarray(targets), np.asarray(rated)


def _even_indices(length: int, maximum: int) -> range | np.ndarray:
    if length <= maximum:
        return range(length)
    return np.linspace(0, length - 1, maximum, dtype=int)
