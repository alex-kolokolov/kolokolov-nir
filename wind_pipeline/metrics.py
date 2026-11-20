from __future__ import annotations

import numpy as np


def regression_metrics(
    y_true_pu: np.ndarray,
    y_pred_pu: np.ndarray,
    rated_power_kw: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true_pu, dtype=float)
    y_pred = np.asarray(y_pred_pu, dtype=float)
    rated = np.asarray(rated_power_kw, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(rated)
    if not valid.any():
        raise ValueError("No finite observations for metrics")
    y_true, y_pred, rated = y_true[valid], y_pred[valid], rated[valid]
    error_kw = (y_pred - y_true) * rated
    residual = y_true - y_pred
    denominator = np.square(y_true - y_true.mean()).sum()
    r2 = float("nan") if denominator == 0 else 1.0 - np.square(residual).sum() / denominator
    return {
        "mae_kw": float(np.abs(error_kw).mean()),
        "rmse_kw": float(np.sqrt(np.square(error_kw).mean())),
        "nmae_pct": float(np.abs(residual).mean() * 100.0),
        "r2": float(r2),
    }


def grouped_metric_rows(
    *,
    model: str,
    horizons: tuple[int, ...],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rated_power_kw: np.ndarray,
    farms: np.ndarray,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    farm_labels = farms.astype(str)
    unique_farms = np.unique(farm_labels)
    for column, horizon in enumerate(horizons):
        farm_metrics = []
        for farm in unique_farms:
            mask = farm_labels == farm
            metrics = regression_metrics(
                y_true[mask, column], y_pred[mask, column], rated_power_kw[mask]
            )
            farm_metrics.append(metrics)
            rows.append({
                "model": model,
                "horizon_steps": horizon,
                "scope": "farm",
                "farm_id": farm,
                **metrics,
            })
        rows.append({
            "model": model,
            "horizon_steps": horizon,
            "scope": "micro",
            "farm_id": "ALL",
            **regression_metrics(y_true[:, column], y_pred[:, column], rated_power_kw),
        })
        rows.append({
            "model": model,
            "horizon_steps": horizon,
            "scope": "macro_farm",
            "farm_id": "ALL",
            **{
                key: float(np.nanmean([metrics[key] for metrics in farm_metrics]))
                for key in ("mae_kw", "rmse_kw", "nmae_pct", "r2")
            },
        })
    return rows
