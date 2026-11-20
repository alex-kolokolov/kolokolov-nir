"""Features from past SCADA and fully separable validation-only decisions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss


POLICIES = (
    "persistence", "ses", "gru",
    "blend_persistence", "blend_ses",
    "activity_persistence", "activity_ses",
    "classifier_persistence", "classifier_ses",
)


def historical_features(power, wind, history_steps):
    """Use columns in increasing timestamp order, with the last column at t."""
    columns = {}
    for name, signal in [("power", power), ("wind", wind)]:
        columns[f"{name}_last"] = signal[:, -1]
        for steps in history_steps:
            values = signal[:, -(steps + 1):]
            increments = np.diff(values, axis=1)
            prefix = f"{name}_{steps * 10}m"
            columns[f"{prefix}_mean"] = values.mean(axis=1)
            columns[f"{prefix}_range"] = np.ptp(values, axis=1)
            columns[f"{prefix}_delta"] = values[:, -1] - values[:, 0]
            columns[f"{prefix}_activity"] = np.abs(increments).mean(axis=1)
            columns[f"{prefix}_max_jump"] = np.abs(increments).max(axis=1)
    return columns


def smoothing(power, alpha):
    """Simple exponential smoothing, initialized at the oldest input."""
    level = power[:, 0].copy()
    for column in power[:, 1:].T:
        level = alpha * column + (1 - alpha) * level
    return level


def ramp_labels(current, future, threshold):
    """Change at the forecast endpoint, not max swing within the horizon."""
    return np.abs(future - current) >= threshold


@dataclass(frozen=True)
class Switch:
    mode: str
    threshold: float | None = None

    def apply(self, score):
        if self.mode == "off":
            return np.zeros(score.shape, dtype=bool)
        if self.mode == "on":
            return np.ones(score.shape, dtype=bool)
        if self.mode != "threshold" or self.threshold is None:
            raise ValueError("Invalid switch")
        return score >= self.threshold


def switch_candidates(score, quantiles):
    yield Switch("off")
    for threshold in np.unique(np.quantile(score, quantiles))[::-1]:
        yield Switch("threshold", float(threshold))
    yield Switch("on")


def event_metrics(labels, detected):
    tp = int(np.count_nonzero(detected & labels))
    fp = int(np.count_nonzero(detected & ~labels))
    fn = int(np.count_nonzero(~detected & labels))
    tn = len(labels) - tp - fp - fn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def fit_event_switch(score, labels, quantiles):
    return max(switch_candidates(score, quantiles), key=lambda s: event_metrics(labels, s.apply(score))["f1"])


def score_detector(labels, score, trigger, *, probability=False):
    return {
        **event_metrics(labels, trigger),
        "prevalence": float(labels.mean()),
        "average_precision": float(average_precision_score(labels, score)) if labels.any() else None,
        "brier": float(brier_score_loss(labels, score)) if probability else None,
    }


def fit_forecast_switch(calm, fast, truth, score, quantiles):
    quiet_loss = np.abs(calm - truth)
    fast_loss = np.abs(fast - truth)
    return min(switch_candidates(score, quantiles),
               key=lambda s: np.where(s.apply(score), fast_loss, quiet_loss).mean())


def fit_policies(power, fast, truth, probability, activity, spec):
    """All inputs to this function are from the source calibration period."""
    last = power[:, -1]
    alpha = min(spec["alpha_grid"], key=lambda a: np.abs(smoothing(power, a) - truth).mean())
    calm = {"persistence": last, "ses": smoothing(power, alpha)}
    params = {
        "alpha": alpha,
        "blend_weights": {},
        "activity": {},
        "classifier": {},
    }
    for name, level in calm.items():
        params["blend_weights"][name] = min(
            spec["blend_weights"], key=lambda weight: np.abs(level + weight * (fast - level) - truth).mean()
        )
        for score_name, score in [("activity", activity), ("classifier", probability)]:
            params[score_name][name] = fit_forecast_switch(
                level, fast, truth, score, spec["switch_quantiles"]
            )
    return params


def apply_policies(power, fast, probability, activity, params):
    """Inference has no target or observed ramp label."""
    forecasts = {"persistence": power[:, -1], "ses": smoothing(power, params["alpha"]), "gru": fast}
    switches = {}
    for name in ("persistence", "ses"):
        calm = forecasts[name]
        forecasts[f"blend_{name}"] = calm + params["blend_weights"][name] * (fast - calm)
        for score_name, score in [("activity", activity), ("classifier", probability)]:
            policy = f"{score_name}_{name}"
            use_fast = params[score_name][name].apply(score)
            forecasts[policy] = np.where(use_fast, fast, calm)
            switches[policy] = use_fast
    return forecasts, switches
