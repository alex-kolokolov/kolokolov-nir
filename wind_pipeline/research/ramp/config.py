"""Separate run specification; the parent forecasting run is read-only."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import ResearchConfig, load_research_config


@dataclass(frozen=True)
class RampConfig:
    parent: ResearchConfig
    output: Path
    spec: dict

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(self.spec["horizons_minutes"])


def load_ramp_config(path: str | Path) -> RampConfig:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "base_config", "output_dir", "horizons_minutes", "primary_horizon_minutes",
        "event_threshold_pu", "adaptation_days", "calibration_start", "calibration_end",
        "test_start", "test_end", "history_minutes", "classifier", "alpha_grid",
        "blend_weights", "switch_quantiles", "bootstrap_repetitions",
    }
    if set(spec) != expected:
        raise ValueError(f"Unknown fields: {set(spec)-expected}; missing: {expected-set(spec)}")
    root = Path(__file__).resolve().parents[3]
    base = load_research_config(root / spec["base_config"])
    output = (root / spec["output_dir"]).resolve()
    if output.is_relative_to(base.output) or base.output.is_relative_to(output):
        raise ValueError("Ramp results need a directory separate from the base run")
    horizons = spec["horizons_minutes"]
    if (not horizons or len(horizons) != len(set(horizons)) or
            not set(horizons) <= {10 * h for h in base["horizons"]} or
            spec["primary_horizon_minutes"] not in horizons):
        raise ValueError("Select unique horizons from the base run")
    if spec["adaptation_days"] not in base["adaptation_days"] or "gru" not in base["methods"]:
        raise ValueError("An existing fine-tuned GRU is required for each selected farm and seed")
    if not 0 < spec["event_threshold_pu"] <= 1:
        raise ValueError("Event threshold must be a fraction of rated power")
    history = spec["history_minutes"]
    if (not history or 60 not in history or len(history) != len(set(history)) or
            any(not isinstance(x, int) or x < 10 or x % 10 or x // 10 >= base["lookback"] for x in history)):
        raise ValueError("History must contain 60 minutes and fit in the prepared windows")
    if "wind_scaled" not in base["features"]:
        raise ValueError("The base run must include wind speed")
    cal_start, cal_end, test_start, test_end = [
        pd.Timestamp(spec[k]) for k in ("calibration_start", "calibration_end", "test_start", "test_end")
    ]
    local_end = pd.Timestamp(base["adaptation_start"]) + pd.Timedelta(days=spec["adaptation_days"])
    if not (max(local_end, pd.Timestamp(base["validation_end"])) <= cal_start < cal_end
            <= test_start < test_end <= pd.Timestamp(base["test_end"])):
        raise ValueError("Fit, policy calibration and evaluation must use disjoint chronological intervals")
    grids = {
        "alpha_grid": lambda x: 0 < x < 1,
        "blend_weights": lambda x: 0 <= x <= 1,
        "switch_quantiles": lambda x: 0 < x < 1,
    }
    for key, condition in grids.items():
        grid = spec[key]
        if not grid or not all(condition(x) for x in grid) or grid != sorted(set(grid)):
            raise ValueError(f"{key} must be a nonempty ordered grid of unique values in the valid range")
    if not {0, 1} <= set(spec["blend_weights"]):
        raise ValueError("Include both constant-mixture endpoints")
    classifier = spec["classifier"]
    if (set(classifier) != {"iterations", "depth", "learning_rate", "l2_leaf_reg", "random_seed"} or
            classifier["iterations"] < 1 or not 1 <= classifier["depth"] <= 10 or
            not 0 < classifier["learning_rate"] <= 1 or classifier["l2_leaf_reg"] < 0 or
            classifier["random_seed"] < 0):
        raise ValueError("Invalid CatBoost classifier settings")
    if spec["bootstrap_repetitions"] < 100:
        raise ValueError("Too few bootstrap repetitions")
    return RampConfig(base, output, spec)
