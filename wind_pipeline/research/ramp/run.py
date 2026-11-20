"""Train a source-only event detector and evaluate forecast policies on another farm."""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from ..artifacts import environment, file_hash, model_path, object_hash, verify_run, write_json
from ..evaluation import paired_block_interval
from ..runner import load_farms
from ..training import checkpoint_complete, predict_gru
from .config import load_ramp_config
from .policy import (
    POLICIES, apply_policies, fit_event_switch, fit_policies, historical_features,
    ramp_labels, score_detector,
)
from .report import write_report


def window_inputs(windows, wind_index, steps, target_column):
    positions = windows.anchors[:, None] + np.arange(-max(steps), 1)
    power = windows.data.values[positions, 0].astype(np.float64)
    wind = windows.data.values[positions, wind_index].astype(np.float64)
    features = pd.DataFrame(historical_features(power, wind, steps))
    return power, features, windows.targets[:, target_column].astype(np.float64)


def atomic_csv(frame, destination):
    temp = destination.with_suffix(".csv.tmp")
    frame.to_csv(temp, index=False)
    temp.replace(destination)


def require_model(base, source, target, method, seed, days, fingerprint, inputs):
    stem = model_path(base, source, target, method, seed, days)
    if not checkpoint_complete(stem, fingerprint):
        raise ValueError(f"Base forecast model is incomplete: {stem}")
    path = stem.with_suffix(".pt")
    inputs[str(path.relative_to(base.root))] = file_hash(path)
    return path


def train_detector(settings, train_x, train_y):
    if np.unique(train_y).size != 2:
        raise ValueError("Both event classes must occur in source training")
    model = CatBoostClassifier(
        **settings, loss_function="Logloss", task_type="CPU", thread_count=4,
        allow_writing_files=False, verbose=False,
    )
    model.fit(train_x, train_y.astype(int))
    return model


def errors_in_kw(forecasts, truth, rated):
    return {k: (np.asarray(p) - truth) * rated for k, p in forecasts.items()}


def forecast_summary(errors, decisions):
    rows = []
    for name in POLICIES:
        error = errors[name]
        seed_mae = np.abs(error).mean(axis=1)
        rows.append({
            "policy": name,
            "mae_kw": float(seed_mae.mean()),
            "rmse_kw": float(np.sqrt(np.square(error).mean(axis=1)).mean()),
            "seed_std_mae_kw": float(seed_mae.std(ddof=1)) if len(seed_mae) > 1 else None,
            "fast_branch_share": float(decisions[name].mean()) if name in decisions else None,
        })
    return rows


def comparisons(errors, times, repetitions, primary):
    """Negative delta means the candidate forecast has lower MAE."""
    losses = {name: np.abs(error).mean(axis=0) for name, error in errors.items()}
    pairs = [
        ("ses", "persistence"),
        ("classifier_ses", "persistence"),
        ("classifier_ses", "blend_ses"),
        ("classifier_ses", "activity_ses"),
        ("classifier_ses", "classifier_persistence"),
        ("classifier_persistence", "persistence"),
        ("classifier_persistence", "blend_persistence"),
    ]
    for candidate, reference in pairs:
        is_primary = primary and (candidate, reference) == ("classifier_ses", "blend_ses")
        confidence = 0.975 if is_primary else 0.95
        difference = losses[candidate] - losses[reference]
        bounds, blocks = paired_block_interval(difference, times, repetitions, confidence=confidence)
        yield {
            "candidate": candidate, "reference": reference, "primary": is_primary,
            "confidence": confidence, "mae_difference_kw": float(difference.mean()),
            "lo_kw": float(bounds[0]), "hi_kw": float(bounds[1]), "weekly_blocks": blocks,
        }


def comparison_rows(config, farms, checkpoints, model_dir, prediction_dir):
    base, spec = config.parent, config.spec
    rated = base["rated_power_kw"]
    wind_index = base["features"].index("wind_scaled")
    steps = tuple(m // 10 for m in spec["history_minutes"])
    seeds = tuple(base["seeds"])
    for source in base["farms"]:
        target = next(f for f in base["farms"] if f != source)
        print(f"{source} -> {target}: fitting detector and policy on source data", flush=True)
        train = farms[source].select(base["train_start"], base["train_end"], base["train_stride"])
        calibration = farms[source].select(
            spec["calibration_start"], spec["calibration_end"], base["evaluation_stride"])
        test = farms[target].select(spec["test_start"], spec["test_end"], base["evaluation_stride"])
        if any(len(s) < 2 for s in (train, calibration, test)):
            raise ValueError("Insufficient observed windows for the requested split")
        source_forecasts = [predict_gru(base, checkpoints[source, "source", k], calibration) for k in seeds]
        prediction_cache = None
        for horizon in config.horizons:
            index = base["horizons"].index(horizon // 10)
            prefix = f"{source}__{target}__{horizon}m"
            train_power, train_x, train_truth = window_inputs(train, wind_index, steps, index)
            train_labels = ramp_labels(train_power[:, -1], train_truth, spec["event_threshold_pu"])
            detector = train_detector(spec["classifier"], train_x, train_labels)
            detector_path = model_dir / f"{prefix}.cbm"
            detector.save_model(str(detector_path))
            cal_power, cal_x, cal_truth = window_inputs(calibration, wind_index, steps, index)
            cal_prob = detector.predict_proba(cal_x, thread_count=4)[:, 1]
            cal_activity = cal_x["power_60m_activity"].to_numpy()
            cal_labels = ramp_labels(cal_power[:, -1], cal_truth, spec["event_threshold_pu"])
            event_switch = {
                name: fit_event_switch(score, cal_labels, spec["switch_quantiles"])
                for name, score in (("classifier", cal_prob), ("activity", cal_activity))
            }
            policies = [
                fit_policies(cal_power, forecast[:, index], cal_truth, cal_prob, cal_activity, spec)
                for forecast in source_forecasts
            ]
            # The test park and future labels have not been used to fit any gate.
            if prediction_cache is None:
                prediction_cache = [
                    predict_gru(base, checkpoints[source, "finetune", k], test) for k in seeds
                ]
            power, x_test, truth = window_inputs(test, wind_index, steps, index)
            activity = x_test["power_60m_activity"].to_numpy()
            probability = detector.predict_proba(x_test, thread_count=4)[:, 1]
            labels = ramp_labels(power[:, -1], truth, spec["event_threshold_pu"])
            detected = {
                "classifier": event_switch["classifier"].apply(probability),
                "activity": event_switch["activity"].apply(activity),
                "never": np.zeros(len(test), dtype=bool),
            }
            score = {
                "classifier": probability,
                "activity": activity,
                "never": np.full(len(test), train_labels.mean()),
            }
            identity = {"source": source, "target": target, "horizon_minutes": horizon}
            detector_rows = [
                {**identity, "detector": name, "windows": len(test),
                 **score_detector(labels, score[name], flags, probability=name != "activity")}
                for name, flags in detected.items()
            ]
            by_method = {name: [] for name in POLICIES}
            choices = {}
            forecasts = []
            for k, prediction, params in zip(seeds, prediction_cache, policies, strict=True):
                point, used = apply_policies(power, prediction[:, index], probability, activity, params)
                for name, error in errors_in_kw(point, truth, rated).items():
                    by_method[name].append(error)
                for name, flags in used.items():
                    choices.setdefault(name, []).append(flags)
                trace = pd.DataFrame({
                    "timestamp": test.times, "turbine_id": test.turbines, "seed": k,
                    "actual_kw": truth * rated, "observed_ramp": labels, "ramp_probability": probability,
                    "past_activity_pu": activity, "detected_ramp": detected["classifier"],
                    "activity_detected_ramp": detected["activity"],
                })
                for name, values in point.items():
                    trace[f"{name}_kw"] = values * rated
                for name, flags in used.items():
                    trace[f"{name}_uses_gru"] = flags
                forecasts.append(trace)
            for key in by_method:
                by_method[key] = np.stack(by_method[key])
            detail = pd.concat(forecasts, ignore_index=True)
            detail_path = prediction_dir / f"{prefix}.parquet"
            detail.to_parquet(detail_path, index=False)
            forecast_rows = [
                {**identity, "windows": len(test), **row}
                for row in forecast_summary(by_method, {k: np.stack(v) for k,v in choices.items()})
            ]
            comparison = [
                {**identity, **row} for row in comparisons(
                    by_method, test.times, spec["bootstrap_repetitions"],
                    horizon == spec["primary_horizon_minutes"])
            ]
            policy_record = {
                **identity,
                "source_train_windows": len(train),
                "source_calibration_windows": len(calibration),
                "test_windows": len(test),
                "test_turbines": int(len(np.unique(test.turbines))),
                "first_test_origin": str(test.times.min()),
                "last_test_origin": str(test.times.max()),
                "train_prevalence": float(train_labels.mean()),
                "event_switches": {k: asdict(v) for k, v in event_switch.items()},
                "policies_by_seed": {
                    str(k): {
                        "alpha": p["alpha"], "weights": p["blend_weights"],
                        "switches": {name: {c: asdict(rule) for c, rule in p[name].items()}
                                     for name in ("classifier", "activity")},
                    } for k, p in zip(seeds, policies, strict=True)
                },
                "features": list(train_x.columns),
                "detector_sha256": file_hash(detector_path),
                "predictions_sha256": file_hash(detail_path),
            }
            print(f"{prefix}: trained on {len(train):,}, evaluated on {len(test):,} windows", flush=True)
            yield detector_rows, forecast_rows, comparison, policy_record


def run_ramps(path):
    config = load_ramp_config(path)
    base = config.parent
    manifest = verify_run(base)
    fingerprint = manifest["run_fingerprint"]
    inputs = {}
    checkpoints = {}
    for source in base["farms"]:
        target = next(f for f in base["farms"] if f != source)
        for seed in base["seeds"]:
            checkpoints[source, "source", seed] = require_model(
                base, source, source, "gru_source", seed, 0, fingerprint, inputs)
            checkpoints[source, "finetune", seed] = require_model(
                base, source, target, "gru_finetune", seed, config.spec["adaptation_days"],
                fingerprint, inputs)
    code = sorted(Path(__file__).parent.glob("*.py"))
    snapshot = {str(p.relative_to(base.root)): file_hash(p) for p in code}
    run_manifest = {
        "config": config.spec,
        "parent_manifest_sha256": file_hash(base.output / "manifest.json"),
        "parent_run_fingerprint": fingerprint,
        "expert_checkpoints": inputs,
        "code": snapshot,
        "environment": environment(),
    }
    run_manifest["run_fingerprint"] = object_hash(run_manifest)
    output = config.output
    run_path = output / "manifest.json"
    if run_path.exists():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        if previous["run_fingerprint"] != run_manifest["run_fingerprint"]:
            raise ValueError("Ramp configuration or inputs changed: use a new output directory")
    model_dir, pred_dir = output / "models", output / "predictions"
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_path, {**run_manifest, "completed": False})
    write_json(output / "config.resolved.json", config.spec)
    for original in code:
        target = output / "source_snapshot" / original.relative_to(base.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    farms = load_farms(base, manifest)
    detectors, forecasts, pairs, policies = [], [], [], []
    for events, point, comparison, policy in comparison_rows(
            config, farms, checkpoints, model_dir, pred_dir):
        detectors.extend(events)
        forecasts.extend(point)
        pairs.extend(comparison)
        policies.append(policy)
    event_df = pd.DataFrame(detectors)
    forecast_df = pd.DataFrame(forecasts)
    pair_df = pd.DataFrame(pairs)
    atomic_csv(event_df, output / "event_metrics.csv")
    atomic_csv(forecast_df, output / "forecast_metrics.csv")
    atomic_csv(pair_df, output / "paired_comparisons.csv")
    write_json(output / "policy_parameters.json", policies)
    write_report(config, event_df, forecast_df, pair_df, policies)
    files = ("config.resolved.json", "event_metrics.csv", "forecast_metrics.csv",
             "paired_comparisons.csv", "policy_parameters.json", "report.md")
    write_json(run_path, {**run_manifest, "completed": True,
                         "outputs": {name: file_hash(output / name) for name in files}})
    return output
