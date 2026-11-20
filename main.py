from __future__ import annotations

import argparse
import json
from pathlib import Path

from wind_pipeline.config import load_config
from wind_pipeline.runner import (
    evaluate_checkpoint,
    experiment_dir,
    prepare,
    run_baselines,
    train,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wind transferability experiment runner")
    parser.add_argument(
        "stage",
        choices=("prepare", "baseline", "train", "evaluate", "legacy-report", "research", "ramps"),
    )
    parser.add_argument("--config", required=True, help="Path to experiment JSON")
    parser.add_argument("--checkpoint", help="Checkpoint for the evaluate stage")
    parser.add_argument("--step", choices=("all", "fetch", "prepare", "train", "evaluate", "report", "verify",
                                           "sensitivity"),
                        default="all", metavar="STAGE",
                        help="Research stage: all, fetch, prepare, train, evaluate, report, sensitivity (default: all)")
    parser.add_argument("--sensitivity-config", help="Settings JSON for the transfer sensitivity stage")
    args = parser.parse_args()
    if args.step == "verify" and args.stage != "research":
        parser.error("--step verify requires the transfer research pipeline")
    if args.step == "sensitivity" and args.stage != "research":
        parser.error("--step sensitivity requires the transfer research pipeline")
    if args.sensitivity_config and args.step != "sensitivity":
        parser.error("--sensitivity-config requires --step sensitivity")
    return args


def main() -> None:
    args = parse_args()
    if args.stage == "ramps":
        if args.step != "all" or args.checkpoint:
            raise SystemExit("The ramps experiment takes its settings from --config")
        from wind_pipeline.research.ramp.run import run_ramps
        print(run_ramps(args.config))
        return
    if args.stage == "research":
        with Path(args.config).open(encoding="utf-8") as settings:
            pipeline = json.load(settings).get("pipeline")
        if pipeline == "transfer":
            from wind_pipeline.research.transfer.runner import run_transfer

            print(run_transfer(args.config, args.step, args.sensitivity_config))
            return
        if args.step in ("verify", "sensitivity"):
            raise SystemExit(f"--step {args.step} requires a configuration with pipeline=transfer")
        from wind_pipeline.research.runner import run_research
        print(run_research(args.config, args.step))
        return
    config = load_config(args.config)
    if args.stage == "legacy-report":
        from wind_pipeline.legacy_results import export_legacy_results

        result = export_legacy_results(
            ".", experiment_dir(config) / "legacy_results.csv"
        )
    elif args.stage == "prepare":
        result = prepare(config)
    elif args.stage == "baseline":
        result = run_baselines(config)
    elif args.stage == "train":
        result = train(config)
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for evaluate")
        result = evaluate_checkpoint(config, args.checkpoint)
    print(result)


if __name__ == "__main__":
    main()
