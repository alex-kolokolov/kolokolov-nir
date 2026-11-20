from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_legacy_results(project_root: str | Path, output: str | Path) -> Path:
    """Put audited historical metrics in one registry without mixing protocols."""
    root = Path(project_root)
    in_domain = pd.read_csv(root / "SAD" / "existing_results_combined.csv").rename(
        columns={
            "MAE_kw": "mae_kw",
            "RMSE_kw": "rmse_kw",
            "NMAE%": "nmae_pct",
            "R2": "r2",
        }
    )
    in_domain["protocol"] = "legacy_temporal_seen_or_heldout_turbine"
    in_domain["evaluation_domain"] = in_domain["split"]

    dundalk = pd.read_csv(root / "SAD" / "dundalk_recomputed_metrics.csv").rename(
        columns={
            "MAE_kw": "mae_kw",
            "RMSE_kw": "rmse_kw",
            "NMAE_850%": "nmae_pct",
            "R2": "r2",
        }
    )
    dundalk["split"] = "zero_shot"
    dundalk["protocol"] = "legacy_cross_farm_zero_shot_recomputed"
    dundalk["evaluation_domain"] = "Dundalk_V52"
    columns = [
        "model", "horizon", "split", "evaluation_domain", "protocol",
        "mae_kw", "rmse_kw", "nmae_pct", "r2",
    ]
    result = pd.concat([in_domain[columns], dundalk[columns]], ignore_index=True)
    result["known_limitation"] = (
        "historical normalization was fitted before the temporal split; "
        "use as a frozen reference, not as primary evidence"
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return destination

