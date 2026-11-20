from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import ModelConfig, TrainingConfig
from .metrics import grouped_metric_rows
from .models import MaskedChannelPatchTST
from .windows import UniversalWindDataset


def fit_model(
    train_data: UniversalWindDataset,
    val_data: UniversalWindDataset,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    output_dir: str | Path,
) -> tuple[MaskedChannelPatchTST, pd.DataFrame]:
    _seed_everything(training_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Training] Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)
    print(f"[Training] Dataset: {len(train_data):,} train windows | {len(val_data):,} val windows | Batch size: {training_config.batch_size}", flush=True)

    target_idx = train_data.features.index("power_pu")
    wind_idx = train_data.features.index("wind_speed_ms") if "wind_speed_ms" in train_data.features else 1
    model = MaskedChannelPatchTST(
        lookback=train_data.lookback,
        horizons=len(train_data.horizons),
        max_channels=len(train_data.features),
        target_channel=target_idx,
        wind_channel=wind_idx,
        **asdict(model_config),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    loss_fn = nn.HuberLoss(delta=0.1)
    train_loader = _loader(train_data, training_config, shuffle=True)
    val_loader = _loader(val_data, training_config, shuffle=False)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best_patchtst.pt"
    best_loss = float("inf")
    history: list[dict[str, float]] = []
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print(f"[Training] Running {training_config.epochs} epochs...\n", flush=True)
    for epoch in range(1, training_config.epochs + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer,
            scaler=scaler,
            log_interval=180,
            stage_name=f"[Epoch {epoch:02d}]",
        )
        val_loss = _run_epoch(
            model,
            val_loader,
            loss_fn,
            device,
            log_interval=0,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        saved_str = ""
        if val_loss < best_loss:
            best_loss = val_loss
            saved_str = " -> [Checkpoint Saved]"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "features": list(train_data.features),
                    "lookback": train_data.lookback,
                    "horizons": list(train_data.horizons),
                    "model_config": asdict(model_config),
                },
                checkpoint_path,
            )

        print(
            f"=== Epoch {epoch:02d}/{training_config.epochs:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Best Val: {best_loss:.5f}{saved_str} ===\n",
            flush=True,
        )

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output / "training_history.csv", index=False)
    metadata = {
        "device": str(device),
        "best_val_loss": best_loss,
        "checkpoint": str(checkpoint_path),
    }
    (output / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    model = load_model(checkpoint_path, device=device)
    return model, history_frame


def predict(
    model: MaskedChannelPatchTST,
    dataset: UniversalWindDataset,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions, targets, rated = [], [], []
    model.eval()
    use_cuda = device.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            p_last = batch["p_last"].to(device) if "p_last" in batch else None
            with torch.amp.autocast("cuda", enabled=use_cuda):
                output = model(
                    batch["x"].to(device),
                    batch["x_mask"].to(device),
                    p_last=p_last,
                )
            predictions.append(output.cpu().numpy())
            targets.append(batch["y"].cpu().numpy())
            rated.append(batch["rated_power_kw"].cpu().numpy())
    return np.concatenate(predictions), np.concatenate(targets), np.concatenate(rated)


def evaluate(
    model: MaskedChannelPatchTST,
    dataset: UniversalWindDataset,
    batch_size: int = 256,
) -> pd.DataFrame:
    prediction, truth, rated = predict(model, dataset, batch_size=batch_size)
    return pd.DataFrame(grouped_metric_rows(
        model="MaskedChannelPatchTST",
        horizons=dataset.horizons,
        y_true=truth,
        y_pred=prediction,
        rated_power_kw=rated,
        farms=dataset.window_farms,
    ))


def load_model(
    checkpoint_path: str | Path, device: str | torch.device | None = None
) -> MaskedChannelPatchTST:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    target_idx = checkpoint["features"].index("power_pu")
    wind_idx = checkpoint["features"].index("wind_speed_ms") if "wind_speed_ms" in checkpoint["features"] else 1
    model = MaskedChannelPatchTST(
        lookback=checkpoint["lookback"],
        horizons=len(checkpoint["horizons"]),
        max_channels=len(checkpoint["features"]),
        target_channel=target_idx,
        wind_channel=wind_idx,
        **checkpoint["model_config"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)


def _loader(
    dataset: UniversalWindDataset,
    config: TrainingConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    sampler = None
    if shuffle and config.balance_domains:
        farms, counts = np.unique(dataset.window_farms, return_counts=True)
        inverse = dict(zip(farms, 1.0 / counts))
        weights = torch.as_tensor(
            [inverse[farm] for farm in dataset.window_farms], dtype=torch.double
        )
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    log_interval: int = 180,
    stage_name: str = "",
) -> float:
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    use_cuda = device.type == "cuda"
    total_batches = len(loader)

    for i, batch in enumerate(loader, start=1):
        x = batch["x"].to(device)
        mask = batch["x_mask"].to(device)
        target = batch["y"].to(device)
        p_last = batch["p_last"].to(device) if "p_last" in batch else None
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_cuda):
            output = model(x, mask, p_last=p_last)
            loss = loss_fn(output, target)
        if training:
            if scaler is not None and use_cuda:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batch_size = x.shape[0]
        batch_loss = float(loss.detach())
        total += batch_loss * batch_size
        count += batch_size

        if training and log_interval > 0 and (i % log_interval == 0 or i == total_batches):
            print(f"  {stage_name} batch {i:03d}/{total_batches} ({(i / total_batches)*100:.0f}%) | running loss: {total / count:.5f}", flush=True)

    if count == 0:
        raise ValueError("Dataset contains no valid windows")
    return total / count


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
