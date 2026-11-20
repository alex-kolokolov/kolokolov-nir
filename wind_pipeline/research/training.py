from __future__ import annotations

import gc
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from catboost import CatBoostRegressor
from torch.utils.data import DataLoader

from ..models.gru import ResidualGRU
from ..research_data import WindowSet
from .artifacts import file_hash, log_event, write_json


def resolve_device(config):
    requested = config['device']
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was explicitly requested but is unavailable')
    return torch.device('cuda' if requested != 'cpu' and torch.cuda.is_available() else 'cpu')


def seed_everything(seed):
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def checkpoint_complete(stem: Path, fingerprint: str) -> bool:
    meta = stem.with_suffix('.json')
    if not meta.exists():
        return False
    item = json.loads(meta.read_text(encoding='utf-8'))
    if item['run_fingerprint'] != fingerprint:
        raise ValueError(f'Checkpoint belongs to a different experiment: {stem}')
    model_file = stem.with_suffix('.pt' if item['family'] == 'gru' else '.cbm')
    if item.get('completed'):
        if not model_file.exists() or file_hash(model_file) != item['model_sha256']:
            raise ValueError(f'Checkpoint integrity failed: {model_file}')
        return True
    return False


def fit_gru(config, train: WindowSet, validation: WindowSet | None, *, stem: Path,
            seed: int, fingerprint: str, initial_checkpoint: Path | None = None):
    if checkpoint_complete(stem, fingerprint):
        print(f'Reuse {stem.name}', flush=True)
        return
    if len(train) < 2 or (validation is not None and len(validation) < 1):
        raise ValueError('Insufficient training or validation windows')
    stem.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    settings = config['gru']
    device = resolve_device(config)
    model = ResidualGRU(len(config['features']), settings['hidden_size'], len(config['horizons']))
    if initial_checkpoint is not None:
        saved = torch.load(initial_checkpoint, map_location='cpu', weights_only=True)
        if saved['run_fingerprint'] != fingerprint:
            raise ValueError('Pretrained model fingerprint mismatch')
        model.load_state_dict(saved['state_dict'])
    model.to(device)
    adaptation = validation is None
    epochs = settings['adaptation_epochs'] if adaptation else settings['epochs']
    lr = settings['adaptation_learning_rate'] if initial_checkpoint is not None else settings['learning_rate']
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(train, batch_size=settings['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0,
                        pin_memory=device.type == 'cuda')
    val_loader = None if validation is None else DataLoader(
        validation, batch_size=settings['batch_size'], shuffle=False, num_workers=0)
    history, best_loss, best_state, stale = [], float('inf'), None, 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x) - y).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
        val_loss = None
        if val_loader is not None:
            model.eval()
            total = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    residual = model(x.to(device)) - y.to(device)
                    total += float(residual.square().mean()) * len(y)
            val_loss = total / len(validation)
            if val_loss < best_loss:
                best_loss, stale = val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        history.append(dict(epoch=epoch, train_mse=loss_sum/len(train), validation_mse=val_loss))
        print(f'{stem.name} epoch {epoch}/{epochs}: train={loss_sum/len(train):.6f}'
              + (f' val={val_loss:.6f}' if val_loss is not None else ''), flush=True)
        log_event(config, model=str(stem.relative_to(config.output)), **history[-1])
        if val_loader is not None and stale >= settings['patience']:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint = dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                      channels=len(config['features']), hidden_size=settings['hidden_size'],
                      horizons=len(config['horizons']), run_fingerprint=fingerprint)
    temporary = stem.with_suffix('.pt.tmp')
    torch.save(checkpoint, temporary)
    os.replace(temporary, stem.with_suffix('.pt'))
    write_json(stem.with_suffix('.history.json'), history)
    write_json(stem.with_suffix('.json'), dict(
        family='gru', completed=True, run_fingerprint=fingerprint, seed=seed,
        train_windows=len(train), validation_windows=0 if validation is None else len(validation),
        initial_checkpoint=str(initial_checkpoint) if initial_checkpoint else None,
        epochs_completed=len(history), best_validation_mse=None if best_state is None else best_loss,
        learning_rate=lr, elapsed_seconds=time.perf_counter()-started,
        model_sha256=file_hash(stem.with_suffix('.pt')), device=str(device)))
    del model, optimizer, loader
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def load_gru(path: Path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = ResidualGRU(saved['channels'], saved['hidden_size'], saved['horizons'])
    model.load_state_dict(saved['state_dict'])
    return model.to(device).eval()


def predict_gru(config, path: Path, dataset: WindowSet):
    device = resolve_device(config)
    model = load_gru(path, device)
    result = []
    with torch.no_grad():
        for x, _ in DataLoader(dataset, batch_size=config['gru']['batch_size'], shuffle=False):
            result.append(model(x.to(device)).cpu().numpy())
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return np.clip(np.concatenate(result), 0, config['preprocessing']['maximum_power_pu'])


def fit_catboost(config, train: WindowSet, validation: WindowSet | None, *, stem: Path,
                 seed: int, fingerprint: str):
    if checkpoint_complete(stem, fingerprint):
        print(f'Reuse {stem.name}', flush=True)
        return
    if len(train) < 2:
        raise ValueError('Insufficient CatBoost training windows')
    stem.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config)
    params = dict(config['catboost'])
    params.update(loss_function='MultiRMSE', random_seed=seed,
                  task_type='GPU' if device.type == 'cuda' else 'CPU',
                  thread_count=4, allow_writing_files=False)
    if device.type == 'cuda':
        params['devices'] = '0'
    model = CatBoostRegressor(**params)
    x = train.tabular()
    delta = train.targets - train.persistence[:, None]
    evaluation = None if validation is None else (
        validation.tabular(), validation.targets - validation.persistence[:, None])
    started = time.perf_counter()
    print(f'{stem.name}: CatBoost on {len(train):,} windows, {x.shape[1]} features', flush=True)
    model.fit(x, delta, eval_set=evaluation, use_best_model=validation is not None,
              early_stopping_rounds=50 if validation is not None else None, verbose=100)
    temporary = stem.with_suffix('.cbm.tmp')
    model.save_model(str(temporary))
    os.replace(temporary, stem.with_suffix('.cbm'))
    write_json(stem.with_suffix('.history.json'), model.get_evals_result())
    write_json(stem.with_suffix('.json'), dict(
        family='catboost', completed=True, run_fingerprint=fingerprint, seed=seed,
        train_windows=len(train), validation_windows=0 if validation is None else len(validation),
        trees=model.tree_count_, elapsed_seconds=time.perf_counter()-started,
        model_sha256=file_hash(stem.with_suffix('.cbm')), device=str(device),
        warning='GPU CatBoost is not guaranteed bitwise deterministic'))
    log_event(config, model=str(stem.relative_to(config.output)), completed=True,
              family='catboost', train_windows=len(train))
    del model, x, delta, evaluation
    gc.collect()


def predict_catboost(config, path: Path, dataset: WindowSet):
    model = CatBoostRegressor()
    model.load_model(str(path))
    output = model.predict(dataset.tabular(), thread_count=4) + dataset.persistence[:, None]
    return np.clip(output, 0, config['preprocessing']['maximum_power_pu'])
