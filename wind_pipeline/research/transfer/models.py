from __future__ import annotations

import math
import os
from dataclasses import dataclass

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from catboost import CatBoostRegressor
from torch import nn

from .data import CORE, FEATURES, tabular
from .weather import WEATHER_FEATURES


def seed_torch(config, seed):
    torch.set_num_threads(config['threads'])
    torch.manual_seed(seed)
    if config.device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


@dataclass(frozen=True)
class BatchIndex:
    history_rows: torch.Tensor
    weather_rows: torch.Tensor
    targets: torch.Tensor
    weights: torch.Tensor

    def __len__(self):
        return len(self.targets)


class TensorStore:
    """Keep measurements once on the chosen device and gather only the requested batches."""

    def __init__(self, config, farms):
        self.device = config.device
        self.offsets = torch.arange(1 - config['history_minutes'] // 10, 1, device=self.device)
        self.farm_offsets = {}
        history_size = weather_size = 0
        for farm in farms.values():
            self.farm_offsets[farm.name] = (history_size, weather_size)
            history_size += farm.scada.shape[0] * farm.scada.shape[1]
            weather_size += len(farm.times)
        self.history = torch.empty((history_size, len(FEATURES)), device=self.device)
        self.weather = (torch.empty((weather_size, len(config.horizons), len(WEATHER_FEATURES)), device=self.device)
                        if config['weather'] else None)
        for farm in farms.values():
            history_offset, weather_offset = self.farm_offsets[farm.name]
            length = len(farm.times)
            for index in range(len(farm.turbines)):
                start = history_offset + index * length
                values = np.array(farm.scada[:, index], copy=True, order='C')
                self.history[start:start + length].copy_(torch.from_numpy(values))
            if self.weather is not None:
                self.weather[weather_offset:weather_offset + length].copy_(torch.from_numpy(np.array(farm.weather)))

    def pack(self, windows):
        histories, weather, targets, weights = [], [], [], []
        for window in windows:
            h_offset, w_offset = self.farm_offsets[window.farm.name]
            histories.append(h_offset + window.turbine_index * len(window.farm.times) + window.time_index)
            weather.append(w_offset + window.time_index)
            targets.append(window.targets)
            # Give each source park the same total weight regardless of turbine count.
            weights.append(np.full(len(window), 1 / len(window), np.float32))
        total = sum(map(len, windows))
        weight = np.concatenate(weights) * (total / len(windows))
        return BatchIndex(*(torch.as_tensor(np.concatenate(x), device=self.device) for x in (histories, weather, targets)),
                          torch.as_tensor(weight, device=self.device))

    def batches(self, data, ids, variant):
        columns = len(CORE) if variant == 'core' else len(FEATURES)
        row_ids = data.history_rows[ids, None] + self.offsets
        past = self.history[row_ids, :columns]
        weather = self.weather[data.weather_rows[ids]] if variant == 'weather' else None
        return past, weather, data.targets[ids], data.weights[ids]


class TransferGRU(nn.Module):
    def __init__(self, config, variant):
        super().__init__()
        self.variant = variant
        self.drop_probability = config['gru']['channel_dropout']
        self.history = config['history_minutes'] // 10
        self.hidden = config['gru']['hidden_size']
        self.sizes = (self.history,) if variant == 'core' else (3, 6, 12, self.history)
        channels = len(CORE) if variant == 'core' else len(FEATURES)
        self.encoder = nn.GRU(channels * 2, self.hidden, batch_first=True)
        width = self.hidden + 6 * len(self.sizes)
        if variant == 'weather':
            width += 3 * len(WEATHER_FEATURES)
        self.head_weight = nn.Parameter(torch.zeros(len(config.horizons), width))
        self.head_bias = nn.Parameter(torch.zeros(len(config.horizons)))

    def forward(self, past, weather):
        present = torch.isfinite(past)
        if self.training and self.drop_probability:
            channel_kept = torch.rand((len(past), 1, past.shape[-1]), device=past.device) >= self.drop_probability
            channel_kept[..., 0] = True
            present = present & channel_kept
        values = torch.where(present, past, 0.0)
        _, hidden = self.encoder(torch.cat((values, present), dim=-1))
        summary = []
        for size in self.sizes:
            measured = present[:, -size:, :2]
            selected = values[:, -size:, :2]
            count = measured.sum(1).clamp(min=1)
            mean = selected.sum(1) / count
            std = (((selected - mean[:, None]) ** 2 * measured).sum(1) / count).sqrt()
            delta = torch.where(measured[:, 0] & measured[:, -1], selected[:, -1] - selected[:, 0], 0.0)
            summary.extend((mean, std, delta))
        context = torch.cat((hidden[-1], *summary), dim=-1)
        if weather is None:
            adjustment = context @ self.head_weight.T
        else:
            available = torch.isfinite(weather)
            forecast = torch.where(available, weather, 0.0)
            slots = context[:, None, :].expand(-1, weather.shape[1], -1)
            # Interact forecast wind with the current operating level.
            x = torch.cat((slots, forecast, available, forecast * values[:, -1, :1, None]), dim=-1)
            adjustment = (x * self.head_weight).sum(-1)
        return values[:, -1, :1] + adjustment + self.head_bias


def loss_sum(prediction, targets, weight, squared=False):
    valid = torch.isfinite(targets)
    difference = prediction - torch.where(valid, targets, 0.0)
    error = difference.square() if squared else difference.abs()
    weights = valid * weight[:, None]
    return (error * weights).sum(), weights.sum()


@torch.inference_mode()
def gru_predict(model, store, dataset, batch):
    model.eval()
    results = []
    for start in range(0, len(dataset), batch):
        past, forecast, _, _ = store.batches(dataset, slice(start, start + batch), model.variant)
        results.append(model(past, forecast).clamp(0, 1.1).cpu().numpy())
    return np.concatenate(results)


def fit_source_gru(config, store, train, validation, seed, variant, squared=False, epochs=None):
    seed_torch(config, seed)
    settings = config['gru']
    model = TransferGRU(config, variant).to(store.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=1e-4)
    batch = settings['batch_size']
    best, best_state, best_epoch = math.inf, None, 0
    history = []
    generator = torch.Generator(device=store.device).manual_seed(seed)
    budget = settings['epochs'] if epochs is None else epochs
    for epoch in range(1, budget + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator, device=store.device)
        total = torch.zeros(2, device=store.device)
        for ids in order.split(batch):
            past, forecast, target, weight = store.batches(train, ids, variant)
            optimizer.zero_grad(set_to_none=True)
            value, count = loss_sum(model(past, forecast), target, weight, squared)
            (value / count.clamp(min=1e-8)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += torch.stack((value.detach(), count))
        score = None
        if validation is not None:
            scores = gru_predict(model, store, validation, batch)
            y = validation.targets.cpu().numpy()
            differences = np.abs(scores - y)
            if squared:
                differences = np.square(differences)
            weight = validation.weights.cpu().numpy()[:, None]
            score = float(np.nansum(differences * weight) / (np.isfinite(y) * weight).sum())
        history.append({'epoch': epoch, 'train_loss': float(total[0] / total[1]), 'validation_loss': score})
        if score is None:
            continue
        if score < best:
            best, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= settings['patience']:
            break
    if validation is not None:
        if best_state is None:
            raise RuntimeError('No finite source validation loss')
        model.load_state_dict(best_state)
    return model, {'objective': 'MSE' if squared else 'MAE', 'best_epoch': best_epoch or budget,
                   'train_windows': len(train), 'validation_windows': 0 if validation is None else len(validation),
                   'history': history}


def fit_local_gru(config, store, data, seed, variant, mode, candidate, source=None):
    seed_torch(config, seed)
    model = TransferGRU(config, variant).to(store.device)
    if source is not None:
        model.load_state_dict(source.state_dict())
    if mode == 'head':
        model.encoder.requires_grad_(False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    anchors = [p.detach().clone() for p in parameters] if source is not None else []
    optimizer = torch.optim.AdamW(parameters, lr=candidate['learning_rate'], weight_decay=1e-4)
    batch = min(config['gru']['adaptation_batch_size'], len(data))
    generator = torch.Generator(device=store.device).manual_seed(seed)
    total = torch.zeros(2, device=store.device)
    model.train()
    for _ in range(candidate['updates']):
        ids = torch.randint(len(data), (batch,), generator=generator, device=store.device)
        past, forecast, y, weight = store.batches(data, ids, variant)
        optimizer.zero_grad(set_to_none=True)
        value, count = loss_sum(model(past, forecast), y, weight)
        loss = value / count.clamp(min=1e-8)
        if anchors and candidate['penalty']:
            loss = loss + candidate['penalty'] * sum((p - original).square().sum()
                                                    for p, original in zip(parameters, anchors, strict=True))
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        total += torch.stack((value.detach(), count))
    return model, {'objective': 'MAE', 'mode': mode, **candidate, 'batch_size': batch,
                   'train_windows': len(data), 'trainable_parameters': sum(p.numel() for p in parameters),
                   'train_mae_pu': float(total[0] / total[1])}


def cat_inputs(windows, variant, horizon_index):
    features, targets, weights = [], [], []
    for window in windows:
        ids = np.flatnonzero(np.isfinite(window.targets[:, horizon_index]))
        if not len(ids):
            raise ValueError(f'Missing CatBoost targets at {window.farm.name}, horizon index {horizon_index}')
        features.append(tabular(window, ids, variant, horizon_index))
        targets.append(window.targets[ids, horizon_index] - window.persistence[ids])
        weights.append(np.full(len(ids), 1 / len(ids), np.float32))
    sample_weight = np.concatenate(weights) * (sum(map(len, targets)) / len(targets))
    return np.concatenate(features), np.concatenate(targets), sample_weight


def fit_catboost(config, train, validation, seed, variant, horizon, iterations, squared=False):
    index = config.horizons.index(horizon)
    x, y, weight = cat_inputs(train, variant, index)
    settings = config['catboost']
    objective = 'RMSE' if squared else 'MAE'
    model = CatBoostRegressor(loss_function=objective, eval_metric=objective, iterations=iterations,
                              depth=settings['depth'], learning_rate=settings['learning_rate'],
                              random_seed=seed, thread_count=config['threads'], allow_writing_files=False,
                              task_type='GPU' if config.device == 'cuda' else 'CPU', verbose=False, border_count=64)
    validation_pool = None
    if validation:
        from catboost import Pool

        v_x, v_y, v_w = cat_inputs(validation, variant, index)
        validation_pool = Pool(v_x, v_y, weight=v_w, thread_count=config['threads'])
    model.fit(x, y, sample_weight=weight, eval_set=validation_pool,
              early_stopping_rounds=35 if validation else None, use_best_model=bool(validation))
    return model, {'objective': objective, 'trees': model.tree_count_, 'train_windows': len(y),
                   'validation_windows': 0 if validation_pool is None else validation_pool.num_row()}


def cat_predict(config, model, windows, variant, horizon, iterations=0):
    index = config.horizons.index(horizon)
    inputs = tabular(windows, np.arange(len(windows)), variant, index)
    return np.clip(windows.persistence + model.predict(inputs, ntree_end=iterations, thread_count=config['threads']),
                   0, 1.1).astype(np.float32)
