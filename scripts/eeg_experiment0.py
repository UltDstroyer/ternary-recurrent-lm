"""Experiment 0: subject-held-out EEGMAT rest versus mental arithmetic."""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from pathlib import Path

import mne
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trlm.ternary import QuantizedLinear

BASE = 'https://physionet.org/files/eegmat/1.0.0'
SUBJECTS = tuple(range(36))


def download(root: Path, subjects: tuple[int, ...]):
    root.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        for condition in (1, 2):
            name = f'Subject{subject:02d}_{condition}.edf'
            dest = root / name
            if not dest.exists():
                print('Downloading', name, flush=True)
                tmp = dest.with_suffix('.edf.part')
                try:
                    urllib.request.urlretrieve(f'{BASE}/{name}', tmp)
                    tmp.replace(dest)
                finally:
                    tmp.unlink(missing_ok=True)


def prepare(root: Path, subjects: tuple[int, ...], synthetic: bool = False):
    """Return nonoverlapping 4s windows, keeping paired subject/condition labels."""
    windows, labels, groups = [], [], []
    channel_names = None
    for subject in subjects:
        for condition in (1, 2):
            if synthetic:
                rng = np.random.default_rng(subject * 100 + condition)
                data = rng.normal(0, 1, (19, 60 * 128)).astype('float32')
                if condition == 2:
                    data[:4] += np.sin(np.arange(data.shape[1]) * 2 * np.pi * 10 / 128)
                names = [f'EEG{i}' for i in range(19)]
            else:
                raw = mne.io.read_raw_edf(root / f'Subject{subject:02d}_{condition}.edf', preload=True, verbose='ERROR')
                # EDF metadata can mark EEG channels as misc; explicit exclusions prevent ECG leakage.
                picks = [i for i, name in enumerate(raw.ch_names)
                         if not any(s in name.upper() for s in ('ECG', 'EKG', 'EOG', 'EMG', 'STATUS', 'ANNOTATION'))]
                names = [raw.ch_names[i] for i in picks]
                raw.pick(picks).filter(1, 30, verbose='ERROR').resample(128, verbose='ERROR')
                data = raw.get_data().astype('float32') * 1e6  # volts -> microvolts
                # Equal duration for both conditions; avoids a duration/position cue.
                data = data[:, :60 * 128]
            if channel_names is None:
                channel_names = names
            if names != channel_names:
                raise ValueError(f'channel mismatch for subject {subject}: {names} != {channel_names}')
            if data.shape[1] < 60 * 128 or not np.isfinite(data).all():
                raise ValueError(f'invalid or short EDF: subject {subject}, condition {condition}')
            for start in range(0, 60 * 128, 4 * 128):
                windows.append(data[:, start:start + 4 * 128])
                labels.append(condition - 1)
                groups.append(subject)
    return np.asarray(windows, dtype='float32'), np.asarray(labels), np.asarray(groups), channel_names


def split(subjects, seed=2026):
    rng = np.random.default_rng(seed)
    ordered = rng.permutation(subjects)
    return {'train': sorted(ordered[:24].tolist()), 'val': sorted(ordered[24:30].tolist()),
            'test': sorted(ordered[30:].tolist())}


def score(y, pred, groups):
    result = {'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
              'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0))}
    result['per_subject'] = {str(int(g)): {'balanced_accuracy': float(balanced_accuracy_score(y[groups == g], pred[groups == g])),
                                        'macro_f1': float(f1_score(y[groups == g], pred[groups == g], average='macro', zero_division=0))}
                             for g in np.unique(groups)}
    return result


def spectral(x):
    # 4s at 128Hz -> 0.25Hz bin spacing, log band power for each channel.
    spectrum = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    hz = np.fft.rfftfreq(x.shape[-1], d=1 / 128)
    bands = [(1, 4), (4, 8), (8, 13), (13, 30)]
    return np.stack([np.log1p(spectrum[..., (hz >= a) & (hz < b)].mean(-1)) for a, b in bands], axis=-1).reshape(len(x), -1)


class EEGNetSmall(nn.Module):
    """Compact EEGNet-inspired temporal / depthwise spatial / separable convolution."""
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 8, (1, 31), padding=(0, 15), bias=False), nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False), nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(.25),
            nn.Conv2d(16, 16, (1, 15), padding=(0, 7), groups=16, bias=False),
            nn.Conv2d(16, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(.25), nn.AdaptiveAvgPool2d((1, 1)))
        self.head = nn.Linear(16, 2)

    def forward(self, x):
        return self.head(self.net(x[:, None]).flatten(1))


class SharedFourLevelEEG(nn.Module):
    """Temporal EEG encoder then 3 passes through one shared four-level core."""
    def __init__(self, channels, hidden=48, passes=3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(channels, hidden, 17, stride=4, padding=8), nn.GELU(),
                                     nn.Conv1d(hidden, hidden, 9, stride=4, padding=4), nn.GELU())
        self.norm = nn.LayerNorm(hidden)
        def q(inp, out):
            return QuantizedLinear(inp, out, bias=True, quantization_levels=4, quantization_scheme='uniform')
        self.core_in = q(hidden, hidden)
        self.core_state = q(hidden, hidden)
        self.passes = passes
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        # Preserve time order; summary updates use one shared core at each pass.
        sequence = self.encoder(x).transpose(1, 2)
        state = torch.zeros_like(sequence[:, 0])
        for _ in range(self.passes):
            for step in sequence.unbind(1):
                state = self.norm(state + torch.tanh(self.core_in(step) + self.core_state(state)))
        return self.head(state)


def train_model(model, x, y, idx, device, seed, epochs, batch_size, lr):
    torch.manual_seed(seed)
    model = model.to(device)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x[idx['train']]), torch.from_numpy(y[idx['train']]).long()),
                              batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    val_x = torch.from_numpy(x[idx['val']]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    best, best_epoch, state = -1, -1, None
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(xb.to(device)), yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
        model.eval()
        with torch.inference_mode():
            pred = model(val_x).argmax(-1).cpu().numpy()
        metric = balanced_accuracy_score(y[idx['val']], pred)
        if metric > best:
            best, best_epoch = metric, epoch + 1
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(state)
    return model, best_epoch, float(best)


def evaluate(model, x, y, groups, mask, device):
    model.eval()
    xb = torch.from_numpy(x[mask]).to(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    with torch.inference_mode():
        start = time.perf_counter()
        pred = model(xb).argmax(-1).cpu().numpy()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        latency = (time.perf_counter() - start) * 1000 / len(xb)
    result = score(y[mask], pred, groups[mask])
    result.update(params=sum(p.numel() for p in model.parameters()), latency_ms_per_window_batch=float(latency),
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=Path('data/eegmat'))
    p.add_argument('--output-dir', type=Path, default=Path('runs/eeg_experiment0'))
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    p.add_argument('--smoke', action='store_true', help='synthetic data: exercise code only, no scientific result')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.smoke and (device.type != 'cuda' or torch.version.hip is None):
        raise RuntimeError('AMD ROCm PyTorch GPU unavailable; install supported ROCm PyTorch and retry (no silent CPU fallback).')
    subjects = SUBJECTS
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if not args.smoke:
        download(args.data_dir, subjects)
    x, y, groups, channels = prepare(args.data_dir, subjects, args.smoke)
    division = split(subjects)
    idx = {key: np.flatnonzero(np.isin(groups, value)) for key, value in division.items()}
    assert all(not set(division[a]) & set(division[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')])
    # Fit normalization exclusively on train subjects; EEGNet and recurrent share the same input.
    mean = x[idx['train']].mean(axis=(0, 2), keepdims=True)
    std = x[idx['train']].std(axis=(0, 2), keepdims=True).clip(min=1e-4)
    x = np.clip((x - mean) / std, -8, 8).astype('float32')
    output = {'dataset': 'EEGMAT 1.0.0' if not args.smoke else 'SYNTHETIC SMOKE ONLY',
              'split_seed': 2026, 'subjects': division, 'channels': channels, 'device': str(device),
              'torch_version': torch.__version__, 'hip_version': torch.version.hip,
              'preprocessing': '1-30Hz; 128Hz; first 60s each; nonoverlap 4s; train-only channel z-score, clip [-8,8]',
              'models': {}, 'seeds': args.seeds}
    features = spectral(x)
    for seed in args.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
        key = str(seed)
        scaler = StandardScaler().fit(features[idx['train']])
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed)
        clf.fit(scaler.transform(features[idx['train']]), y[idx['train']])
        for name, mask in [('val', idx['val']), ('test', idx['test'])]:
            pred = clf.predict(scaler.transform(features[mask]))
            output['models'].setdefault('spectral_logistic', {}).setdefault(key, {})[name] = score(y[mask], pred, groups[mask])
        for name, constructor in [('eegnet_small', EEGNetSmall), ('shared_four_level', SharedFourLevelEEG)]:
            torch.manual_seed(seed)
            model, epoch, val = train_model(constructor(x.shape[1]), x, y, idx, device, seed, args.epochs, args.batch_size, args.lr)
            result = {'selected_epoch': epoch, 'best_val_balanced_accuracy': val}
            for part in ('val', 'test'):
                result[part] = evaluate(model, x, y, groups, idx[part], device)
            if name == 'shared_four_level':
                quantized = sum(m.weight.numel() for m in model.modules() if isinstance(m, QuantizedLinear))
                total = sum(p.numel() for p in model.parameters())
                result['theoretical_weight_bits'] = quantized * 2 + (total - quantized) * 32
                result['actual_weights_packed'] = False
                result['recurrent_passes'] = model.passes
            output['models'].setdefault(name, {})[key] = result
            print(f'{name} seed={seed} val={val:.3f} test={result["test"]["balanced_accuracy"]:.3f}', flush=True)
        (args.output_dir / 'metrics.json').write_text(json.dumps(output, indent=2))
    print('Saved', args.output_dir / 'metrics.json')


if __name__ == '__main__':
    main()
