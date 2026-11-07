#!/usr/bin/env python3
"""Plot an RL training run: learning curves and weight-snapshot progression.

Reads ``training_log.csv`` and ``snapshots/ep_*/`` from the RL data dir and
writes PNGs plus ``snapshots.csv`` to ``<data_dir>/plots/``. Needs matplotlib;
the snapshot comparison also needs torch (skipped without it).
Usage: ``python3 tools/plot_training.py [rl_data] [--smooth 50]``.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROBE_SEED = 0
PROBE_CANDIDATES = 32


def load_log(path: Path) -> list[dict]:
    """Read the telemetry CSV into dicts with numeric values."""
    with open(path, newline='') as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def smooth(values: list[float], window: int) -> list[float]:
    """Trailing moving average (window shrinks at the start)."""
    out, acc = [], 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def per_episode(rows: list[dict]) -> dict[str, list[float]]:
    """Mean loss/reward and last ε/buffer per episode."""
    eps: dict[int, list[dict]] = {}
    for r in rows:
        eps.setdefault(int(r['episode']), []).append(r)
    keys = sorted(eps)
    mean = (lambda k, f: sum(r[f] for r in eps[k]) / len(eps[k]))
    return {
        'episode': [float(k) for k in keys],
        'loss': [mean(k, 'loss') for k in keys],
        'reward': [mean(k, 'reward') for k in keys],
        'epsilon': [eps[k][-1]['epsilon'] for k in keys],
        'buffer': [eps[k][-1]['buffer'] for k in keys],
    }


def plot_curves(rows: list[dict], window: int, out: Path) -> Path:
    """Loss, reward and ε over training (per-step raw + smoothed, per-episode ε)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    steps = [r['step'] for r in rows]
    ep = per_episode(rows)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=False)
    for ax, key, label in ((axes[0], 'loss', 'Loss'), (axes[1], 'reward', 'Reward')):
        vals = [r[key] for r in rows]
        ax.plot(steps, vals, alpha=0.25, lw=0.6)
        ax.plot(steps, smooth(vals, window), lw=1.5, label=f'MA({window})')
        ax.set_xlabel('Gradient step')
        ax.set_ylabel(label)
        ax.legend()
    axes[2].plot(ep['episode'], ep['epsilon'])
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('ε')
    fig.tight_layout()
    path = out / 'learning_curves.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def snapshot_dirs(data_dir: Path) -> list[Path]:
    """Snapshot directories in training order."""
    return sorted((data_dir / 'snapshots').glob('ep_*'))


def probe_scores(snaps: list[Path]) -> list[dict]:
    """Score a fixed random candidate set with each snapshot's agent net.

    Shows how the learned preference per observation feature evolves: the
    Pearson correlation of Q with each of the OBS_DIM features, plus the
    parameter distance to the previous snapshot.
    """
    import numpy as np
    import torch
    sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'tb3_reinforced_exploration'))
    sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'tb3_slam'))
    from tb3_reinforced_exploration.rl_core import OBS_DIM, make_q_network

    obs = np.random.default_rng(PROBE_SEED).random(
        (PROBE_CANDIDATES, OBS_DIM)).astype(np.float32)
    rows, prev = [], None
    for snap in snaps:
        net = make_q_network(OBS_DIM)
        state = torch.load(snap / 'agent.pt', weights_only=True, map_location='cpu')
        net.load_state_dict(state)
        with torch.no_grad():
            q = net(torch.from_numpy(obs)).squeeze(-1).numpy()
        flat = torch.cat([v.flatten() for v in state.values()])
        meta = json.loads((snap / 'meta.json').read_text())
        row = {'episode': meta['episode'], 'step': meta['step'], 'epsilon': meta['epsilon'],
               'delta': float((flat - prev).norm()) if prev is not None else 0.0}
        for i in range(OBS_DIM):
            c = np.corrcoef(obs[:, i], q)[0, 1] if q.std() > 0 else 0.0
            row[f'corr_f{i}'] = float(c)
        rows.append(row)
        prev = flat
    return rows


def plot_snapshots(rows: list[dict], out: Path) -> Path:
    """Feature sensitivity and weight drift across snapshots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    eps = [r['episode'] for r in rows]
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for k in sorted(k for k in rows[0] if k.startswith('corr_f')):
        a.plot(eps, [r[k] for r in rows], marker='o', label=k[5:])
    a.set_ylabel('corr(Q, feature) on probe set')
    a.legend(ncol=5)
    b.plot(eps, [r['delta'] for r in rows], marker='o')
    b.set_ylabel('‖Δ weights‖ to previous snapshot')
    b.set_xlabel('Episode')
    fig.tight_layout()
    path = out / 'snapshot_progression.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('data_dir', nargs='?', default=str(PROJECT_ROOT / 'rl_data'),
                    help='RL data dir (default: rl_data/)')
    ap.add_argument('--smooth', type=int, default=50, help='moving-average window (steps)')
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    log = data_dir / 'training_log.csv'
    if not log.exists():
        print(f'no {log} – train first', file=sys.stderr)
        return 1
    out = data_dir / 'plots'
    out.mkdir(exist_ok=True)
    rows = load_log(log)
    if not rows:
        print(f'{log} has no rows', file=sys.stderr)
        return 1
    print(f'wrote {plot_curves(rows, max(1, args.smooth), out)}')
    snaps = snapshot_dirs(data_dir)
    if not snaps:
        print('no snapshots found – skipping progression plot')
        return 0
    try:
        probe = probe_scores(snaps)
    except ImportError as exc:
        print(f'skipping snapshot progression ({exc})')
        return 0
    with open(out / 'snapshots.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(probe[0]))
        w.writeheader()
        w.writerows(probe)
    print(f'wrote {plot_snapshots(probe, out)} and {out / "snapshots.csv"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
