#!/usr/bin/env python3
"""Turn training_runner result CSVs into the paper's tables and plots.

Per (method, world size): completion time (mean/std/median over *completed*
episodes), success rate, and the headline RL/classic completion-time ratio per
size plus its change from the in-distribution size to the unseen one.
Stdlib only; plots are written when matplotlib is available.
Usage: ``python3 tools/analyze_results.py results/*classic*.csv results/*rl-eval*.csv``.
"""
import argparse
import csv
import glob
import math
import random
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SIZE = 8.0          # in-distribution edge length (training_runner.TRAIN_WORLD_SIZE)
METHODS = ('classic', 'rl')
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def is_success(reason: str) -> bool:
    """An episode completed iff it ended on reaching the coverage threshold."""
    return reason.strip().startswith('coverage=')


def distribution_of(size: float) -> str:
    """ID for the training size, OOD otherwise."""
    return 'ID' if abs(size - TRAIN_SIZE) < 1e-6 else 'OOD'


def load_episodes(paths: list[str]) -> tuple[list[dict], int]:
    """Read all CSVs; return episode dicts and the number of unusable rows."""
    episodes: list[dict] = []
    skipped = 0
    for path in paths:
        with open(path, newline='') as f:
            for row in csv.DictReader(f, delimiter=';'):
                try:
                    size = float(row['world_size'])
                    episodes.append({
                        'file': Path(path).name,
                        'mode': row['mode'].strip(),
                        'world_size': size,
                        'distribution': (row.get('distribution') or '').strip()
                        or distribution_of(size),
                        'success': is_success(row['reason']),
                        'reason': row['reason'].strip(),
                        'elapsed_s': float(row['elapsed_s']),
                    })
                except (KeyError, TypeError, ValueError):
                    skipped += 1
    return episodes, skipped


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _spread(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else float('nan')


def summarise(episodes: list[dict]) -> list[dict]:
    """One row per (method, world size); times are over completed episodes only."""
    rows = []
    keys = sorted({(e['mode'], e['world_size']) for e in episodes},
                  key=lambda k: (k[1], k[0]))
    for mode, size in keys:
        eps = [e for e in episodes if e['mode'] == mode and e['world_size'] == size]
        times = [e['elapsed_s'] for e in eps if e['success']]
        rows.append({
            'mode': mode,
            'world_size': size,
            'distribution': distribution_of(size),
            'episodes': len(eps),
            'completed': len(times),
            'success_rate': len(times) / len(eps),
            'time_mean': statistics.fmean(times) if times else float('nan'),
            'time_std': _spread(times) if times else float('nan'),
            'time_median': statistics.median(times) if times else float('nan'),
            'time_min': min(times) if times else float('nan'),
            'time_max': max(times) if times else float('nan'),
        })
    return rows


def _mean_ratio(rl: list[float], classic: list[float]) -> float:
    return statistics.fmean(rl) / statistics.fmean(classic)


def _bootstrap(rng: random.Random, samples: list[float]) -> list[float]:
    return [rng.choice(samples) for _ in samples]


def _ci(values: list[float]) -> tuple[float, float]:
    values = sorted(values)
    return (values[int(0.025 * (len(values) - 1))],
            values[int(math.ceil(0.975 * (len(values) - 1)))])


def _times(episodes: list[dict], mode: str, size: float) -> list[float]:
    return [e['elapsed_s'] for e in episodes
            if e['mode'] == mode and e['world_size'] == size and e['success']]


def ratios(episodes: list[dict]) -> tuple[list[dict], dict | None]:
    """RL/classic mean completion-time ratio per size, and its change ID → OOD.

    Absolute times are not comparable across sizes, so the ratio is the result.
    95 % intervals come from a percentile bootstrap over episodes. The change is
    reported as ratio_OOD / ratio_ID (>1: RL degrades relative to classic on
    the unseen size) and as the difference ratio_OOD − ratio_ID.
    """
    rng = random.Random(BOOTSTRAP_SEED)
    per_size: list[dict] = []
    boots: dict[float, list[float]] = {}
    for size in sorted({e['world_size'] for e in episodes}):
        rl, classic = _times(episodes, 'rl', size), _times(episodes, 'classic', size)
        if not rl or not classic:
            continue
        draws = [_mean_ratio(_bootstrap(rng, rl), _bootstrap(rng, classic))
                 for _ in range(BOOTSTRAP_N)]
        boots[size] = draws
        lo, hi = _ci(draws)
        per_size.append({
            'world_size': size, 'distribution': distribution_of(size),
            'ratio': _mean_ratio(rl, classic), 'ci_low': lo, 'ci_high': hi,
            'n_rl': len(rl), 'n_classic': len(classic),
        })

    base = next((r for r in per_size if r['distribution'] == 'ID'), None)
    others = [r for r in per_size if r['distribution'] == 'OOD']
    if base is None or not others:
        return per_size, None
    ood = others[0]
    quot = [o / b for o, b in zip(boots[ood['world_size']], boots[base['world_size']])]
    diff = [o - b for o, b in zip(boots[ood['world_size']], boots[base['world_size']])]
    change = {
        'from_size': base['world_size'], 'to_size': ood['world_size'],
        'quotient': ood['ratio'] / base['ratio'],
        'quotient_ci': _ci(quot),
        'difference': ood['ratio'] - base['ratio'],
        'difference_ci': _ci(diff),
    }
    return per_size, change


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt(x: float, spec: str = '.1f') -> str:
    return 'n/a' if isinstance(x, float) and math.isnan(x) else format(x, spec)


def print_tables(summary: list[dict], per_size: list[dict], change: dict | None) -> None:
    print('\nCompletion time (sim s, completed episodes) and success rate')
    print(f'{"method":<8}{"size":>6}{"dist":>6}{"n":>5}{"done":>6}{"success":>9}'
          f'{"mean":>9}{"std":>8}{"median":>9}{"min":>8}{"max":>8}')
    for r in summary:
        print(f'{r["mode"]:<8}{r["world_size"]:>6g}{r["distribution"]:>6}{r["episodes"]:>5}'
              f'{r["completed"]:>6}{r["success_rate"]:>9.0%}{_fmt(r["time_mean"]):>9}'
              f'{_fmt(r["time_std"]):>8}{_fmt(r["time_median"]):>9}'
              f'{_fmt(r["time_min"]):>8}{_fmt(r["time_max"]):>8}')

    print('\nRL / classic completion-time ratio (mean of means, 95 % bootstrap CI)')
    if not per_size:
        print('  n/a – need completed episodes of both methods on the same size')
    for r in per_size:
        print(f'  {r["world_size"]:g} x {r["world_size"]:g} ({r["distribution"]}): '
              f'{r["ratio"]:.3f}  [{r["ci_low"]:.3f}, {r["ci_high"]:.3f}]  '
              f'(n_rl={r["n_rl"]}, n_classic={r["n_classic"]})')
    if change:
        print(f'  change {change["from_size"]:g} -> {change["to_size"]:g}: '
              f'x{change["quotient"]:.3f} [{change["quotient_ci"][0]:.3f}, '
              f'{change["quotient_ci"][1]:.3f}], '
              f'difference {change["difference"]:+.3f} '
              f'[{change["difference_ci"][0]:+.3f}, {change["difference_ci"][1]:+.3f}]')
    else:
        print('  change ID -> OOD: n/a – need ratios for both sizes')


def write_csvs(out: Path, summary: list[dict], per_size: list[dict],
               change: dict | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (('summary.csv', summary), ('ratio.csv', per_size)):
        if not rows:
            continue
        with open(out / name, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=';')
            writer.writeheader()
            writer.writerows(rows)
    if change:
        with open(out / 'ratio_change.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['from_size', 'to_size', 'quotient', 'quotient_low',
                             'quotient_high', 'difference', 'difference_low',
                             'difference_high'])
            writer.writerow([change['from_size'], change['to_size'], change['quotient'],
                             *change['quotient_ci'], change['difference'],
                             *change['difference_ci']])


def write_plots(out: Path, episodes: list[dict], summary: list[dict],
                per_size: list[dict]) -> bool:
    """Write PNG plots; return False if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    out.mkdir(parents=True, exist_ok=True)
    sizes = sorted({e['world_size'] for e in episodes})
    colors = {'classic': '#4c72b0', 'rl': '#dd8452'}
    labels = [f'{s:g}x{s:g}\n({distribution_of(s)})' for s in sizes]

    # Completion times: box + points per method and size.
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    for i, mode in enumerate(METHODS):
        pos = [k + (i - 0.5) * width for k in range(len(sizes))]
        data = [_times(episodes, mode, s) for s in sizes]
        keep = [(p, d) for p, d in zip(pos, data) if d]
        if keep:
            ax.boxplot([d for _, d in keep], positions=[p for p, _ in keep],
                       widths=width * 0.9, patch_artist=True,
                       boxprops={'facecolor': colors[mode], 'alpha': 0.6},
                       medianprops={'color': 'black'})
        ax.plot([], [], 's', color=colors[mode], label=mode)
    ax.set_xticks(range(len(sizes)), labels)
    ax.set_ylabel('Completion time [sim s]')
    ax.legend()
    ax.set_title('Completion time (completed episodes)')
    fig.tight_layout()
    fig.savefig(out / 'completion_time.png', dpi=200)
    plt.close(fig)

    # Success rate.
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, mode in enumerate(METHODS):
        rates = [next((r['success_rate'] for r in summary
                       if r['mode'] == mode and r['world_size'] == s), 0.0)
                 for s in sizes]
        ax.bar([k + (i - 0.5) * width for k in range(len(sizes))], rates, width,
               color=colors[mode], label=mode)
    ax.set_xticks(range(len(sizes)), labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Success rate')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'success_rate.png', dpi=200)
    plt.close(fig)

    # RL/classic ratio per size with 95 % CI.
    if per_size:
        fig, ax = plt.subplots(figsize=(5, 4))
        xs = list(range(len(per_size)))
        vals = [r['ratio'] for r in per_size]
        errs = [[r['ratio'] - r['ci_low'] for r in per_size],
                [r['ci_high'] - r['ratio'] for r in per_size]]
        ax.bar(xs, vals, 0.5, yerr=errs, capsize=4, color='#55a868')
        ax.axhline(1.0, color='grey', linestyle='--', linewidth=1)
        ax.set_xticks(xs, [f'{r["world_size"]:g}x{r["world_size"]:g}\n({r["distribution"]})'
                           for r in per_size])
        ax.set_ylabel('RL / classic completion time')
        ax.set_title('Time ratio (<1: RL faster)')
        fig.tight_layout()
        fig.savefig(out / 'time_ratio.png', dpi=200)
        plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Tables and plots from training_runner result CSVs')
    parser.add_argument('csv', nargs='*',
                        help='Result CSVs or globs (default: results/*.csv). Pass only the '
                             'runs to compare, e.g. classic runs and rl-eval runs.')
    parser.add_argument('--out', type=Path, default=PROJECT_ROOT / 'results' / 'analysis',
                        help='Output directory for CSV tables and plots')
    parser.add_argument('--no-plots', action='store_true', help='Tables only')
    args = parser.parse_args()

    patterns = args.csv or [str(PROJECT_ROOT / 'results' / '*.csv')]
    paths = sorted({p for pat in patterns for p in glob.glob(pat)})
    if not paths:
        print('No result CSVs found.', file=sys.stderr)
        return 1

    episodes, skipped = load_episodes(paths)
    print(f'Read {len(episodes)} episodes from {len(paths)} file(s)'
          + (f'; skipped {skipped} rows without world_size/elapsed_s' if skipped else ''))
    episodes = [e for e in episodes if e['mode'] in METHODS]
    if not episodes:
        print('No usable episodes.', file=sys.stderr)
        return 1

    summary = summarise(episodes)
    per_size, change = ratios(episodes)
    print_tables(summary, per_size, change)
    write_csvs(args.out, summary, per_size, change)
    if args.no_plots:
        print(f'\nTables written to {args.out}')
    elif write_plots(args.out, episodes, summary, per_size):
        print(f'\nTables and plots written to {args.out}')
    else:
        print(f'\nmatplotlib not available – tables only, written to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
