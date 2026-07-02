"""
calibration_metrics.py

Adds the two things currently missing from the paper:

1. Bootstrap 95% CI on every AUROC (baseline + MC dropout signals).
   The mentor flagged that the rare-tier AUROC of 0.464 has a CI of
   roughly [0.24, 0.71] -- indistinguishable from both random and the
   overall baseline. This script computes those intervals honestly.

2. Reliability diagram + ECE/MCE.
   Your calibration story is currently argued via AUROC (discrimination),
   but the actual finding -- confidence barely shifts while accuracy
   collapses -- is a calibration statement. This produces the reliability
   diagram (does stated confidence match empirical accuracy?) and the
   standard Expected Calibration Error and Maximum Calibration Error
   metrics, both overall and per frequency tier.

Inputs:
  - ../outputs/localretro_baseline.csv         (uq, top_score, error)
  - ../outputs/ood_frequency_analysis.csv      (bucket labels)
  - ../outputs/raw_prediction/LocalRetro_USPTO_50K_mcdropout_uq.txt  (optional)

Outputs:
  - ../outputs/calibration_reliability_diagram.png   (main paper figure)
  - ../outputs/calibration_per_tier.png              (per-tier reliability)
  - ../outputs/bootstrap_ci_results.csv              (all AUROC + CIs)
  - printed ECE/MCE table

Run from scripts/ directory:
    python calibration_metrics.py
"""

import argparse
import csv
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve

plt.rcParams.update({
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

PALETTE = {
    'correct': '#2E75B6',
    'incorrect': '#C0392B',
    'accent': '#27AE60',
    'neutral': '#5B6770',
}


def bootstrap_auroc_ci(errors, scores, n_boot=2000, seed=42):
    """Returns (point, ci_low, ci_high, n)."""
    rng = np.random.RandomState(seed)
    errors = np.asarray(errors)
    scores = np.asarray(scores)
    n = len(errors)
    if len(set(errors)) < 2 or n < 10:
        return None, None, None, n
    point = roc_auc_score(errors, scores)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        e, s = errors[idx], scores[idx]
        if len(set(e)) < 2:
            continue
        boots.append(roc_auc_score(e, s))
    boots = np.array(boots)
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), n


def compute_ece_mce(confidences, labels, n_bins=10):
    """
    Standard ECE and MCE via equal-width bins on confidence (top_score).
    confidences: array of predicted confidence values (top_score, not UQ)
    labels: array of binary correctness (1 = correct, 0 = wrong)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(confidences)
    bin_stats = []
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if i == n_bins - 1:
            mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            bin_stats.append(None)
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = labels[mask].mean()
        bin_n = mask.sum()
        gap = abs(bin_conf - bin_acc)
        ece += (bin_n / n) * gap
        mce = max(mce, gap)
        bin_stats.append((float(bins[i]), float(bins[i+1]), float(bin_conf), float(bin_acc), int(bin_n), float(gap)))
    return ece, mce, bin_stats


def plot_reliability_diagram(ax, confidences, labels, n_bins=10, label='', color='#2E75B6', n=None):
    """Plot a reliability diagram on the given axes."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_confs = []
    bin_accs = []
    bin_sizes = []
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if i == n_bins - 1:
            mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_confs.append(confidences[mask].mean())
        bin_accs.append(labels[mask].mean())
        bin_sizes.append(mask.sum())

    ece, mce, _ = compute_ece_mce(np.array(confidences), np.array(labels), n_bins)
    lab = label + (' (n=%d)' % n if n else '') + '\nECE=%.3f  MCE=%.3f' % (ece, mce)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Perfect calibration')
    ax.plot(bin_confs, bin_accs, 'o-', color=color, linewidth=2, markersize=6, label=lab)
    ax.fill_between(bin_confs, bin_accs, bin_confs, alpha=0.12, color=color, label='Calibration gap')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Mean predicted confidence (top_score)')
    ax.set_ylabel('Fraction correct (empirical accuracy)')
    ax.legend(fontsize=9, loc='upper left')
    return ece, mce


def main(args):
    os.makedirs('../outputs', exist_ok=True)

    print('Loading data...')
    baseline = pd.read_csv(args.baseline)
    baseline['correctness'] = 1 - baseline['error']

    have_tiers = False
    if os.path.exists(args.ood_tiers):
        ood = pd.read_csv(args.ood_tiers)
        baseline = baseline.merge(ood[['mol_idx', 'bucket', 'frequency']], on='mol_idx', how='left')
        have_tiers = True

    have_mc = False
    if os.path.exists(args.mc_uq):
        mc = pd.read_csv(args.mc_uq, sep='\t')
        baseline = baseline.merge(mc[['mol_idx', 'mean_top_score', 'std_top_score', 'agreement_rate', 'predictive_entropy']],
                                   on='mol_idx', how='left', suffixes=('', '_mc'))
        baseline['mc_uq_mean'] = 1 - baseline['mean_top_score']
        baseline['mc_uq_std'] = baseline['std_top_score']
        baseline['mc_uq_dissent'] = 1 - baseline['agreement_rate']
        have_mc = True

    n_total = len(baseline)
    print('Loaded %d molecules.' % n_total)

    # ============================================================
    # 1. RELIABILITY DIAGRAM (main figure)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: overall reliability diagram
    ax = axes[0]
    ece_overall, mce_overall = plot_reliability_diagram(
        ax, baseline['top_score'].values, baseline['correctness'].values,
        label='LocalRetro (all)', color=PALETTE['correct'], n=n_total)
    ax.set_title('Overall Reliability Diagram\n(does stated confidence match empirical accuracy?)', fontsize=10.5)

    # Right: overlay by tier if available
    ax = axes[1]
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Perfect')
    tier_colors = {
        '1) very rare (1-5)': '#C0392B',
        '2) rare (6-20)': '#E67E22',
        '3) common (21-100)': '#27AE60',
        '4) very common (100+)': '#2E75B6',
    }
    if have_tiers:
        for bucket in sorted(baseline['bucket'].dropna().unique()):
            sub = baseline[baseline['bucket'] == bucket]
            color = tier_colors.get(bucket, '#888888')
            bins = np.linspace(0, 1, 11)
            bcs, bas = [], []
            for i in range(10):
                mask = (sub['top_score'] >= bins[i]) & (sub['top_score'] <= bins[i+1])
                if mask.sum() < 3:
                    continue
                bcs.append(sub.loc[mask, 'top_score'].mean())
                bas.append(sub.loc[mask, 'correctness'].mean())
            if bcs:
                ece_t, _, _ = compute_ece_mce(sub['top_score'].values, sub['correctness'].values)
                label = '%s (n=%d, ECE=%.3f)' % (bucket.split(') ')[1] if ') ' in bucket else bucket, len(sub), ece_t)
                ax.plot(bcs, bas, 'o-', color=color, linewidth=2, markersize=5, label=label)
        ax.set_title('Reliability by Training-Template Frequency Tier\n(reveals where calibration breaks down)', fontsize=10.5)
    else:
        ax.set_title('(Per-tier breakdown not available)', fontsize=10.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('Mean predicted confidence (top_score)')
    ax.set_ylabel('Fraction correct')
    ax.legend(fontsize=8.5, loc='upper left')

    plt.tight_layout()
    plt.savefig(args.reliability_out, bbox_inches='tight')
    plt.close()
    print('Saved reliability diagram to %s' % args.reliability_out)

    # ============================================================
    # 2. ECE / MCE TABLE
    # ============================================================
    print()
    print('=== ECE and MCE (10 equal-width bins on top_score) ===')
    print('%-30s %8s %8s %8s' % ('Subset', 'n', 'ECE', 'MCE'))
    ece, mce, _ = compute_ece_mce(baseline['top_score'].values, baseline['correctness'].values)
    print('%-30s %8d %8.4f %8.4f' % ('Overall', n_total, ece, mce))
    if have_tiers:
        for bucket in sorted(baseline['bucket'].dropna().unique()):
            sub = baseline[baseline['bucket'] == bucket]
            ece_t, mce_t, _ = compute_ece_mce(sub['top_score'].values, sub['correctness'].values)
            print('%-30s %8d %8.4f %8.4f' % (bucket[:30], len(sub), ece_t, mce_t))

    # ============================================================
    # 3. BOOTSTRAP CI ON ALL AUROC NUMBERS
    # ============================================================
    print()
    print('=== Bootstrap 95% CI on all AUROC numbers (2000 resamples) ===')
    ci_rows = []

    signals = {'1 - top_score (baseline)': baseline['uq'].values}
    if have_mc:
        signals['1 - mean_top_score (MC avg)'] = baseline['mc_uq_mean'].values
        signals['std_top_score (MC dropout)'] = baseline['mc_uq_std'].values
        signals['1 - agreement_rate (MC)'] = baseline['mc_uq_dissent'].values
        signals['predictive_entropy (MC)'] = baseline['predictive_entropy'].values

    print()
    print('--- Overall (n=%d) ---' % n_total)
    for name, vals in signals.items():
        point, lo, hi, n = bootstrap_auroc_ci(baseline['error'].values, vals)
        if point is None:
            continue
        print('  %-40s AUROC=%.4f  95%% CI=[%.4f, %.4f]' % (name, point, lo, hi))
        ci_rows.append({'subset': 'overall', 'signal': name, 'n': n,
                        'auroc': point, 'ci_low': lo, 'ci_high': hi})

    if have_tiers:
        for bucket in sorted(baseline['bucket'].dropna().unique()):
            sub = baseline[baseline['bucket'] == bucket]
            print()
            print('--- Tier: %s (n=%d) ---' % (bucket, len(sub)))
            for name, _ in signals.items():
                vals = sub['uq'].values if 'top_score (baseline)' in name else \
                       sub.get('mc_uq_mean' if 'mean' in name else
                               'mc_uq_std' if 'std' in name else
                               'mc_uq_dissent' if 'agreement' in name else
                               'predictive_entropy', sub['uq']).values
                point, lo, hi, n = bootstrap_auroc_ci(sub['error'].values, vals)
                if point is None:
                    print('  %-40s skipped (n=%d, insufficient class balance)' % (name, len(sub)))
                    ci_rows.append({'subset': bucket, 'signal': name, 'n': len(sub),
                                    'auroc': None, 'ci_low': None, 'ci_high': None})
                    continue
                ci_str = '[%.4f, %.4f]' % (lo, hi) if lo is not None else 'small n'
                verdict = ''
                if lo is not None:
                    if lo <= 0.5 <= hi:
                        verdict = ' <- CI OVERLAPS 0.5 (indistinguishable from random)'
                    if lo > 0.5:
                        verdict = ' <- CI excludes 0.5 (real signal)'
                print('  %-40s AUROC=%.4f  95%% CI=%s%s' % (name, point, ci_str, verdict))
                ci_rows.append({'subset': bucket, 'signal': name, 'n': n,
                                'auroc': point, 'ci_low': lo, 'ci_high': hi})

    with open(args.ci_out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['subset', 'signal', 'n', 'auroc', 'ci_low', 'ci_high'])
        writer.writeheader()
        for row in ci_rows:
            writer.writerow(row)
    print()
    print('Saved bootstrap CI results to %s' % args.ci_out)
    print()
    print('NOTE: CI intervals that contain 0.5 do NOT support the claim that')
    print('the signal is meaningfully different from random for that tier.')
    print('The rare-tier AUROC of ~0.464 is expected to have a wide CI given n=100.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Calibration metrics: reliability diagram + ECE + bootstrap CIs')
    parser.add_argument('--baseline', default='../outputs/localretro_baseline.csv')
    parser.add_argument('--ood_tiers', default='../outputs/ood_frequency_analysis.csv')
    parser.add_argument('--mc_uq', default='../outputs/raw_prediction/LocalRetro_USPTO_50K_mcdropout_uq.txt')
    parser.add_argument('--reliability_out', default='../outputs/calibration_reliability_diagram.png')
    parser.add_argument('--ci_out', default='../outputs/bootstrap_ci_results.csv')
    args = parser.parse_args()
    main(args)
