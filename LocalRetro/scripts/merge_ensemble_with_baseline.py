"""
merge_ensemble_with_baseline.py

Merges deep ensemble UQ signals with the existing baseline error labels
(seed 1 = your original baseline checkpoint, used as the correctness
reference by convention) and template-frequency tiers, then computes
bootstrap-CI AUROC for every signal, both overall and within each
frequency tier.

This directly answers the mentor's key open question for Section 2.1:
does the ensemble's disagreement signal ALSO collapse on the rare/OOD
tier (meaning the problem is the data regime, not the UQ method), or
does it recover where plain softmax confidence failed (meaning deep
ensembles are worth their 5x training cost here)?

Inputs:
  - ../outputs/deep_ensemble_signals.csv     (from deep_ensemble_eval.py)
  - ../outputs/localretro_baseline.csv       (uq, top_score, error -- correctness reference)
  - ../outputs/ood_frequency_analysis.csv    (bucket labels, optional but recommended)

Output:
  - printed full results table with bootstrap 95% CIs
  - ../outputs/ensemble_vs_baseline_bootstrap_results.csv

Run from scripts/ directory:
    python merge_ensemble_with_baseline.py
"""

import argparse
import csv

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def bootstrap_auroc_ci(errors, scores, n_boot=2000, seed=42):
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
    if len(boots) < 50:
        return point, None, None, n
    boots = np.array(boots)
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), n


def main(args):
    ensemble = pd.read_csv(args.ensemble)
    baseline = pd.read_csv(args.baseline)
    merged = baseline.merge(ensemble, on='mol_idx', how='inner')
    print('Merged %d molecules (baseline + ensemble signals).' % len(merged))

    have_tiers = False
    if args.ood_tiers:
        try:
            ood = pd.read_csv(args.ood_tiers)
            merged = merged.merge(ood[['mol_idx', 'bucket']], on='mol_idx', how='left')
            have_tiers = True
            print('Merged frequency tier labels.')
        except FileNotFoundError:
            print('WARNING: %s not found, skipping per-tier breakdown.' % args.ood_tiers)

    merged['ensemble_uq_mean'] = 1 - merged['ensemble_mean_top_score']
    merged['ensemble_uq_disagreement'] = merged['ensemble_std_top_score']
    merged['ensemble_uq_dissent'] = 1 - merged['ensemble_agreement_rate']

    signals = {
        'baseline: 1 - top_score (seed 1 only)': 'uq',
        'ensemble: 1 - mean_top_score': 'ensemble_uq_mean',
        'ensemble: std_top_score (disagreement)': 'ensemble_uq_disagreement',
        'ensemble: 1 - agreement_rate (dissent)': 'ensemble_uq_dissent',
    }

    results_rows = []

    def report(label_prefix, df):
        for name, col in signals.items():
            point, lo, hi, n = bootstrap_auroc_ci(df['error'].values, df[col].values)
            if point is None:
                print('  %-45s skipped (n=%d, insufficient class balance)' % (name, n))
                continue
            ci_str = ('[%.3f, %.3f]' % (lo, hi)) if lo is not None else 'n/a (small n)'
            print('  %-45s AUROC=%.4f  95%% CI=%s  (n=%d)' % (name, point, ci_str, n))
            results_rows.append({
                'tier': label_prefix, 'signal': name, 'auroc': point,
                'ci_low': lo, 'ci_high': hi, 'n': n
            })

    print()
    print('=== OVERALL (all %d molecules) ===' % len(merged))
    report('overall', merged)

    if have_tiers:
        for bucket in sorted(merged['bucket'].dropna().unique()):
            sub = merged[merged['bucket'] == bucket]
            print()
            print('=== TIER: %s (n=%d) ===' % (bucket, len(sub)))
            report(bucket, sub)

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['tier', 'signal', 'auroc', 'ci_low', 'ci_high', 'n'])
        writer.writeheader()
        for row in results_rows:
            writer.writerow(row)
    print()
    print('Wrote %s' % args.output)

    # direct answer to the mentor's open question
    if have_tiers:
        very_rare = merged[merged['bucket'].astype(str).str.contains('very rare', na=False)]
        if len(very_rare) > 0:
            print()
            print('=== DIRECT ANSWER: does the ensemble recover discrimination in the very-rare tier? ===')
            for name, col in signals.items():
                point, lo, hi, n = bootstrap_auroc_ci(very_rare['error'].values, very_rare[col].values)
                if point is None:
                    continue
                overlaps_random = (lo is not None and lo <= 0.5 <= hi)
                verdict = 'CI overlaps 0.5 (indistinguishable from random)' if overlaps_random else 'CI excludes 0.5 (real signal)'
                print('  %-45s AUROC=%.4f  -> %s' % (name, point, verdict))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Merge ensemble signals with baseline and evaluate with bootstrap CIs')
    parser.add_argument('--ensemble', default='../outputs/deep_ensemble_signals.csv')
    parser.add_argument('--baseline', default='../outputs/localretro_baseline.csv')
    parser.add_argument('--ood_tiers', default='../outputs/ood_frequency_analysis.csv')
    parser.add_argument('--output', default='../outputs/ensemble_vs_baseline_bootstrap_results.csv')
    args = parser.parse_args()
    main(args)
