"""
ood_frequency_analysis.py

Proxy out-of-distribution (OOD) analysis: uses each test reaction's TRUE
template's frequency in the TRAINING set (from preprocessed_test.csv's
'Frequency' column) as a measure of how in-distribution vs. out-of-
distribution that reaction is, without requiring a held-out-class retrain.

Two questions:
  1. Does accuracy degrade for rarer (more OOD-like) templates? (sanity check
     -- this should be true almost by construction, since the model has
     seen fewer examples of rare templates during training)
  2. Does the model's CONFIDENCE (1 - top_score) correctly track this --
     i.e. is the model appropriately less confident on rare-template
     reactions, or does it stay falsely confident even as it gets less
     reliable? This is the real calibration/OOD-awareness question.

Inputs:
  - ../data/USPTO_50K/preprocessed_test.csv   (Frequency column)
  - ../outputs/localretro_baseline.csv        (uq, error columns)

Output:
  - ../outputs/ood_frequency_analysis.csv     (per-molecule merged table)
  - printed bucketed summary stats

Run from scripts/ directory:
    python ood_frequency_analysis.py
"""

import argparse
import csv

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def load_frequency(path):
    """Returns dict: mol_idx -> Frequency (int)"""
    freq = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for mol_idx, row in enumerate(reader):
            try:
                freq[mol_idx] = int(row['Frequency'])
            except (KeyError, ValueError):
                freq[mol_idx] = None
    return freq


def load_baseline(path):
    rows = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[int(row['mol_idx'])] = {
                'uq': float(row['uq']),
                'top_score': float(row['top_score']),
                'error': int(row['error']),
            }
    return rows


def bucket_for(freq):
    if freq is None:
        return None
    if freq <= 5:
        return '1) very rare (1-5)'
    elif freq <= 20:
        return '2) rare (6-20)'
    elif freq <= 100:
        return '3) common (21-100)'
    else:
        return '4) very common (100+)'


def main(args):
    print('Loading template frequencies from %s ...' % args.preprocessed)
    freq = load_frequency(args.preprocessed)
    print('Loading baseline UQ table from %s ...' % args.baseline)
    baseline = load_baseline(args.baseline)

    rows_out = []
    for mol_idx in sorted(set(freq.keys()) & set(baseline.keys())):
        f = freq[mol_idx]
        b = baseline[mol_idx]
        bucket = bucket_for(f)
        if bucket is None:
            continue
        rows_out.append({
            'mol_idx': mol_idx,
            'frequency': f,
            'bucket': bucket,
            'uq': b['uq'],
            'top_score': b['top_score'],
            'error': b['error'],
        })

    print('Merged %d molecules.' % len(rows_out))

    with open(args.output, 'w', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=['mol_idx', 'frequency', 'bucket', 'uq', 'top_score', 'error'])
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)
    print('Wrote %s' % args.output)

    # --- overall correlation: does frequency itself predict error/uq? ---
    freqs_arr = np.array([r['frequency'] for r in rows_out])
    errors_arr = np.array([r['error'] for r in rows_out])
    uq_arr = np.array([r['uq'] for r in rows_out])

    rho_freq_error, p1 = spearmanr(freqs_arr, errors_arr)
    rho_freq_uq, p2 = spearmanr(freqs_arr, uq_arr)
    print()
    print('=== Overall correlations ===')
    print('Spearman(frequency, error):      rho=%.4f (p=%.3g)  [negative = rarer templates -> more errors, as expected]' % (rho_freq_error, p1))
    print('Spearman(frequency, uq):          rho=%.4f (p=%.3g)  [negative = rarer templates -> higher uq/less confidence, GOOD if model is well-calibrated]' % (rho_freq_uq, p2))

    # --- bucketed breakdown ---
    print()
    print('=== Bucketed breakdown ===')
    print('%-25s %8s %10s %10s %10s' % ('Bucket', 'N', 'Accuracy', 'Mean UQ', 'Mean top_score'))
    buckets = sorted(set(r['bucket'] for r in rows_out))
    for bucket in buckets:
        bucket_rows = [r for r in rows_out if r['bucket'] == bucket]
        n = len(bucket_rows)
        acc = 1 - np.mean([r['error'] for r in bucket_rows])
        mean_uq = np.mean([r['uq'] for r in bucket_rows])
        mean_score = np.mean([r['top_score'] for r in bucket_rows])
        print('%-25s %8d %10.4f %10.4f %10.4f' % (bucket, n, acc, mean_uq, mean_score))

    # --- AUROC of uq WITHIN each bucket (does confidence still separate correct/incorrect even within a frequency tier?) ---
    print()
    print('=== AUROC (uq vs error) within each frequency bucket ===')
    for bucket in buckets:
        bucket_rows = [r for r in rows_out if r['bucket'] == bucket]
        errs = [r['error'] for r in bucket_rows]
        uqs = [r['uq'] for r in bucket_rows]
        if len(set(errs)) < 2:
            print('%-25s skipped (only one class present)' % bucket)
            continue
        auroc = roc_auc_score(errs, uqs)
        print('%-25s AUROC=%.4f  (n=%d)' % (bucket, auroc, len(bucket_rows)))

    # --- the key calibration check: is the model's CONFIDENCE GAP between buckets proportional to the ACCURACY GAP? ---
    print()
    print('=== Calibration check: very rare vs very common ===')
    very_rare = [r for r in rows_out if r['bucket'] == '1) very rare (1-5)']
    very_common = [r for r in rows_out if r['bucket'] == '4) very common (100+)']
    if very_rare and very_common:
        acc_rare = 1 - np.mean([r['error'] for r in very_rare])
        acc_common = 1 - np.mean([r['error'] for r in very_common])
        uq_rare = np.mean([r['uq'] for r in very_rare])
        uq_common = np.mean([r['uq'] for r in very_common])
        print('Accuracy gap (common - rare):     %.4f' % (acc_common - acc_rare))
        print('Mean UQ gap (rare - common):       %.4f  [should be POSITIVE and substantial if model correctly signals lower confidence on rare/OOD-like templates]' % (uq_rare - uq_common))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('OOD proxy analysis via template training frequency')
    parser.add_argument('--preprocessed', default='../data/USPTO_50K/preprocessed_test.csv')
    parser.add_argument('--baseline', default='../outputs/localretro_baseline.csv')
    parser.add_argument('--output', default='../outputs/ood_frequency_analysis.csv')
    args = parser.parse_args()
    main(args)
