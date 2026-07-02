"""
mc_dropout_eval.py

Merges MC-dropout uncertainty signals with the baseline calibration table
(specifically the 'error' correctness label) and evaluates each candidate
UQ signal against it: AUROC, Spearman rho.

Inputs:
  - ../outputs/localretro_baseline.csv               (mol_idx, ..., error)
  - ../outputs/raw_prediction/LocalRetro_USPTO_50K_mcdropout_uq.txt
        (mol_idx, mean_top_score, std_top_score, agreement_rate, predictive_entropy)

Run from scripts/ directory:
    python mc_dropout_eval.py
"""

import argparse
import csv

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


def load_baseline_errors(path):
    errors = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            errors[int(row['mol_idx'])] = int(row['error'])
    return errors


def load_mc_uq(path):
    rows = {}
    with open(path, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 5:
                continue
            mol_idx, mean_top_score, std_top_score, agreement_rate, predictive_entropy = parts
            rows[int(mol_idx)] = {
                'mean_top_score': float(mean_top_score),
                'std_top_score': float(std_top_score),
                'agreement_rate': float(agreement_rate),
                'predictive_entropy': float(predictive_entropy),
            }
    return rows


def eval_signal(name, values, errors):
    values = np.array(values)
    errors = np.array(errors)
    n_error = errors.sum()
    n_correct = len(errors) - n_error
    if n_error == 0 or n_correct == 0:
        print('%-30s skipped (need both classes present)' % name)
        return
    auroc = roc_auc_score(errors, values)
    rho, p = spearmanr(values, errors)
    print('%-30s AUROC=%.4f   Spearman rho=%.4f (p=%.3g)' % (name, auroc, rho, p))


def main(args):
    errors = load_baseline_errors(args.baseline)
    mc_uq = load_mc_uq(args.mc_uq)

    common_ids = sorted(set(errors.keys()) & set(mc_uq.keys()))
    print('Matched %d molecules between baseline and MC-dropout UQ files.' % len(common_ids))
    if len(common_ids) < len(errors):
        print('WARNING: %d baseline molecules had no matching MC-dropout row.' % (len(errors) - len(common_ids)))

    error_arr = [errors[i] for i in common_ids]

    print()
    print('=== Baseline signal (for reference) ===')
    # if uq column exists in baseline.csv we could reload it, but for now
    # just report MC-dropout-derived signals here

    print()
    print('=== MC Dropout candidate UQ signals ===')
    eval_signal('1 - mean_top_score', [1 - mc_uq[i]['mean_top_score'] for i in common_ids], error_arr)
    eval_signal('std_top_score', [mc_uq[i]['std_top_score'] for i in common_ids], error_arr)
    eval_signal('1 - agreement_rate', [1 - mc_uq[i]['agreement_rate'] for i in common_ids], error_arr)
    eval_signal('predictive_entropy', [mc_uq[i]['predictive_entropy'] for i in common_ids], error_arr)

    # combined signal: simple average of two normalized signals as an example ensemble
    combined = []
    std_vals = np.array([mc_uq[i]['std_top_score'] for i in common_ids])
    std_norm = (std_vals - std_vals.min()) / (std_vals.max() - std_vals.min() + 1e-12)
    uq_vals = np.array([1 - mc_uq[i]['mean_top_score'] for i in common_ids])
    uq_norm = (uq_vals - uq_vals.min()) / (uq_vals.max() - uq_vals.min() + 1e-12)
    combined = (std_norm + uq_norm) / 2
    print()
    eval_signal('ensemble: (1-mean_top_score)+std, normalized avg', combined, error_arr)

    # distribution check on agreement_rate, since it looked saturated at 1.0 in the head() preview
    agreement_vals = np.array([mc_uq[i]['agreement_rate'] for i in common_ids])
    print()
    print('agreement_rate distribution: min=%.3f  mean=%.3f  median=%.3f  pct_at_1.0=%.1f%%' % (
        agreement_vals.min(), agreement_vals.mean(), np.median(agreement_vals),
        100 * (agreement_vals == 1.0).mean()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MC dropout UQ signal evaluation')
    parser.add_argument('--baseline', default='../outputs/localretro_baseline.csv')
    parser.add_argument('--mc_uq', default='../outputs/raw_prediction/LocalRetro_USPTO_50K_mcdropout_uq.txt')
    args = parser.parse_args()
    main(args)
