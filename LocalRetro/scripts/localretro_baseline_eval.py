"""
localretro_baseline_eval.py

Reads ../outputs/localretro_baseline.csv (produced by localretro_baseline_calibration.py)
and computes:
  - AUROC: does uq = 1 - top_score separate correct (error=0) from incorrect (error=1) predictions?
  - Spearman correlation (all molecules, and hard-only subset where error=1)

Run from the scripts/ directory:
    python localretro_baseline_eval.py
"""

import argparse
import csv

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


def load_table(path):
    uq = []
    error = []
    top_score = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            uq.append(float(row['uq']))
            error.append(int(row['error']))
            top_score.append(float(row['top_score']))
    return np.array(uq), np.array(error), np.array(top_score)


def main(args):
    uq, error, top_score = load_table(args.input)
    n = len(uq)
    n_error = error.sum()
    n_correct = n - n_error

    print('Loaded %d molecules (%d correct, %d incorrect).' % (n, n_correct, n_error))
    print('Top-1 accuracy: %.4f' % (n_correct / n))

    if n_error == 0 or n_correct == 0:
        print('Cannot compute AUROC: need both correct and incorrect predictions present.')
        return

    auroc = roc_auc_score(error, uq)
    print('AUROC (uq separating error=1 from error=0): %.4f' % auroc)

    rho_all, p_all = spearmanr(uq, error)
    print('Spearman rho (uq vs error, all molecules): %.4f (p=%.4g)' % (rho_all, p_all))

    # Hard-only subset: among incorrect predictions, does uq rank severity?
    # there's no continuous "severity" ground truth here, so as a proxy we
    # correlate uq against top_score itself restricted to the error=1 subset
    # (sanity: should be strongly negative since uq = 1 - top_score by construction)
    hard_mask = error == 1
    if hard_mask.sum() > 2:
        rho_hard, p_hard = spearmanr(uq[hard_mask], top_score[hard_mask])
        print('Spearman rho (uq vs top_score, hard/error=1 subset, sanity check): %.4f (p=%.4g)' % (rho_hard, p_hard))

    # reliability-style summary: mean uq for correct vs incorrect
    print('Mean uq | correct predictions:   %.4f' % uq[error == 0].mean())
    print('Mean uq | incorrect predictions: %.4f' % uq[error == 1].mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LocalRetro baseline UQ evaluation')
    parser.add_argument('--input', default='../outputs/localretro_baseline.csv',
                         help='Path to calibration CSV produced by localretro_baseline_calibration.py')
    args = parser.parse_args()
    main(args)
