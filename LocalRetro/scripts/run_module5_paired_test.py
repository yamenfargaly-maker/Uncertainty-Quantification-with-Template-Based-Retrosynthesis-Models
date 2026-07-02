"""
Module 5 — Paired AUROC-difference test (DeLong + paired bootstrap)
Run on the LocalRetro pod against your existing ensemble output CSV.

Expected input: /workspace/LocalRetro/outputs/deep_ensemble_signals.csv
with columns including: error (1=wrong), top_score (baseline), ensemble_mean_score.
Adjust COL_* below if your actual column names differ.
"""
import numpy as np
import pandas as pd
from scipy import stats

CSV_PATH = "/workspace/LocalRetro/outputs/deep_ensemble_signals.csv"
COL_ERROR = "error"
COL_BASELINE_SCORE = "top_score"            # higher score = more confident correct
COL_ENSEMBLE_SCORE = "ensemble_mean_score"   # higher score = more confident correct


def _auc_and_midranks(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    assert m and n, "need both classes present"
    order = np.argsort(neg)
    neg_s = neg[order]
    lo = np.searchsorted(neg_s, pos, side="left")
    hi = np.searchsorted(neg_s, pos, side="right")
    V10 = (lo + (hi - lo) * 0.5) / n
    order2 = np.argsort(pos)
    pos_s = pos[order2]
    lo2 = np.searchsorted(pos_s, neg, side="left")
    hi2 = np.searchsorted(pos_s, neg, side="right")
    V01 = 1.0 - (lo2 + (hi2 - lo2) * 0.5) / m
    auc = V10.mean()
    return auc, V10, V01, m, n


def delong_paired(y, score_a, score_b):
    aA, V10a, V01a, m, n = _auc_and_midranks(y, score_a)
    aB, V10b, V01b, _, _ = _auc_and_midranks(y, score_b)
    S10 = np.cov(np.vstack([V10a, V10b]))
    S01 = np.cov(np.vstack([V01a, V01b]))
    S = S10 / m + S01 / n
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    se = np.sqrt(max(var_diff, 1e-300))
    diff = aB - aA
    z = diff / se
    p = 2 * stats.norm.sf(abs(z))
    return dict(auc_a=round(aA, 4), auc_b=round(aB, 4), diff=round(diff, 4),
                se=round(se, 4), z=round(z, 3), p=float(p))


def paired_bootstrap_diff(y, score_a, score_b, reps=2000, seed=0, alpha=0.05):
    y = np.asarray(y)
    a = np.asarray(score_a, float)
    b = np.asarray(score_b, float)
    rng = np.random.default_rng(seed)
    N = len(y)
    diffs = np.empty(reps)
    for k in range(reps):
        idx = rng.integers(0, N, N)
        yi = y[idx]
        if yi.min() == yi.max():
            diffs[k] = np.nan
            continue
        aa, *_ = _auc_and_midranks(yi, a[idx])
        bb, *_ = _auc_and_midranks(yi, b[idx])
        diffs[k] = bb - aa
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return dict(diff_mean=round(float(diffs.mean()), 4),
                ci=(round(float(lo), 4), round(float(hi), 4)),
                excludes_zero=bool(lo > 0 or hi < 0), reps=len(diffs))


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    y = df[COL_ERROR].values.astype(int)
    # nonconformity scores: higher = more likely wrong, so use (1 - confidence)
    a = 1 - df[COL_BASELINE_SCORE].values
    b = 1 - df[COL_ENSEMBLE_SCORE].values

    dl = delong_paired(y, a, b)
    bs = paired_bootstrap_diff(y, a, b)

    print("=== DeLong paired test (baseline vs ensemble mean) ===")
    print(dl)
    print()
    print("=== Paired bootstrap difference CI ===")
    print(bs)
    print()
    print(f"Sentence for the paper: ensemble mean improves AUROC by "
          f"{dl['diff']:.3f} (paired bootstrap 95% CI [{bs['ci'][0]:.3f}, {bs['ci'][1]:.3f}]; "
          f"DeLong p={dl['p']:.4f})")
