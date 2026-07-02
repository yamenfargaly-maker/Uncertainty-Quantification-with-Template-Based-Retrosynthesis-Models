"""
MHNreact deep ensemble evaluation - mirrors LocalRetro's deep_ensemble_eval.py
methodology as reconstructed from its runtime output:

  - Correctness (`error`) and frequency `tier` labels stay FIXED to the
    original baseline (seed 0) evaluation - the ensemble does not redefine
    "correct," it asks whether ensemble-derived confidence signals rank
    against the baseline model's actual right/wrong outcomes.
  - ensemble_mean_score = mean of each of the 4 new seeds' own top-1
    confidence score per molecule (independent of whether seeds agree on
    which template is top-1).
  - ensemble_std_score = std of the same 4 values (disagreement signal).
  - AUROC computed overall and in the very-rare tier, with Bonferroni
    correction across the 3 signals compared (baseline, ensemble mean,
    ensemble std), matching LocalRetro's Table 6 structure exactly.
  - Adds the paired DeLong / bootstrap difference test (Module 5) for
    ensemble mean vs. baseline, since that's the correct test (not
    marginal-CI overlap) — this was the E1 fix already applied to
    LocalRetro's numbers in the paper.

Run this AFTER run_mhn_ensemble_inference.sh has produced the 4 per-seed CSVs.
"""
import numpy as np
import pandas as pd
from scipy import stats

BASELINE_CSV = "mhn_baseline_seed0.csv"        # seed 0, fixed correctness/tier
SEED_CSVS = [f"mhn_ens_seed{i}_topscore.csv" for i in (1, 2, 3, 4)]
OUT_CSV = "mhn_deep_ensemble_signals.csv"


def auroc(score, label):
    score = np.asarray(score, float)
    label = np.asarray(label, int)
    n1 = label.sum()
    n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(score)
    return (r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auroc_ci_boot(score, label, n_boot=2000, seed=0, alpha=0.05):
    score = np.asarray(score, float)
    label = np.asarray(label, int)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(label))
    vals = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if label[b].sum() in (0, len(b)):
            continue
        vals.append(auroc(score[b], label[b]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(float(auroc(score, label)), 4), (round(float(lo), 4), round(float(hi), 4))


def auroc_ci_bonferroni(score, label, n_signals, n_boot=2000, seed=0):
    _, ci_raw = auroc_ci_boot(score, label, n_boot, seed, alpha=0.05)
    a_corr, ci_corr = auroc_ci_boot(score, label, n_boot, seed, alpha=0.05 / n_signals)
    return a_corr, ci_raw, ci_corr


def _auc_and_midranks(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
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
    return V10.mean(), V10, V01, m, n


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
    baseline = pd.read_csv(BASELINE_CSV).sort_values("mol_idx").reset_index(drop=True)

    seed_scores = []
    for path in SEED_CSVS:
        df = pd.read_csv(path).sort_values("mol_idx").reset_index(drop=True)
        assert (df["mol_idx"].values == baseline["mol_idx"].values).all(), \
            f"mol_idx mismatch between {path} and baseline - check ordering"
        seed_scores.append(df["top_score"].values)
        print(f"  {path}: n={len(df)}, mean_top_score={df['top_score'].mean():.4f}")

    seed_scores = np.vstack(seed_scores)  # shape (4, n_molecules)
    ensemble_mean_score = seed_scores.mean(axis=0)
    ensemble_std_score = seed_scores.std(axis=0)

    out = baseline[["mol_idx", "top_score", "error", "frequency", "tier"]].copy()
    out["ensemble_mean_score"] = ensemble_mean_score
    out["ensemble_std_score"] = ensemble_std_score
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")

    y = out["error"].values
    baseline_uq = 1 - out["top_score"].values
    ensemble_mean_uq = 1 - ensemble_mean_score
    ensemble_std = ensemble_std_score

    print("\n=== OVERALL ===")
    a_base, ci_base = auroc_ci_boot(baseline_uq, y)
    a_mean, ci_mean = auroc_ci_boot(ensemble_mean_uq, y)
    a_std, ci_std = auroc_ci_boot(ensemble_std, y)
    print(f"  Baseline (1-top_score)         AUROC={a_base:.4f} {ci_base}")
    print(f"  Ensemble mean uq                AUROC={a_mean:.4f} {ci_mean}")
    print(f"  Ensemble std (disagree)         AUROC={a_std:.4f} {ci_std}")

    print("\n=== VERY-RARE TIER ===")
    vr = out[out["tier"].astype(str).str.contains("very rare", case=False, na=False)]
    print(f"  n = {len(vr)}")
    for name, sig in [("Baseline", 1 - vr["top_score"].values),
                       ("Ensemble mean", 1 - vr["ensemble_mean_score"].values),
                       ("Ensemble std", vr["ensemble_std_score"].values)]:
        a_corr, ci_raw, ci_corr = auroc_ci_bonferroni(sig, vr["error"].values, n_signals=3)
        print(f"  {name:16s} AUROC={a_corr:.4f}  raw_CI={ci_raw}  bonferroni_CI={ci_corr}")

    print("\n=== ALL TIERS ===")
    for t in sorted(out["tier"].astype(str).unique()):
        g = out[out["tier"] == t]
        a_b = auroc(1 - g["top_score"].values, g["error"].values)
        a_m = auroc(1 - g["ensemble_mean_score"].values, g["error"].values)
        a_s = auroc(g["ensemble_std_score"].values, g["error"].values)
        print(f"  {t:<25} n={len(g):>5}  base={a_b:.3f}  ens_mean={a_m:.3f}  ens_std={a_s:.3f}")

    print("\n=== PAIRED TEST: ensemble mean vs. baseline (Module 5) ===")
    dl = delong_paired(y, baseline_uq, ensemble_mean_uq)
    bs = paired_bootstrap_diff(y, baseline_uq, ensemble_mean_uq)
    print(f"  DeLong: {dl}")
    print(f"  Paired bootstrap: {bs}")
    print(f"\n  Sentence for the paper: ensemble mean improves AUROC by "
          f"{dl['diff']:.3f} (paired bootstrap 95% CI [{bs['ci'][0]:.3f}, {bs['ci'][1]:.3f}]; "
          f"DeLong p={dl['p']:.4g})")
