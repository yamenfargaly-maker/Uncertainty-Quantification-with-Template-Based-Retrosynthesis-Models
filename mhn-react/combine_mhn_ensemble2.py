"""
Second, INDEPENDENT MHNreact ensemble (seeds 5-8) analysis - tests whether
the ensemble-disagreement inversion (very-rare tier AUROC < 0.5) replicates
with a totally different set of ensemble seeds, addressing the mentor's
"across boundaries AND SEEDS" robustness recommendation (Part F).

Runs the same overall/very-rare-tier AUROC analysis as the first ensemble,
AND the Module 6 boundary sweep on this second ensemble's disagreement
signal - so this one script settles both robustness axes at once.
"""
import numpy as np
import pandas as pd
from scipy import stats

BASELINE_CSV = "mhn_baseline_seed0.csv"
SEED_CSVS = [f"mhn_ens2_seed{i}_topscore.csv" for i in (5, 6, 7, 8)]
OUT_CSV = "mhn_deep_ensemble2_signals.csv"
BOUNDARY_CUTS = [3, 5, 8, 10, 15, 20]


def auroc(score, label):
    s = np.asarray(score, float); l = np.asarray(label, int)
    n1 = l.sum(); n0 = len(l) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (stats.rankdata(s)[l == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auroc_ci_boot(score, label, n_boot=2000, seed=0, alpha=0.05):
    s = np.asarray(score, float); l = np.asarray(label, int)
    rng = np.random.default_rng(seed); idx = np.arange(len(l)); vals = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if l[b].sum() in (0, len(b)):
            continue
        vals.append(auroc(s[b], l[b]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(float(auroc(s, l)), 4), (round(float(lo), 4), round(float(hi), 4))


def auroc_ci_bonferroni(score, label, n_signals, n_boot=2000, seed=0):
    _, ci_raw = auroc_ci_boot(score, label, n_boot, seed, alpha=0.05)
    a_corr, ci_corr = auroc_ci_boot(score, label, n_boot, seed, alpha=0.05 / n_signals)
    return a_corr, ci_raw, ci_corr


def boundary_sweep(df, freq_col, signal_col, error_col, cuts, n_signals=3):
    rows = []
    for cut in cuts:
        g = df[df[freq_col] <= cut]
        if g[error_col].nunique() < 2 or len(g) < 30:
            rows.append(dict(cut=cut, n=len(g), note="too few / one class"))
            continue
        a_raw, ci_raw = auroc_ci_boot(g[signal_col].values, g[error_col].values, alpha=0.05)
        a_b, ci_b = auroc_ci_boot(g[signal_col].values, g[error_col].values, alpha=0.05 / n_signals)
        rows.append(dict(cut=cut, n=len(g), auroc=a_raw,
                          bonf_lo=ci_b[0], bonf_hi=ci_b[1],
                          excludes_half=bool(ci_b[0] > 0.5 or ci_b[1] < 0.5),
                          direction=("inverted" if a_raw < 0.5 else "positive")))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    baseline = pd.read_csv(BASELINE_CSV).sort_values("mol_idx").reset_index(drop=True)

    seed_scores = []
    for path in SEED_CSVS:
        df = pd.read_csv(path).sort_values("mol_idx").reset_index(drop=True)
        assert (df["mol_idx"].values == baseline["mol_idx"].values).all(), \
            f"mol_idx mismatch between {path} and baseline"
        seed_scores.append(df["top_score"].values)
        print(f"  {path}: n={len(df)}, mean_top_score={df['top_score'].mean():.4f}")

    seed_scores = np.vstack(seed_scores)
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

    print("\n=== OVERALL (second ensemble, seeds 5-8) ===")
    a_base, ci_base = auroc_ci_boot(baseline_uq, y)
    a_mean, ci_mean = auroc_ci_boot(ensemble_mean_uq, y)
    a_std, ci_std = auroc_ci_boot(ensemble_std, y)
    print(f"  Baseline (1-top_score)   AUROC={a_base:.4f} {ci_base}")
    print(f"  Ensemble mean uq         AUROC={a_mean:.4f} {ci_mean}")
    print(f"  Ensemble std (disagree)  AUROC={a_std:.4f} {ci_std}")

    print("\n=== VERY-RARE TIER (second ensemble) ===")
    vr = out[out["tier"].astype(str).str.contains("very rare", case=False, na=False)]
    print(f"  n = {len(vr)}")
    for name, sig in [("Baseline", 1 - vr["top_score"].values),
                       ("Ensemble mean", 1 - vr["ensemble_mean_score"].values),
                       ("Ensemble std", vr["ensemble_std_score"].values)]:
        a_corr, ci_raw, ci_corr = auroc_ci_bonferroni(sig, vr["error"].values, n_signals=3)
        print(f"  {name:16s} AUROC={a_corr:.4f}  raw_CI={ci_raw}  bonferroni_CI={ci_corr}")

    print("\n=== BOUNDARY SWEEP on second-ensemble disagreement (seeds 5-8) ===")
    tbl = boundary_sweep(out, "frequency", "ensemble_std_score", "error", BOUNDARY_CUTS, n_signals=3)
    print(tbl.to_string(index=False))
    robust = tbl["excludes_half"].all() if "excludes_half" in tbl else False
    print(f"\nRobust across all cuts (second ensemble): {robust}")

    print("\n=== SEED-ROBUSTNESS SUMMARY ===")
    print("First ensemble (seeds 1-4), very-rare tier: AUROC=0.427, Bonferroni CI [0.394, 0.461]")
    vr_std_a, _, vr_std_ci = auroc_ci_bonferroni(vr["ensemble_std_score"].values, vr["error"].values, n_signals=3)
    print(f"Second ensemble (seeds 5-8), very-rare tier: AUROC={vr_std_a:.4f}, Bonferroni CI {vr_std_ci}")
    replicates = vr_std_ci[1] < 0.5
    print(f"Inversion replicates with independent seeds: {replicates}")
