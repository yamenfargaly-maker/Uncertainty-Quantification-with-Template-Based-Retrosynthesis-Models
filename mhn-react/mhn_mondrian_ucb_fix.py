"""
Finite-sample-corrected Mondrian threshold via Hoeffding upper confidence
bound on the risk, instead of the naive empirical-mean threshold search.

For each candidate threshold tau, the naive method asks: "is the empirical
error rate among calibration points with score<=tau below target?" This is
optimistic because we then pick the MOST PERMISSIVE tau satisfying that
naive check - a search/selection effect (we implicitly tried every tau and
kept the best-looking one).

The corrected method asks instead: "am I CONFIDENT (at level 1-delta) that
the TRUE error rate at this tau is below target?" using a Hoeffding upper
confidence bound on the empirical risk, and picks the most permissive tau
satisfying THAT. This directly guards against the optimism.
"""
import numpy as np
import pandas as pd

baseline = pd.read_csv('mhn_baseline_seed0.csv')
mc = pd.read_csv('mhn_mc_dropout_signals.csv')
df = baseline.merge(mc[['mol_idx', 'std_top_score']], on='mol_idx')
df['uq'] = 1 - df['top_score']
df['tier'] = pd.Categorical(df['tier'], categories=sorted(df['tier'].unique()), ordered=True)

def _logreg_fit(X, y, iters=800, lr=0.1, l2=1e-3):
    X = np.column_stack([np.ones(len(X)), X]); w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1/(1+np.exp(-X@w))
        g = X.T@(p-y)/len(y) + l2*np.r_[0, w[1:]]
        w -= lr*g
    return w

def _logreg_pred(w, X):
    X = np.column_stack([np.ones(len(X)), X])
    return 1/(1+np.exp(-X@w))

def combined_score(cal, test, feats=('uq', 'std_top_score')):
    mu = cal[list(feats)].mean(); sd = cal[list(feats)].std().replace(0, 1)
    Xc = ((cal[list(feats)]-mu)/sd).values
    Xt = ((test[list(feats)]-mu)/sd).values
    w = _logreg_fit(Xc, cal['error'].values.astype(float))
    return _logreg_pred(w, Xc), _logreg_pred(w, Xt)

def naive_threshold(sc, ec, target):
    order = np.argsort(sc); ec_s = ec[order]
    cum = np.cumsum(ec_s)/np.arange(1, len(ec_s)+1)
    ok = np.where(cum <= target)[0]
    return np.sort(sc)[ok[-1]] if len(ok) else -np.inf

def hoeffding_ucb_threshold(sc, ec, target, delta=0.1):
    """
    Most permissive threshold tau such that the Hoeffding upper confidence
    bound on the error rate among {score <= tau} stays <= target.
    UCB(tau) = empirical_risk(tau) + sqrt(log(1/delta) / (2*n_kept(tau))).
    """
    order = np.argsort(sc)
    sc_sorted = sc[order]; ec_sorted = ec[order]
    cum_err = np.cumsum(ec_sorted)
    n_kept = np.arange(1, len(sc_sorted) + 1)
    emp_risk = cum_err / n_kept
    ucb = emp_risk + np.sqrt(np.log(1/delta) / (2 * n_kept))
    ok = np.where(ucb <= target)[0]
    return sc_sorted[ok[-1]] if len(ok) else -np.inf

TARGET = 0.20
n_repeats = 20

for label, thresh_fn in [("v1: naive (empirical mean)", naive_threshold),
                          ("v3: Hoeffding UCB (delta=0.1)", lambda sc, ec, t: hoeffding_ucb_threshold(sc, ec, t, delta=0.1)),
                          ("v3: Hoeffding UCB (delta=0.2)", lambda sc, ec, t: hoeffding_ucb_threshold(sc, ec, t, delta=0.2))]:
    print(f"\n=== {label} ===")
    for tier in df['tier'].cat.categories:
        errs, covs = [], []
        for seed in range(n_repeats):
            d = df.sample(frac=1, random_state=seed).reset_index(drop=True)
            cal = d.iloc[:len(d)//2].copy()
            tst = d.iloc[len(d)//2:].copy()
            s_cal, s_tst = combined_score(cal, tst)
            cal = cal.assign(_s=s_cal); tst = tst.assign(_s=s_tst)
            cm = (cal['tier'] == tier).values; tm = (tst['tier'] == tier).values
            if cm.sum() < 20 or tm.sum() == 0:
                continue
            sc = cal['_s'].values[cm]; ec = cal['error'].values[cm]
            te = tst['error'].values[tm]; s_t = tst['_s'].values[tm]
            thr = thresh_fn(sc, ec, TARGET)
            sel = s_t <= thr
            if sel.sum() > 0:
                errs.append(te[sel].mean()); covs.append(sel.mean())
            else:
                covs.append(0.0)
        err_str = f"{np.mean(errs):.3f}" if errs else "abstains"
        print(f"  {str(tier):<25} true_held_out_error={err_str}  coverage={np.mean(covs):.3f}  "
              f"(nonzero-coverage: {len(errs)}/{n_repeats})")
