"""
Mondrian (per-tier) conformal selective prediction for MHNreact.
Ported directly from LocalRetro's mondrian_conformal.py, same logic,
same hyperparameters, same split methodology - only the data loading
differs (MHNreact's mol_idx/uq/error/tier/frequency already live in one
merged file, plus std_top_score from the MC dropout run).

Inputs:  mhn_baseline_seed0.csv        (mol_idx, top_score, uq, error, frequency, tier)
         mhn_mc_dropout_signals.csv    (mol_idx, std_top_score, ...)
Outputs: prints global vs Mondrian comparison table
         saves chart to mhn_conformal_solution.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -- Load data ---------------------------------------------------------------
baseline = pd.read_csv('mhn_baseline_seed0.csv')
mc = pd.read_csv('mhn_mc_dropout_signals.csv')

df = baseline.merge(mc[['mol_idx', 'std_top_score']], on='mol_idx')
df['uq'] = 1 - df['top_score']
df['tier'] = pd.Categorical(df['tier'], categories=sorted(df['tier'].unique()), ordered=True)

# -- Logistic combiner (softmax + MC std) - identical to LocalRetro's --------
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

# -- Split into calibration / test halves - identical to LocalRetro's --------
d = df.sample(frac=1, random_state=42).reset_index(drop=True)
cal = d.iloc[:len(d)//2].copy()
tst = d.iloc[len(d)//2:].copy()
for c in (cal, tst):
    c['tier'] = pd.Categorical(c['tier'], categories=df['tier'].cat.categories, ordered=True)

s_cal, s_tst = combined_score(cal, tst)
cal = cal.assign(_s=s_cal); tst = tst.assign(_s=s_tst)

TARGET = 0.20

# -- Global (marginal) threshold ----------------------------------------------
order = np.argsort(s_cal); ec_s = cal['error'].values[order]
cum = np.cumsum(ec_s)/np.arange(1, len(ec_s)+1)
ok = np.where(cum <= TARGET)[0]
thr_g = np.sort(s_cal)[ok[-1]] if len(ok) else -np.inf
sel_g = tst['_s'].values <= thr_g

glob_rows = []
for t in df['tier'].cat.categories:
    tm = (tst['tier'] == t).values
    s = sel_g[tm]; te = tst['error'].values[tm]
    glob_rows.append(dict(tier=t, coverage=s.mean(),
                           err_kept=te[s].mean() if s.sum() > 0 else None))
glob = pd.DataFrame(glob_rows)

# -- Mondrian (per-tier) threshold --------------------------------------------
mond_rows = []
for t in df['tier'].cat.categories:
    cm = (cal['tier'] == t).values; tm = (tst['tier'] == t).values
    if cm.sum() < 20 or tm.sum() == 0:
        continue
    sc = cal['_s'].values[cm]; ec = cal['error'].values[cm]
    order2 = np.argsort(sc); ec_s2 = ec[order2]; sc_s2 = sc[order2]
    cum2 = np.cumsum(ec_s2)/np.arange(1, len(ec_s2)+1)
    ok2 = np.where(cum2 <= TARGET)[0]
    thr = sc_s2[ok2[-1]] if len(ok2) else -np.inf
    sel = tst['_s'].values[tm] <= thr; te = tst['error'].values[tm]
    mond_rows.append(dict(tier=str(t), n_test=int(tm.sum()),
                           coverage=round(float(sel.mean()), 3),
                           err_kept=round(float(te[sel].mean()), 3) if sel.sum() > 0 else None,
                           err_all=round(float(te.mean()), 3)))
mond = pd.DataFrame(mond_rows)

# -- Print results --------------------------------------------------------------
print(f"\nTarget error rate: {TARGET*100:.0f}%")
print("\nGLOBAL threshold (breaks guarantee in rare tiers):")
print(glob.to_string(index=False))
print("\nMONDRIAN threshold (per-tier, correct abstention):")
print(mond.to_string(index=False))

# -- Figure ------------------------------------------------------------------
tiers_short = ['Very rare\n(1-5)', 'Rare\n(6-20)', 'Common\n(21-100)', 'Very common\n(100+)']
colors = ['#C0392B', '#E67E22', '#27AE60', '#2E75B6']

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
errs = [r['err_kept']*100 if r['err_kept'] else 0 for _, r in glob.iterrows()]
covs = [r['coverage'] for _, r in glob.iterrows()]
bars = ax.bar(range(4), errs, color=colors, alpha=0.85, width=0.6)
ax.axhline(TARGET*100, color='black', linestyle='--', linewidth=1.5, label=f'Target ({TARGET*100:.0f}%)')
for b, cov, err in zip(bars, covs, errs):
    ax.text(b.get_x()+b.get_width()/2, max(err, 2)+1.5, f'{cov*100:.0f}% kept', ha='center', fontsize=9)
ax.set_xticks(range(4)); ax.set_xticklabels(tiers_short, fontsize=9)
ax.set_ylabel('Error rate among kept predictions (%)'); ax.set_ylim(0, 120)
ax.set_title('MHNreact: Global (marginal) threshold', fontsize=11)
ax.legend(fontsize=9)

ax = axes[1]
m_errs = [r['err_kept']*100 if r['err_kept'] else 0 for _, r in mond.iterrows()]
m_covs = [r['coverage'] for _, r in mond.iterrows()]
bars = ax.bar(range(len(mond)), m_errs, color=colors[:len(mond)], alpha=0.85, width=0.6)
ax.axhline(TARGET*100, color='black', linestyle='--', linewidth=1.5, label=f'Target ({TARGET*100:.0f}%)')
for i, (b, cov, err) in enumerate(zip(bars, m_covs, m_errs)):
    if cov == 0:
        ax.text(b.get_x()+b.get_width()/2, 3, 'Abstains\n(0% kept)',
                ha='center', fontsize=9, color='#C0392B', fontweight='bold')
    else:
        ax.text(b.get_x()+b.get_width()/2, err+1.5, f'{cov*100:.0f}% kept', ha='center', fontsize=9)
ax.set_xticks(range(len(mond))); ax.set_xticklabels([tiers_short[i] for i in range(len(mond))], fontsize=9)
ax.set_ylabel('Error rate among kept predictions (%)'); ax.set_ylim(0, 70)
ax.set_title('MHNreact: Mondrian (per-tier) threshold', fontsize=11)
ax.legend(fontsize=9)

plt.suptitle('MHNreact Selective Prediction: Global vs. Mondrian Conformal (target error = 20%)',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('mhn_conformal_solution.png', bbox_inches='tight', dpi=150)
print("\nFigure saved to mhn_conformal_solution.png")
