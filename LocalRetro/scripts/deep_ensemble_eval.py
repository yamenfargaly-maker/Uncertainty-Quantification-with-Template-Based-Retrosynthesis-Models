import sys, os, re, numpy as np, pandas as pd
from scipy import stats

SEEDS     = ['seed2', 'seed3', 'seed4', 'seed5']
MODEL_DIR = '/workspace/LocalRetro/models'
PRED_DIR  = '/workspace/LocalRetro/outputs/raw_prediction'
BASELINE  = '/workspace/LocalRetro/outputs/localretro_baseline.csv'
OUT_CSV   = '/workspace/LocalRetro/outputs/deep_ensemble_signals.csv'
DEFAULT_CKPT = f'{MODEL_DIR}/LocalRetro_USPTO_50K.pth'

def parse_top1_score(pred_file):
    scores = []
    with open(pred_file) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                scores.append(np.nan); continue
            m = re.search(r'\(.*?,\s*\d+,\s*\d+,\s*([\d.]+)\)', parts[2])
            scores.append(float(m.group(1)) if m else np.nan)
    return np.array(scores)

# Step 1: Run inference for each seed
for seed in SEEDS:
    out = f'{PRED_DIR}/LocalRetro_USPTO_50K_{seed}.txt'
    if os.path.exists(out):
        print(f"Already exists, skipping: {seed}"); continue

    seed_ckpt = f'{MODEL_DIR}/LocalRetro_USPTO_50K_{seed}.pth'
    print(f"\nRunning inference for {seed}...")

    # Backup existing default checkpoint if present
    backup = f'{MODEL_DIR}/LocalRetro_USPTO_50K_backup.pth'
    if os.path.exists(DEFAULT_CKPT):
        os.rename(DEFAULT_CKPT, backup)

    # Put seed checkpoint in default location
    os.symlink(seed_ckpt, DEFAULT_CKPT)

    # Run inference
    os.system('cd /workspace/LocalRetro/scripts && python Test.py -d USPTO_50K 2>&1')

    # Remove symlink, restore backup
    os.remove(DEFAULT_CKPT)
    if os.path.exists(backup):
        os.rename(backup, DEFAULT_CKPT)

    # Rename output
    default_out = f'{PRED_DIR}/LocalRetro_USPTO_50K.txt'
    if os.path.exists(default_out):
        os.rename(default_out, out)
        print(f"  Saved: {out}")
    else:
        print(f"  WARNING: output not found for {seed}")

# Step 2: Parse scores
print("\nParsing predictions...")
all_scores = []
for seed in SEEDS:
    pred_file = f'{PRED_DIR}/LocalRetro_USPTO_50K_{seed}.txt'
    if not os.path.exists(pred_file):
        print(f"MISSING: {pred_file}"); continue
    s = parse_top1_score(pred_file)
    print(f"  {seed}: {len(s)} preds, mean={np.nanmean(s):.4f}")
    all_scores.append(s)

scores_arr = np.array(all_scores)  # (4, 5007)
mean_score = np.nanmean(scores_arr, axis=0)
std_score  = np.nanstd(scores_arr, axis=0)

# Step 3: Merge and save
baseline = pd.read_csv(BASELINE)
baseline['ensemble_mean_score'] = mean_score
baseline['ensemble_std_score']  = std_score
baseline['ensemble_uq']         = 1 - mean_score
baseline.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")

# Step 4: AUROC
def auroc(score, label):
    s=np.asarray(score,float); l=np.asarray(label,int)
    n1=l.sum(); n0=len(l)-n1
    if n1==0 or n0==0: return np.nan
    return (stats.rankdata(s)[l==1].sum()-n1*(n1+1)/2)/(n1*n0)

def auroc_ci(score, label, n_boot=2000, seed=0):
    s=np.asarray(score,float); l=np.asarray(label,int)
    rng=np.random.default_rng(seed); idx=np.arange(len(l)); vals=[]
    for _ in range(n_boot):
        b=rng.choice(idx,len(idx),replace=True)
        if l[b].sum() in (0,len(b)): continue
        vals.append(auroc(s[b],l[b]))
    lo,hi=np.percentile(vals,[2.5,97.5])
    return auroc(s,l),lo,hi

errors = baseline['error'].values
print("\n=== OVERALL ===")
for name, sig in [('Baseline (1-top_score)', baseline['uq'].values),
                  ('Ensemble mean uq',       1-mean_score),
                  ('Ensemble std (disagree)',  std_score)]:
    a,lo,hi = auroc_ci(sig, errors)
    print(f"  {name:<30} AUROC={a:.4f} [{lo:.3f},{hi:.3f}]")

print("\n=== VERY-RARE TIER ===")
vr = baseline[baseline['tier'].str.contains('very rare', case=False, na=False)]
vi = vr.index
print(f"  n={len(vr)}")
for name, sig in [('Baseline',          vr['uq'].values),
                  ('Ensemble mean uq',  (1-mean_score)[vi]),
                  ('Ensemble std',       std_score[vi])]:
    a,lo,hi = auroc_ci(sig, errors[vi])
    print(f"  {name:<30} AUROC={a:.4f} [{lo:.3f},{hi:.3f}]")
print("\nDone. Check ensemble std in very-rare tier — this tests the MC dropout hypothesis.")

# Fix: merge tier info from OOD file
import pandas as pd, numpy as np
from scipy import stats

df = pd.read_csv('/workspace/LocalRetro/outputs/deep_ensemble_signals.csv')
ood = pd.read_csv('/workspace/LocalRetro/outputs/ood_frequency_analysis.csv')
df = df.merge(ood[['mol_idx','bucket','frequency']], on='mol_idx')
df.rename(columns={'bucket':'tier'}, inplace=True)

def auroc(score, label):
    s=np.asarray(score,float); l=np.asarray(label,int)
    n1=l.sum(); n0=len(l)-n1
    if n1==0 or n0==0: return np.nan
    return (stats.rankdata(s)[l==1].sum()-n1*(n1+1)/2)/(n1*n0)

def auroc_ci_bonf(score, label, n_signals=3, n_boot=2000, seed=0):
    s=np.asarray(score,float); l=np.asarray(label,int)
    rng=np.random.default_rng(seed); idx=np.arange(len(l)); vals=[]
    for _ in range(n_boot):
        b=rng.choice(idx,len(idx),replace=True)
        if l[b].sum() in (0,len(b)): continue
        vals.append(auroc(s[b],l[b]))
    alpha=0.05/n_signals
    lo,hi=np.percentile(vals,[100*alpha/2,100*(1-alpha/2)])
    return auroc(s,l),lo,hi

errors = df['error'].values
print("\n=== VERY-RARE TIER (n=100) ===")
vr = df[df['tier']=='1) very rare (1-5)']
vi = vr.index
print(f"n={len(vr)}")
for name, sig in [('Baseline', vr['uq'].values),
                  ('Ensemble mean uq', vr['ensemble_uq'].values),
                  ('Ensemble std', vr['ensemble_std_score'].values)]:
    a,lo,hi = auroc_ci_bonf(sig, errors[vi])
    excl = "EXCLUDES 0.5" if lo>0.5 else "spans 0.5"
    print(f"  {name:<25} AUROC={a:.4f} Bonf CI=[{lo:.3f},{hi:.3f}] {excl}")

print("\n=== ALL TIERS ===")
for t in sorted(df['tier'].unique()):
    g=df[df['tier']==t]
    print(f"  {t:<25} n={len(g):>5}  base={auroc(g['uq'].values,g['error'].values):.3f}  "
          f"ens_mean={auroc(g['ensemble_uq'].values,g['error'].values):.3f}  "
          f"ens_std={auroc(g['ensemble_std_score'].values,g['error'].values):.3f}")
