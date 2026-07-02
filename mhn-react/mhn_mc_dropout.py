"""
MC Dropout for MHNreact - mirrors LocalRetro's protocol:
  - Reactivate only nn.Dropout submodules at inference (rest of model stays eval())
  - 20 stochastic forward passes per molecule
  - Derive: mean_top_score, std_top_score, agreement_rate, predictive_entropy
  - Correctness (`error`) stays FIXED to the baseline (seed 0) evaluation -
    this script does NOT re-run rdchiral correctness checking (~10 min x 20
    passes would be prohibitive and is unnecessary: only confidence signals
    vary under dropout, not which molecules are "correct").

Model loading is copied verbatim from mhn_ood_eval_full.py to guarantee the
exact same architecture/config is used.
"""
import sys, numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
sys.path.insert(0, "/workspace/mhn-react")
from mhnreact.molutils import convert_smiles_to_fp
from mhnreact.model import MHN, ModelConfig

CKPT       = "/workspace/mhn-react/data/model/mhn_best_checkpoint_mhn_uspto50k_es_1782942938.pt"
CSV_PATH   = "/workspace/mhn-react/data/USPTO_50k_MHN_prepro.csv.gz"
BASELINE   = "/workspace/mhn-react/mhn_baseline_seed0.csv"
OUT_CSV    = "/workspace/mhn-react/mhn_mc_dropout_signals.csv"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH      = 256
N_PASSES   = 20

print("Loading data...")
df = pd.read_csv(CSV_PATH)
test_df = df[df["split"] == "test"].reset_index(drop=True)

print("Loading checkpoint...")
full_state = torch.load(CKPT, map_location=DEVICE)
templates_tensor = full_state.pop("templates+noise")
n_templates = templates_tensor.shape[0]
fp_size = full_state["mol_encoder.W_0.weight"].shape[1]
asso_dim = full_state["mol_encoder.W_0.weight"].shape[0]
print(f"  fp_size={fp_size}  asso_dim={asso_dim}  n_templates={n_templates}")

print("Building model...")
cfg = ModelConfig(fp_size=fp_size, hopf_input_size=fp_size, hopf_output_size=None,
    hopf_asso_dim=asso_dim, hopf_beta=0.05, hopf_num_heads=1, hopf_n_layers=1,
    hopf_association_activation="None", hopf_pooling_operation_head="mean",
    pooling_operation_state_embedding="mean", pooling_operation_head="max",
    mol_encoder_layers=1, temp_encoder_layers=1, encoder_af="ReLU",
    norm_input=True, norm_asso=True, dropout=0.2, num_templates=n_templates,
    device=DEVICE, fp_type="morgan", template_fp_type="rdk")
model = MHN(config=cfg)
missing, unexpected = model.load_state_dict(full_state, strict=False)
print(f"  Missing: {missing}  Unexpected: {unexpected}")
model.templates = templates_tensor.float().to(DEVICE)
model = model.to(DEVICE)
model.eval()

# Reactivate only nn.Dropout submodules - rest of model (including any other
# eval-sensitive layers) stays in eval mode, matching LocalRetro's approach.
n_dropout_modules = 0
for m in model.modules():
    if isinstance(m, nn.Dropout):
        m.train()
        n_dropout_modules += 1
print(f"  Model loaded OK. Reactivated {n_dropout_modules} nn.Dropout submodules.")
if n_dropout_modules == 0:
    print("  WARNING: no nn.Dropout submodules found - MC dropout will be a no-op.")

print("Computing fingerprints...")
X_fp = convert_smiles_to_fp(test_df["prod_smiles"].tolist(), fp_size=fp_size, which="morgan", radius=2)
X_fp = torch.tensor(X_fp, dtype=torch.float32)
n_mol = len(X_fp)
print(f"  FP shape: {X_fp.shape}")

print(f"Running {N_PASSES} stochastic forward passes...")
pass_top_scores = np.zeros((N_PASSES, n_mol), dtype=np.float32)
pass_top_ids = np.zeros((N_PASSES, n_mol), dtype=np.int64)
prob_sum = np.zeros((n_mol, n_templates), dtype=np.float64)

with torch.no_grad():
    for p in range(N_PASSES):
        offset = 0
        for i in range(0, n_mol, BATCH):
            b = X_fp[i:i + BATCH].to(DEVICE)
            probs = F.softmax(model.forward(b), dim=-1)
            ts, ti = probs.max(dim=-1)
            bs = probs.shape[0]
            pass_top_scores[p, offset:offset + bs] = ts.cpu().numpy()
            pass_top_ids[p, offset:offset + bs] = ti.cpu().numpy()
            prob_sum[offset:offset + bs] += probs.cpu().numpy().astype(np.float64)
            offset += bs
        print(f"  pass {p + 1}/{N_PASSES} done, mean_top_score={pass_top_scores[p].mean():.4f}")

print("Computing derived signals...")
mean_top_score = pass_top_scores.mean(axis=0)
std_top_score = pass_top_scores.std(axis=0)

# Consensus top-1 = the modal (most frequent) predicted template id across
# the N_PASSES stochastic passes, per molecule. (Computed manually rather
# than via scipy.stats.mode to avoid version-dependent keepdims behavior.)
consensus_id = np.array([
    np.bincount(pass_top_ids[:, i]).argmax() for i in range(n_mol)
])
agreement_rate = (pass_top_ids == consensus_id[None, :]).mean(axis=0)

mean_probs = prob_sum / N_PASSES
eps = 1e-12
predictive_entropy = -(mean_probs * np.log(mean_probs + eps)).sum(axis=1)

baseline = pd.read_csv(BASELINE).sort_values("mol_idx").reset_index(drop=True)
assert len(baseline) == n_mol, "baseline row count does not match test set size"

out = baseline[["mol_idx", "top_score", "error", "frequency", "tier"]].copy()
out["mean_top_score"] = mean_top_score
out["std_top_score"] = std_top_score
out["agreement_rate"] = agreement_rate
out["predictive_entropy"] = predictive_entropy
out.to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}")

# --- AUROC evaluation, matching LocalRetro Table 1/2 structure ---
def auroc(score, label):
    s = np.asarray(score, float); l = np.asarray(label, int)
    n1 = l.sum(); n0 = len(l) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (stats.rankdata(s)[l == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def auroc_ci(score, label, n_boot=2000, seed=0, alpha=0.05):
    s = np.asarray(score, float); l = np.asarray(label, int)
    rng = np.random.default_rng(seed); idx = np.arange(len(l)); vals = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if l[b].sum() in (0, len(b)):
            continue
        vals.append(auroc(s[b], l[b]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return auroc(s, l), lo, hi

errors = out["error"].values
print("\n=== OVERALL ===")
signals_overall = [
    ("Baseline (1-top_score)", 1 - out["top_score"].values),
    ("1-mean_top_score (MC avg)", 1 - out["mean_top_score"].values),
    ("std_top_score", out["std_top_score"].values),
    ("1-agreement_rate", 1 - out["agreement_rate"].values),
    ("predictive_entropy", out["predictive_entropy"].values),
]
for name, sig in signals_overall:
    a, lo, hi = auroc_ci(sig, errors)
    rho, p = stats.spearmanr(sig, errors)
    print(f"  {name:<28} AUROC={a:.4f} [{lo:.3f},{hi:.3f}]  Spearman={rho:.4f} (p={p:.2e})")

print("\n=== VERY-RARE TIER (Bonferroni, n_signals=5) ===")
vr = out[out["tier"].astype(str).str.contains("very rare", case=False, na=False)]
print(f"  n = {len(vr)}")
vr_signals = [
    ("Baseline", 1 - vr["top_score"].values),
    ("1-mean_top_score", 1 - vr["mean_top_score"].values),
    ("std_top_score", vr["std_top_score"].values),
    ("1-agreement_rate", 1 - vr["agreement_rate"].values),
    ("predictive_entropy", vr["predictive_entropy"].values),
]
for name, sig in vr_signals:
    a, lo_raw, hi_raw = auroc_ci(sig, vr["error"].values, alpha=0.05)
    _, lo_b, hi_b = auroc_ci(sig, vr["error"].values, alpha=0.05 / 5)
    print(f"  {name:<20} AUROC={a:.4f}  raw_CI=[{lo_raw:.3f},{hi_raw:.3f}]  bonf_CI=[{lo_b:.3f},{hi_b:.3f}]")

print("\nDone.")
