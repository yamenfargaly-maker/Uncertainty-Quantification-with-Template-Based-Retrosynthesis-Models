import sys, os, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from scipy import stats
from rdkit import Chem
from rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants
sys.path.insert(0, "/workspace/mhn-react")
from mhnreact.molutils import convert_smiles_to_fp
from mhnreact.model import MHN, ModelConfig

CKPT     = "/workspace/mhn-react/data/model/mhn_best_checkpoint_mhn_uspto50k_es_1782942938.pt"
CSV_PATH = "/workspace/mhn-react/data/USPTO_50k_MHN_prepro.csv.gz"
OUT_CSV  = "/workspace/mhn-react/mhn_baseline.csv"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
BATCH    = 256

print("Loading data...")
df       = pd.read_csv(CSV_PATH)
test_df  = df[df["split"] == "test"].reset_index(drop=True)
train_df = df[df["split"] == "train"].reset_index(drop=True)
label2smarts  = dict(zip(df["label"].astype(int), df["reaction_smarts"]))
template_freq = train_df["label"].astype(int).value_counts().to_dict()

print("Loading checkpoint...")
full_state       = torch.load(CKPT, map_location=DEVICE)
templates_tensor = full_state.pop("templates+noise")
n_templates      = templates_tensor.shape[0]
fp_size          = full_state["mol_encoder.W_0.weight"].shape[1]
asso_dim         = full_state["mol_encoder.W_0.weight"].shape[0]
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
print("  Model loaded OK")

print("Computing fingerprints...")
X_fp = convert_smiles_to_fp(test_df["prod_smiles"].tolist(), fp_size=fp_size, which="morgan", radius=2)
X_fp = torch.tensor(X_fp, dtype=torch.float32)
print(f"  FP shape: {X_fp.shape}")

print("Running inference...")
all_scores, all_ids = [], []
with torch.no_grad():
    for i in range(0, len(X_fp), BATCH):
        b = X_fp[i:i+BATCH].to(DEVICE)
        probs = F.softmax(model.forward(b), dim=-1)
        ts, ti = probs.max(dim=-1)
        all_scores.append(ts.cpu().numpy())
        all_ids.append(ti.cpu().numpy())
        if i % (BATCH*10) == 0: print(f"  {i+len(b)}/{len(X_fp)}")
top_scores = np.concatenate(all_scores)
top_ids    = np.concatenate(all_ids)
print(f"  top_score mean={top_scores.mean():.4f}")

def canonicalize(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m, canonical=True) if m else None
    except: return None

def apply_template(prod, smarts):
    try:
        return rdchiralRun(rdchiralReaction(smarts), rdchiralReactants(prod), combine_enantiomers=False)
    except: return []

print("Evaluating correctness (~10 min)...")
errors = []
for i, (prod, gt, pid) in enumerate(zip(test_df["prod_smiles"], test_df["reactants_can"], top_ids)):
    if i % 500 == 0: print(f"  {i}/{len(test_df)}")
    smarts = label2smarts.get(int(pid))
    if not smarts: errors.append(1); continue
    results = apply_template(prod, smarts)
    if not results: errors.append(1); continue
    gt_c = ".".join(sorted(p for p in [canonicalize(s) for s in gt.split(".")] if p))
    correct = any(".".join(sorted(p for p in [canonicalize(s) for s in r.split(".")] if p))==gt_c for r in results)
    errors.append(0 if correct else 1)
errors = np.array(errors)
print(f"  Top-1 accuracy: {(1-errors.mean())*100:.2f}%")

def assign_tier(f):
    if f<=5: return "1) very rare (1-5)"
    elif f<=20: return "2) rare (6-20)"
    elif f<=100: return "3) common (21-100)"
    else: return "4) very common (100+)"

gt_labels   = test_df["label"].astype(int).values
frequencies = np.array([template_freq.get(l, 0) for l in gt_labels])
tiers       = [assign_tier(f) for f in frequencies]

out = pd.DataFrame({"mol_idx": range(len(test_df)),
    "top_score": top_scores, "uq": 1-top_scores, "error": errors,
    "frequency": frequencies, "tier": tiers})
out.to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}")

def auroc(score, label):
    s=np.asarray(score,float); l=np.asarray(label,int)
    n1=l.sum(); n0=len(l)-n1
    if n1==0 or n0==0: return np.nan
    return (stats.rankdata(s)[l==1].sum()-n1*(n1+1)/2)/(n1*n0)

print(f"Overall AUROC: {auroc(1-top_scores, errors):.4f}")
print(f"{chr(10)}{chr(84)}ier{chr(32)*22} n     Acc    Mean_UQ   AUROC")
for t in sorted(out["tier"].unique()):
    g = out[out["tier"]==t]
    err_rate = 1 - g["error"].mean(); uq_mean = g["uq"].mean(); ar = auroc(g["uq"], g["error"]); print(f"{t:<25} {len(g):>6} {err_rate:>8.3f} {uq_mean:>10.3f} {ar:>8.3f}")
print("Done.")
