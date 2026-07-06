"""
Extension 1: Mondrian rare-tier expansion, using USPTO-MIT (479,035 reactions:
409,035 train + 40,000 test + 30,000 valid; Jin et al. 2017) as the source of
"more reactions" to run through the already-trained MHNreact model.

No new training. No class labels needed (per the design note - Extension 1's
tier assignment is keyed on predicted-template frequency, not reaction class).
None of USPTO-MIT's reactions were used to train the MHNreact model (which was
trained only on USPTO-50K), so all three USPTO-MIT files - its own train,
test, AND valid splits - are equally valid "new" data for this purpose; USPTO-
MIT's own split labels are irrelevant here and all three are pooled together.

USPTO-MIT's line format differs from the CSV format used elsewhere in this
project: each line is `reaction_smiles bond_edit_annotations`, space-
separated, no header, no id/class columns. This script parses that format
directly rather than assuming a CSV.

Two-stage design, same as before:
  Stage 1 (fast, all ~479K reactions): fingerprint + forward pass + tier
    assignment. No correctness evaluation yet.
  Stage 2 (slower, filtered subset only): rdchiral correctness evaluation,
    run ONLY on population-1/2 (very-rare and rare bin) rows.

Run on the MHNreact pod from /workspace/mhn-react.
"""
import sys, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger
from rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, "/workspace/mhn-react")
from mhnreact.molutils import convert_smiles_to_fp
from mhnreact.model import MHN, ModelConfig

CKPT = "/workspace/mhn-react/data/model/mhn_best_checkpoint_mhn_uspto50k_es_1782942938.pt"
ORIGINAL_TRAIN_CSV = "/workspace/mhn-react/data/USPTO_50k_MHN_prepro.csv.gz"
USPTO_MIT_FILES = [
    "/workspace/mhn-react/train.txt",
    "/workspace/mhn-react/test.txt",
    "/workspace/mhn-react/valid.txt",
]
STAGE1_OUT = "/workspace/mhn-react/uspto_mit_stage1_tiers.csv"
STAGE2_OUT = "/workspace/mhn-react/uspto_mit_rare_tier_expansion.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 256

def assign_tier(f):
    if f == 0:
        return "3) novel (freq=0, EXCLUDE from rare-tier pool)"
    elif f <= 5:
        return "1) very rare (1-5)"
    elif f <= 20:
        return "2) rare (6-20)"
    elif f <= 100:
        return "3) common (21-100)"
    else:
        return "4) very common (100+)"

def load_uspto_mit(paths):
    """Parses 'reaction_smiles bond_edit_annotations' line format into a
    DataFrame with prod_smiles / reactants_can, matching the schema the rest
    of this project's scripts already expect."""
    rows = []
    for path in paths:
        print(f"  Reading {path} ...")
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.rsplit(" ", 1)
                if len(parts) != 2:
                    continue
                rxn_smiles, edits = parts
                if ">>" not in rxn_smiles:
                    continue
                reactants, product = rxn_smiles.split(">>", 1)
                rows.append({"reactants_can": reactants, "prod_smiles": product, "edits": edits})
    return pd.DataFrame(rows)

# ============================================================
# STAGE 1: fingerprint + predict + tier-assign everything (fast)
# ============================================================
print("=== STAGE 1: prediction + tier assignment (USPTO-MIT, all reactions) ===")

print("Loading original template frequencies (from the ORIGINAL 40,008-reaction "
      "training set - fixed, does not change)...")
orig = pd.read_csv(ORIGINAL_TRAIN_CSV)
orig_train = orig[orig["split"] == "train"]
template_freq = orig_train["label"].astype(int).value_counts().to_dict()
print(f"  {len(template_freq)} templates have nonzero original training frequency.")

print("Loading USPTO-MIT (all three files pooled - none were used to train "
      "the MHNreact model, so USPTO-MIT's own train/test/valid split is "
      "irrelevant here):")
new_df = load_uspto_mit(USPTO_MIT_FILES)
n_before_cleaning = len(new_df)
print(f"  Loaded {n_before_cleaning} reactions total (before data hygiene pass).")

print("Running data hygiene pass (no-product / single-ion / sub-5-atom reactions "
      "removed)...")

def heavy_atom_count(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return m.GetNumHeavyAtoms() if m else 0
    except Exception:
        return 0

def is_junk(row):
    prod = row["prod_smiles"]
    react = row["reactants_can"]
    if not prod or pd.isna(prod) or Chem.MolFromSmiles(prod) is None:
        return True
    react_frags = react.split(".") if react and not pd.isna(react) else []
    if len(react_frags) == 1 and heavy_atom_count(react_frags[0]) <= 1:
        return True
    if heavy_atom_count(prod) < 5:
        return True
    react_total_atoms = sum(heavy_atom_count(f) for f in react_frags)
    if react_frags and react_total_atoms < 5:
        return True
    return False

junk_mask = new_df.apply(is_junk, axis=1)
n_junk = junk_mask.sum()
new_df = new_df[~junk_mask].reset_index(drop=True)
print(f"  Removed {n_junk} junk reactions ({100*n_junk/n_before_cleaning:.1f}%), "
      f"{len(new_df)} remain.")

print("Loading trained checkpoint (no training, inference only)...")
full_state = torch.load(CKPT, map_location=DEVICE)
templates_tensor = full_state.pop("templates+noise")
n_templates = templates_tensor.shape[0]
fp_size = full_state["mol_encoder.W_0.weight"].shape[1]
asso_dim = full_state["mol_encoder.W_0.weight"].shape[0]

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

print("Computing fingerprints (this is the largest step - ~479K molecules)...")
X_fp = convert_smiles_to_fp(new_df["prod_smiles"].tolist(), fp_size=fp_size, which="morgan", radius=2)
X_fp = torch.tensor(X_fp, dtype=torch.float32)

print("Running inference (predicted top-1 template + score, no training)...")
all_scores, all_ids = [], []
with torch.no_grad():
    for i in range(0, len(X_fp), BATCH):
        b = X_fp[i:i+BATCH].to(DEVICE)
        probs = F.softmax(model.forward(b), dim=-1)
        ts, ti = probs.max(dim=-1)
        all_scores.append(ts.cpu().numpy())
        all_ids.append(ti.cpu().numpy())
        if i % (BATCH*100) == 0:
            print(f"  {i+len(b)}/{len(X_fp)}")
top_scores = np.concatenate(all_scores)
top_ids = np.concatenate(all_ids)

new_df["pred_template_id"] = top_ids
new_df["top_score"] = top_scores
new_df["uq"] = 1 - new_df["top_score"]
new_df["pred_template_orig_freq"] = new_df["pred_template_id"].map(lambda t: template_freq.get(int(t), 0))
new_df["tier"] = new_df["pred_template_orig_freq"].apply(assign_tier)

print("\nTier distribution (Stage 1, before correctness evaluation):")
print(new_df["tier"].value_counts())

new_df.to_csv(STAGE1_OUT, index=False)
print(f"\nSaved Stage 1 output: {STAGE1_OUT}")

# ============================================================
# STAGE 2: correctness evaluation, ONLY on the rare-tier population
# ============================================================
print("\n=== STAGE 2: correctness evaluation (rare-tier population only) ===")

rare_pop = new_df[new_df["tier"] == "2) rare (6-20)"].reset_index(drop=True)
very_rare_pop = new_df[new_df["tier"] == "1) very rare (1-5)"].reset_index(drop=True)
target_pop = pd.concat([very_rare_pop, rare_pop], ignore_index=True)
print(f"  Rare-tier population to evaluate: {len(target_pop)} reactions "
      f"(very-rare: {len(very_rare_pop)}, rare: {len(rare_pop)})")
print(f"  (Excluded: common/very-common - not needed; population-3 novel/freq=0 "
      f"- excluded by design)")

label2smarts = dict(zip(orig["label"].astype(int), orig["reaction_smarts"]))

def strip_atom_map(smiles):
    """Remove atom-map numbers before canonicalizing - required because
    USPTO-MIT's raw SMILES retain atom maps, unlike MHNreact's own
    preprocessed reactants_can column (already stripped at source), and
    rdchiral's predicted output is always unmapped. Comparing a mapped
    ground truth against unmapped predictions silently fails every time
    without this step - this is exactly what caused the 0% result."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol

def canonicalize(smi):
    try:
        mol = strip_atom_map(smi)
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None

def apply_template(prod, smarts):
    try:
        return rdchiralRun(rdchiralReaction(smarts), rdchiralReactants(prod), combine_enantiomers=False)
    except Exception:
        return []

print("Evaluating correctness on the rare-tier population (this is the slow step)...")
errors = []
for i, row in target_pop.iterrows():
    if i % 500 == 0:
        print(f"  {i}/{len(target_pop)}")
    smarts = label2smarts.get(int(row["pred_template_id"]))
    if not smarts:
        errors.append(1); continue
    results = apply_template(row["prod_smiles"], smarts)
    if not results:
        errors.append(1); continue
    gt_c = ".".join(sorted(p for p in [canonicalize(s) for s in row["reactants_can"].split(".")] if p))
    correct = any(".".join(sorted(p for p in [canonicalize(s) for s in r.split(".")] if p)) == gt_c
                  for r in results)
    errors.append(0 if correct else 1)

target_pop["error"] = errors
target_pop["mol_idx"] = range(len(target_pop))
target_pop[["mol_idx", "top_score", "uq", "error", "pred_template_orig_freq", "tier"]].to_csv(
    STAGE2_OUT, index=False)

print(f"\nSaved Stage 2 output (ready to merge into the Mondrian calibration pool): {STAGE2_OUT}")
print(f"Top-1 accuracy on expanded rare-tier population: {(1 - target_pop['error'].mean())*100:.2f}%")
print("\nDone. Next step: merge this with the existing mhn_baseline_seed0.csv rare-tier "
      "rows and re-run the Mondrian harness with the larger calibration pool.")
