"""
Reruns ONLY Stage 2 (correctness evaluation) using the already-computed,
still-valid Stage 1 output (uspto_mit_stage1_tiers.csv) - Stage 1's
fingerprinting/prediction/tier-assignment was never affected by the atom-map
bug, so there is no need to redo it.

Fix applied: canonicalize() now strips atom-map numbers before comparison
(matching LocalRetro's original correct implementation). The previous 0%
accuracy result was entirely an artifact of comparing atom-mapped ground
truth against rdchiral's unmapped predicted output - not a real finding.

Run on the MHNreact pod from /workspace/mhn-react.
"""
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants
RDLogger.DisableLog('rdApp.*')

ORIGINAL_TRAIN_CSV = "/workspace/mhn-react/data/USPTO_50k_MHN_prepro.csv.gz"
STAGE1_IN = "/workspace/mhn-react/uspto_mit_stage1_tiers.csv"
STAGE2_OUT = "/workspace/mhn-react/uspto_mit_rare_tier_expansion.csv"

print("Loading Stage 1 output (already valid, not affected by the bug)...")
new_df = pd.read_csv(STAGE1_IN)
print(f"  {len(new_df)} reactions loaded.")

orig = pd.read_csv(ORIGINAL_TRAIN_CSV)
label2smarts = dict(zip(orig["label"].astype(int), orig["reaction_smarts"]))

def strip_atom_map(smiles):
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

rare_pop = new_df[new_df["tier"] == "2) rare (6-20)"].reset_index(drop=True)
very_rare_pop = new_df[new_df["tier"] == "1) very rare (1-5)"].reset_index(drop=True)
target_pop = pd.concat([very_rare_pop, rare_pop], ignore_index=True)
print(f"Rare-tier population to evaluate: {len(target_pop)} reactions "
      f"(very-rare: {len(very_rare_pop)}, rare: {len(rare_pop)})")

print("Evaluating correctness (fixed: atom maps stripped before comparison)...")
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

print(f"\nSaved corrected Stage 2 output: {STAGE2_OUT}")
print(f"Top-1 accuracy on expanded rare-tier population: {(1 - target_pop['error'].mean())*100:.2f}%")
for t in target_pop["tier"].unique():
    g = target_pop[target_pop["tier"] == t]
    print(f"  {t}: n={len(g)}, acc={(1-g['error'].mean())*100:.2f}%")
