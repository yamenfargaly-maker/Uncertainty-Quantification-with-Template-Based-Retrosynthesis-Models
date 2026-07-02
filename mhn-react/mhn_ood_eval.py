import sys, os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/mhn-react")
from mhnreact.molutils import convert_smiles_to_fp
from mhnreact.model import MHN, ModelConfig

CKPT     = "/workspace/mhn-react/data/model/USPTO_50k_mhn_valloss3.195_mhn_uspto50k_1782860869.pt"
CSV_PATH = "/workspace/mhn-react/data/USPTO_50k_MHN_prepro.csv.gz"
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading data...")
df = pd.read_csv(CSV_PATH)
print("Columns:", list(df.columns))
print("Split counts:", df['split'].value_counts().to_dict())
test_df = df[df['split'] == 'test'].reset_index(drop=True)
print(f"Test: {len(test_df)}")

print("\nLoading checkpoint...")
state_dict = torch.load(CKPT, map_location=DEVICE)
print("State dict keys:", list(state_dict.keys()))
num_templates = state_dict['templates+noise'].shape[0]
print("num_templates:", num_templates)

print("\nChecking hist directory...")
hist_dir = "/workspace/mhn-react/data/hist"
for f in os.listdir(hist_dir):
    print(" ", f)

print("\nLabel column sample:")
if 'label' in df.columns:
    print(df[df['split']=='test']['label'].describe())
else:
    print("No 'label' column. Columns:", list(df.columns))
