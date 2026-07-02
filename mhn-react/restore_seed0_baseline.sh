#!/bin/bash
# Restores the correct seed-0 (retrained, early-stopped) baseline CSV,
# which was accidentally overwritten by run_mhn_ensemble_inference.sh
# looping over seeds 1-4. The seed-0 checkpoint itself is untouched on
# disk, so this is a clean, deterministic re-run - no data is actually lost.

set -e
cd /workspace/mhn-react

SEED0_CKPT="data/model/mhn_best_checkpoint_mhn_uspto50k_es_1782942938.pt"

if [ ! -f "$SEED0_CKPT" ]; then
  echo "ERROR: seed-0 checkpoint not found at $SEED0_CKPT"
  echo "Run: ls data/model/ | grep mhn_uspto50k_es    to find the correct path,"
  echo "then edit SEED0_CKPT in this script."
  exit 1
fi

cp mhn_ood_eval_full.py mhn_ood_eval_full.py.bak2

python3 << PYEOF
import re
with open("mhn_ood_eval_full.py") as f:
    content = f.read()
content = re.sub(r'CKPT\s*=\s*"[^"]*"', 'CKPT     = "${SEED0_CKPT}"', content, count=1)
with open("mhn_ood_eval_full.py", "w") as f:
    f.write(content)
PYEOF

echo "=== Confirming CKPT points at seed 0: ==="
grep -n "CKPT" mhn_ood_eval_full.py | head -2

python mhn_ood_eval_full.py

cp mhn_baseline.csv mhn_baseline_seed0.csv
echo "Restored and saved as mhn_baseline_seed0.csv"

cp mhn_ood_eval_full.py.bak2 mhn_ood_eval_full.py
rm mhn_ood_eval_full.py.bak2
echo "Eval script restored to its normal state."
