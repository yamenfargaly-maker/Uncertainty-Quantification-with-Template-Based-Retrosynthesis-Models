#!/bin/bash
# Runs inference for each of the second-ensemble's 4 checkpoints (seeds 5-8),
# same approach as the first ensemble's inference script: reuse the existing
# validated mhn_ood_eval_full.py, swap CKPT per seed, save one CSV per seed.
#
# Run this AFTER train_mhn_ensemble2.sh has completed for all 4 seeds.

cd /workspace/mhn-react

cp mhn_ood_eval_full.py mhn_ood_eval_full.py.bak

# Back up current mhn_baseline.csv before this loop overwrites it repeatedly
# (this was the bug fixed after the first ensemble run - keeping the fix here).
if [ -f mhn_baseline.csv ]; then
  cp mhn_baseline.csv mhn_baseline_seed0_backup_$(date +%s).csv
  echo "Backed up current mhn_baseline.csv before starting ensemble-2 inference."
fi

for SEED in 5 6 7 8; do
  echo "=== Running inference for MHNreact ensemble-2 seed ${SEED} ==="

  CKPT_PATH=$(ls data/model/mhn_best_checkpoint_mhn_ens2_seed${SEED}_*.pt 2>/dev/null | head -1)
  if [ -z "$CKPT_PATH" ]; then
    echo "  ERROR: no checkpoint found for seed ${SEED}, skipping."
    continue
  fi
  echo "  Using checkpoint: ${CKPT_PATH}"

  cp mhn_ood_eval_full.py.bak mhn_ood_eval_full.py
  python3 - "$CKPT_PATH" <<'PYEOF'
import re, sys
ckpt_path = sys.argv[1]
with open("mhn_ood_eval_full.py") as f:
    content = f.read()
content = re.sub(
    r'CKPT\s*=\s*"[^"]*"',
    f'CKPT     = "{ckpt_path}"',
    content,
    count=1,
)
with open("mhn_ood_eval_full.py", "w") as f:
    f.write(content)
PYEOF

  grep -n "CKPT" mhn_ood_eval_full.py | head -2
  python mhn_ood_eval_full.py

  if [ -f mhn_baseline.csv ]; then
    cp mhn_baseline.csv mhn_ens2_seed${SEED}_topscore.csv
    echo "  Saved: mhn_ens2_seed${SEED}_topscore.csv"
  else
    echo "  ERROR: mhn_baseline.csv not produced for seed ${SEED}"
  fi
  echo ""
done

cp mhn_ood_eval_full.py.bak mhn_ood_eval_full.py
rm mhn_ood_eval_full.py.bak

echo "=== Done. Per-seed CSVs: ==="
ls -la mhn_ens2_seed*_topscore.csv
