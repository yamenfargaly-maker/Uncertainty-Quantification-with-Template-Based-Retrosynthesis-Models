#!/bin/bash
# Runs inference for each of the 4 MHNreact ensemble checkpoints, reusing
# your existing mhn_ood_eval_full.py rather than reimplementing inference
# logic (safer - it's already validated against the seed-0 baseline).
#
# For each seed, this:
#   1. Backs up mhn_ood_eval_full.py
#   2. Points CKPT at that seed's best checkpoint
#   3. Runs it, which overwrites mhn_baseline.csv
#   4. Renames the output to mhn_ens_seed{N}_topscore.csv
#   5. Restores the original eval script (pointed back at seed 0) when done
#
# Run this AFTER train_mhn_ensemble.sh has completed for all 4 seeds.

cd /workspace/mhn-react

cp mhn_ood_eval_full.py mhn_ood_eval_full.py.bak

# Back up the current mhn_baseline.csv (should be the seed-0 baseline) BEFORE
# the loop below overwrites it 4 times. This was missing in the first version
# of this script and caused the seed-0 baseline to be lost after a run.
if [ -f mhn_baseline.csv ]; then
  cp mhn_baseline.csv mhn_baseline_seed0_backup_$(date +%s).csv
  echo "Backed up current mhn_baseline.csv before starting ensemble inference."
fi

for SEED in 1 2 3 4; do
  echo "=== Running inference for MHNreact ensemble seed ${SEED} ==="

  CKPT_PATH=$(ls data/model/mhn_best_checkpoint_mhn_ens_seed${SEED}_*.pt 2>/dev/null | head -1)
  if [ -z "$CKPT_PATH" ]; then
    echo "  ERROR: no checkpoint found for seed ${SEED}, skipping."
    continue
  fi
  echo "  Using checkpoint: ${CKPT_PATH}"

  cp mhn_ood_eval_full.py.bak mhn_ood_eval_full.py
  # Swap CKPT to this seed's checkpoint (matches the sed pattern used for
  # the seed-0 retraining eval swap earlier)
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
    cp mhn_baseline.csv mhn_ens_seed${SEED}_topscore.csv
    echo "  Saved: mhn_ens_seed${SEED}_topscore.csv"
  else
    echo "  ERROR: mhn_baseline.csv not produced for seed ${SEED}"
  fi
  echo ""
done

# Restore the original eval script (pointed at seed 0 baseline) so it's
# not left pointing at seed 4's checkpoint
cp mhn_ood_eval_full.py.bak mhn_ood_eval_full.py
rm mhn_ood_eval_full.py.bak

echo "=== Done. Per-seed CSVs: ==="
ls -la mhn_ens_seed*_topscore.csv
