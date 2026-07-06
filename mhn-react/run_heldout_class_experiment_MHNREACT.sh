#!/bin/bash
# Held-out-reaction-class experiment for MHNreact.
# For each held-out class (4, 8):
#   1. Runs prepare_heldout_class_mhn.py to build the two CSVs (already
#      done once if you've already run it - this re-checks it exists).
#   2. Trains from scratch on the _ood.csv.gz version (class N excluded
#      from train/valid; same verified patience-5 early-stopping protocol
#      used throughout this project).
#   3. Runs inference on the OOD test set (all of class N's reactions).
#   4. Runs inference again on the in-distribution test set, reusing the
#      same trained checkpoint, by swapping CSV_PATH only.
#
# Upload prepare_heldout_class_mhn.py to /workspace/mhn-react/ before running
# this, or let this script call it directly (it does, in Step 1).

set -e
set -o pipefail
cd /workspace/mhn-react

echo "=== Step 1: Preparing held-out-class data (both classes) ==="
python3 prepare_heldout_class_mhn.py

for CLASS in 4 8; do
  echo "================================================================"
  echo "=== Held-out class ${CLASS} ==="
  echo "================================================================"

  OOD_CSV="data/USPTO_50k_MHN_prepro_heldout_class${CLASS}_ood.csv.gz"
  INDIST_CSV="data/USPTO_50k_MHN_prepro_heldout_class${CLASS}_indist.csv.gz"
  EXP_NAME="mhn_heldout_class${CLASS}"

  echo "=== Step 2: Training from scratch (class ${CLASS} excluded, patience=5) ==="
  python -m mhnreact.train \
    --model_type=mhn \
    --fp_size=4096 \
    --fp_type morgan \
    --template_fp_type rdk \
    --concat_rand_template_thresh 1 \
    --exp_name "${EXP_NAME}" \
    --dataset_type 50k \
    --csv_path "${OOD_CSV}" \
    --ssretroeval False \
    --save_model True \
    --epochs 50 \
    --seed 0 \
    2>&1 | tee "${EXP_NAME}_train_log.txt"

  CKPT_PATH=$(ls -t data/model/mhn_best_checkpoint_${EXP_NAME}_*.pt 2>/dev/null | head -1)
  if [ -z "$CKPT_PATH" ]; then
    echo "ERROR: no checkpoint found for ${EXP_NAME}, skipping evaluation."
    continue
  fi
  echo "  Checkpoint: ${CKPT_PATH}"

  echo "=== Step 3: Evaluating on OOD test set (class ${CLASS}) ==="
  cp mhn_ood_eval_full.py mhn_ood_eval_full.py.bak_heldout

  python3 - "$CKPT_PATH" "$OOD_CSV" "outputs/mhn_baseline_heldout_class${CLASS}_ood.csv" <<'PYEOF'
import re, sys
ckpt_path, csv_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open("mhn_ood_eval_full.py") as f:
    content = f.read()
content = re.sub(r'CKPT\s*=\s*"[^"]*"', f'CKPT     = "{ckpt_path}"', content, count=1)
content = re.sub(r'CSV_PATH\s*=\s*"[^"]*"', f'CSV_PATH = "{csv_path}"', content, count=1)
content = re.sub(r'OUT_CSV\s*=\s*"[^"]*"', f'OUT_CSV  = "{out_path}"', content, count=1)
with open("mhn_ood_eval_full.py", "w") as f:
    f.write(content)
PYEOF
  mkdir -p outputs
  python mhn_ood_eval_full.py

  echo "=== Step 4: Evaluating on in-distribution test set (class ${CLASS}), same checkpoint ==="
  cp mhn_ood_eval_full.py.bak_heldout mhn_ood_eval_full.py
  python3 - "$CKPT_PATH" "$INDIST_CSV" "outputs/mhn_baseline_heldout_class${CLASS}_indist.csv" <<'PYEOF'
import re, sys
ckpt_path, csv_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open("mhn_ood_eval_full.py") as f:
    content = f.read()
content = re.sub(r'CKPT\s*=\s*"[^"]*"', f'CKPT     = "{ckpt_path}"', content, count=1)
content = re.sub(r'CSV_PATH\s*=\s*"[^"]*"', f'CSV_PATH = "{csv_path}"', content, count=1)
content = re.sub(r'OUT_CSV\s*=\s*"[^"]*"', f'OUT_CSV  = "{out_path}"', content, count=1)
with open("mhn_ood_eval_full.py", "w") as f:
    f.write(content)
PYEOF
  python mhn_ood_eval_full.py

  cp mhn_ood_eval_full.py.bak_heldout mhn_ood_eval_full.py
  rm mhn_ood_eval_full.py.bak_heldout

  echo "=== Held-out class ${CLASS} complete. ==="
  echo "  OOD result CSV:    outputs/mhn_baseline_heldout_class${CLASS}_ood.csv"
  echo "  In-dist result CSV: outputs/mhn_baseline_heldout_class${CLASS}_indist.csv"
  echo "  Train log: ${EXP_NAME}_train_log.txt"
  echo ""
done

echo "=== All MHNreact held-out-class experiments complete. ==="
