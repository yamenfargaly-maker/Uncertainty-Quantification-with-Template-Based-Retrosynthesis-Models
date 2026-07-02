#!/bin/bash
# Second, INDEPENDENT MHNreact ensemble - seeds 5-8, same protocol as the
# first ensemble (seeds 1-4). Purpose: test whether the ensemble-disagreement
# inversion finding (very-rare tier AUROC < 0.5) replicates with a totally
# different set of seeds, not just the original ensemble - addressing the
# mentor's "across boundaries AND SEEDS" robustness recommendation (Part F).
#
# Same architecture, hyperparameters, and verified patience-5 early-stopping
# protocol as both the seed-0 baseline and the first ensemble.

cd /workspace/mhn-react

for SEED in 5 6 7 8; do
  echo "=== Training MHNreact ensemble-2 seed ${SEED} ==="
  nohup python -m mhnreact.train \
    --model_type=mhn \
    --fp_size=4096 \
    --fp_type morgan \
    --template_fp_type rdk \
    --concat_rand_template_thresh 1 \
    --exp_name mhn_ens2_seed${SEED} \
    --dataset_type 50k \
    --csv_path ./data/USPTO_50k_MHN_prepro.csv.gz \
    --ssretroeval False \
    --save_model True \
    --epochs 50 \
    --seed ${SEED} \
    > mhn_ens2_seed${SEED}_train_log.txt 2>&1

  echo "=== Seed ${SEED} finished. Best checkpoint: ==="
  ls -la data/model/ | grep "mhn_best_checkpoint_mhn_ens2_seed${SEED}"
  grep -E "Early stopping triggered|Best val_loss" mhn_ens2_seed${SEED}_train_log.txt
  echo ""
done

echo "=== All 4 second-ensemble seeds trained. Checkpoints: ==="
ls -la data/model/ | grep "mhn_best_checkpoint_mhn_ens2_seed"
