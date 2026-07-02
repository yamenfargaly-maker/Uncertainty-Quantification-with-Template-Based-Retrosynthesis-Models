#!/bin/bash
# MHNreact deep ensemble training - mirrors LocalRetro's ensemble protocol:
#   - Same architecture and hyperparameters as the baseline (seed 0) run
#   - Only the random seed differs across ensemble members
#   - Same early-stopping discipline already verified for seed 0
#     (patience=5, independently confirmed against patience=10 with no
#      improvement found - see the seed-0 verification run)
#   - Trains sequentially (one seed at a time) to avoid GPU memory contention
#
# LocalRetro's ensemble used seeds {2,3,4,5} as the 4 new members (original
# seed 1 checkpoint was lost, so its baseline CSV stood in for "seed 1").
# MHNreact's baseline is seed 0, so the 4 new members here are {1,2,3,4}.

cd /workspace/mhn-react

for SEED in 1 2 3 4; do
  echo "=== Training MHNreact ensemble seed ${SEED} ==="
  nohup python -m mhnreact.train \
    --model_type=mhn \
    --fp_size=4096 \
    --fp_type morgan \
    --template_fp_type rdk \
    --concat_rand_template_thresh 1 \
    --exp_name mhn_ens_seed${SEED} \
    --dataset_type 50k \
    --csv_path ./data/USPTO_50k_MHN_prepro.csv.gz \
    --ssretroeval False \
    --save_model True \
    --epochs 50 \
    --seed ${SEED} \
    > mhn_ens_seed${SEED}_train_log.txt 2>&1

  echo "=== Seed ${SEED} finished. Best checkpoint: ==="
  ls -la data/model/ | grep "mhn_best_checkpoint_mhn_ens_seed${SEED}"
  grep -E "Early stopping triggered|Best val_loss" mhn_ens_seed${SEED}_train_log.txt
  echo ""
done

echo "=== All 4 ensemble seeds trained. Checkpoints: ==="
ls -la data/model/ | grep "mhn_best_checkpoint_mhn_ens_seed"
