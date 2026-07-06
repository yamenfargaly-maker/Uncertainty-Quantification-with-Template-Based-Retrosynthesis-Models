#!/bin/bash
# Held-out-reaction-class experiment for LocalRetro.
# For each held-out class (4, 8):
#   1. Sets up a self-contained dataset folder (data/USPTO_50K_heldout_classN/)
#      so the original data/USPTO_50K/ is never touched.
#   2. Extracts templates from the held-out training set (class N's templates
#      are absent by construction, since extraction only sees this filtered
#      training data).
#   3. Preprocesses train/val + the OOD test set (all of class N's reactions).
#   4. Trains from scratch (same defaults as your original run: patience=5,
#      50 max epochs).
#   5. Runs the full inference chain (Test.py -> Decode_predictions.py ->
#      localretro_baseline_calibration.py) on the OOD test set, backs up
#      each intermediate output.
#   6. Swaps in the in-distribution comparison test set, re-preprocesses
#      just that split (reusing the already-extracted templates), and
#      re-runs the same inference chain, saving to distinct filenames.
#
# All paths verified directly against your actual Extract_from_train_data.py,
# Run_preprocessing.py, Train.py, Test.py, Decode_predictions.py, and
# localretro_baseline_calibration.py source.
#
# Upload the 8 raw_*_heldout_class*.csv / raw_test_ood_class*.csv files to
# /workspace/LocalRetro/ before running this.

set -e
set -o pipefail
cd /workspace/LocalRetro

run_inference_chain () {
  local DATASET=$1
  local TAG=$2   # "ood" or "indist"

  cd scripts
  python Test.py -d "${DATASET}"
  python Decode_predictions.py -d "${DATASET}"
  cd ..

  # Back up this run's outputs before the next Test.py call overwrites them
  cp "outputs/raw_prediction/LocalRetro_${DATASET}.txt" \
     "outputs/raw_prediction/LocalRetro_${DATASET}_${TAG}.txt"
  cp "outputs/decoded_prediction/LocalRetro_${DATASET}.txt" \
     "outputs/decoded_prediction/LocalRetro_${DATASET}_${TAG}.txt"

  cd scripts
  python localretro_baseline_calibration.py \
    --decoded "../outputs/decoded_prediction/LocalRetro_${DATASET}_${TAG}.txt" \
    --raw_test "../data/${DATASET}/raw_test.csv" \
    --output "../outputs/localretro_baseline_heldout_${DATASET}_${TAG}.csv"
  cd ..
}

for CLASS in 4 8; do
  echo "================================================================"
  echo "=== Held-out class ${CLASS} ==="
  echo "================================================================"

  DATASET="USPTO_50K_heldout_class${CLASS}"
  DATADIR="data/${DATASET}"
  mkdir -p "${DATADIR}"

  cp raw_train_heldout_class${CLASS}.csv "${DATADIR}/raw_train.csv"
  cp raw_val_heldout_class${CLASS}.csv "${DATADIR}/raw_val.csv"
  cp raw_test_ood_class${CLASS}.csv "${DATADIR}/raw_test.csv"

  echo "=== Step 2: Extracting templates (class ${CLASS} excluded from training) ==="
  cd preprocessing
  python Extract_from_train_data.py -d "${DATASET}"
  cd ..

  echo "=== Step 3: Preprocessing train/val/OOD-test ==="
  cd preprocessing
  python Run_preprocessing.py -d "${DATASET}"
  cd ..

  echo "=== Step 4: Training from scratch (patience=5, up to 50 epochs) ==="
  cd scripts
  python Train.py -d "${DATASET}" 2>&1 | tee "../${DATASET}_train_log.txt"
  cd ..

  echo "=== Step 5: Full inference chain on OOD test set (class ${CLASS}) ==="
  run_inference_chain "${DATASET}" "ood"

  echo "=== Step 6: Swap in in-distribution test set, re-preprocess, re-run inference chain ==="
  cp raw_test_indist_heldout_class${CLASS}.csv "${DATADIR}/raw_test.csv"
  # LocalRetro caches test DGL graphs keyed only by dataset folder name, not
  # by test-set content - must clear this or Test.py silently loads the
  # stale OOD-test graphs (wrong size) instead of regenerating for the new
  # in-distribution test set.
  rm -f "data/saved_graphs/${DATASET}_test_dglgraph.bin"
  cd preprocessing
  python Run_preprocessing.py -d "${DATASET}" -f True
  cd ..
  run_inference_chain "${DATASET}" "indist"

  echo "=== Held-out class ${CLASS} complete. ==="
  echo "  OOD result CSV:    outputs/localretro_baseline_heldout_${DATASET}_ood.csv"
  echo "  In-dist result CSV: outputs/localretro_baseline_heldout_${DATASET}_indist.csv"
  echo "  Train log: ${DATASET}_train_log.txt"
  echo ""
done

echo "=== All held-out-class experiments complete. ==="
