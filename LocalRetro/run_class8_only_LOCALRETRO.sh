#!/bin/bash
# Held-out-class experiment for LocalRetro - Class 8 ONLY.
# Class 4 already completed successfully; this skips it entirely rather
# than re-running the full two-class orchestration script from scratch.

set -e
set -o pipefail
cd /workspace/LocalRetro

DATASET="USPTO_50K_heldout_class8"
DATADIR="data/${DATASET}"
mkdir -p "${DATADIR}"

cp raw_train_heldout_class8.csv "${DATADIR}/raw_train.csv"
cp raw_val_heldout_class8.csv "${DATADIR}/raw_val.csv"
cp raw_test_ood_class8.csv "${DATADIR}/raw_test.csv"

echo "=== Extracting templates (class 8 excluded from training) ==="
cd preprocessing
python Extract_from_train_data.py -d "${DATASET}"
cd ..

echo "=== Preprocessing train/val/OOD-test ==="
cd preprocessing
python Run_preprocessing.py -d "${DATASET}"
cd ..

echo "=== Training from scratch (patience=5, up to 50 epochs) ==="
cd scripts
python Train.py -d "${DATASET}" 2>&1 | tee "../${DATASET}_train_log.txt"
cd ..

echo "=== Inference chain on OOD test set (class 8) ==="
cd scripts
python Test.py -d "${DATASET}"
python Decode_predictions.py -d "${DATASET}"
cd ..
cp "outputs/raw_prediction/LocalRetro_${DATASET}.txt" "outputs/raw_prediction/LocalRetro_${DATASET}_ood.txt"
cp "outputs/decoded_prediction/LocalRetro_${DATASET}.txt" "outputs/decoded_prediction/LocalRetro_${DATASET}_ood.txt"
cd scripts
python localretro_baseline_calibration.py \
  --decoded "../outputs/decoded_prediction/LocalRetro_${DATASET}_ood.txt" \
  --raw_test "../data/${DATASET}/raw_test.csv" \
  --output "../outputs/localretro_baseline_heldout_${DATASET}_ood.csv"
cd ..

echo "=== Swap in-distribution test set, re-preprocess, re-run inference chain ==="
cp raw_test_indist_heldout_class8.csv "${DATADIR}/raw_test.csv"
rm -f "data/saved_graphs/${DATASET}_test_dglgraph.bin"
cd preprocessing
python Run_preprocessing.py -d "${DATASET}" -f True
cd ..

cd scripts
python Test.py -d "${DATASET}"
python Decode_predictions.py -d "${DATASET}"
cd ..
cp "outputs/raw_prediction/LocalRetro_${DATASET}.txt" "outputs/raw_prediction/LocalRetro_${DATASET}_indist.txt"
cp "outputs/decoded_prediction/LocalRetro_${DATASET}.txt" "outputs/decoded_prediction/LocalRetro_${DATASET}_indist.txt"
cd scripts
python localretro_baseline_calibration.py \
  --decoded "../outputs/decoded_prediction/LocalRetro_${DATASET}_indist.txt" \
  --raw_test "../data/${DATASET}/raw_test.csv" \
  --output "../outputs/localretro_baseline_heldout_${DATASET}_indist.csv"
cd ..

echo "=== Class 8 complete. ==="
echo "  OOD:    outputs/localretro_baseline_heldout_${DATASET}_ood.csv"
echo "  Indist: outputs/localretro_baseline_heldout_${DATASET}_indist.csv"
