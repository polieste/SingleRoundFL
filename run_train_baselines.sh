#!/bin/bash

DATA_PATH="/home/khoi.ho/ML709/PolypGen2021_MultiCenterData_v3"
DATASET_CLASS="PolypGenFLDataset"
EPOCHS=50
BATCH_SIZE=8
LR=1e-4

CENTERS=("1" "2" "3" "4" "5" "all")
CENTERS=("all")
MODELS=("Unet" "UnetPlusPlus" "DeepLabV3" "FPN" "PAN" "Segformer")
# MODELS=("DeepLabV3" "PAN")

for MODEL in "${MODELS[@]}"; do
  for CENTER in "${CENTERS[@]}"; do
    echo "Running MODEL=${MODEL}, CENTER=${CENTER}"

    python train_baselines.py \
      --dataset_class "${DATASET_CLASS}" \
      --data_path "${DATA_PATH}" \
      --center "${CENTER}" \
      --model_name "${MODEL}" \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --lr "${LR}" \
      --save_dir "weights" \
      --save_name "${MODEL}_${DATASET_CLASS}_C${CENTER}.pth" \
      --use_amp
  done
done