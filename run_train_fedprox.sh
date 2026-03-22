#!/bin/bash

# DATA_PATH="../PolypGen2021_MultiCenterData_v3/PolypGen2021_MultiCenterData_v3"
DATA_PATH="/kaggle/input/datasets/poliesteeee/iot-2d/PolypGen2021_MultiCenterData_v3/PolypGen2021_MultiCenterData_v3"
DATASET_CLASS="PolypGenFLDataset"
ROUNDS=1
LOCAL_EPOCHS=50
BATCH_SIZE=64
LR=2e-5
PROX_MU=1e-2
MODELS=("Unet" "UnetPlusPlus" "DeepLabV3" "FPN" "PAN" "Segformer")

for MODEL in "${MODELS[@]}"; do
  echo "Running METHOD=FedProx, MODEL=${MODEL}"

  python train_fedprox.py \
    --dataset_class "${DATASET_CLASS}" \
    --data_path "${DATA_PATH}" \
    --csv_path "polypgen_split.csv" \
    --model_name "${MODEL}" \
    --rounds "${ROUNDS}" \
    --local_epochs "${LOCAL_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --prox_mu "${PROX_MU}" \
    --save_dir "weights_fedprox" \
    --save_name "FedProx_${MODEL}_${DATASET_CLASS}.pth" \
    --log_dir "logs_fedprox" \
    --use_amp
done
