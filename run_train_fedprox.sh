#!/bin/bash

DATA_PATH="../PolypGen2021_MultiCenterData_v3/PolypGen2021_MultiCenterData_v3"
DATASET_CLASS="PolypGenFLDataset"
MODEL_NAME="Unet"
ROUNDS=20
LOCAL_EPOCHS=1
BATCH_SIZE=8
LR=1e-4
PROX_MU=1e-2

python train_fedprox.py \
  --dataset_class "${DATASET_CLASS}" \
  --data_path "${DATA_PATH}" \
  --csv_path "polypgen_split.csv" \
  --model_name "${MODEL_NAME}" \
  --rounds "${ROUNDS}" \
  --local_epochs "${LOCAL_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --prox_mu "${PROX_MU}" \
  --save_dir "weights_fedprox" \
  --save_name "FedProx_${MODEL_NAME}_${DATASET_CLASS}.pth" \
  --log_dir "logs_fedprox" \
  --use_amp
