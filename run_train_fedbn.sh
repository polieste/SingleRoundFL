#!/bin/bash

DATA_PATH="../PolypGen2021_MultiCenterData_v3/PolypGen2021_MultiCenterData_v3"
DATASET_CLASS="PolypGenFLDataset"
ROUNDS=20
LOCAL_EPOCHS=5
BATCH_SIZE=64
LR=2e-5
MODELS=("Unet" "UnetPlusPlus" "DeepLabV3" "FPN" "PAN" "Segformer")

for MODEL in "${MODELS[@]}"; do
  echo "Running METHOD=FedBN, MODEL=${MODEL}"

  python train_fedbn.py \
    --dataset_class "${DATASET_CLASS}" \
    --data_path "${DATA_PATH}" \
    --csv_path "polypgen_split.csv" \
    --model_name "${MODEL}" \
    --rounds "${ROUNDS}" \
    --local_epochs "${LOCAL_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --save_dir "weights_fedbn" \
    --save_name "FedBN_${MODEL}_${DATASET_CLASS}.pth" \
    --log_dir "logs_fedbn" \
    --use_amp
done
