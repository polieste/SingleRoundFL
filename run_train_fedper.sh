#!/bin/bash

DATA_PATH="../PolypGen2021_MultiCenterData_v3/PolypGen2021_MultiCenterData_v3"
DATASET_CLASS="PolypGenFLDataset"
ROUNDS=20
LOCAL_EPOCHS=5
BATCH_SIZE=64
LR=2e-5
PERSONALIZED_PREFIXES="decoder.,segmentation_head.,classification_head."
MODELS=("Unet" "UnetPlusPlus" "DeepLabV3" "FPN" "PAN" "Segformer")

for MODEL in "${MODELS[@]}"; do
  echo "Running METHOD=FedPer, MODEL=${MODEL}"

  python train_fedper.py \
    --dataset_class "${DATASET_CLASS}" \
    --data_path "${DATA_PATH}" \
    --csv_path "polypgen_split.csv" \
    --model_name "${MODEL}" \
    --rounds "${ROUNDS}" \
    --local_epochs "${LOCAL_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --personalized_prefixes "${PERSONALIZED_PREFIXES}" \
    --save_dir "weights_fedper" \
    --save_name "FedPer_${MODEL}_${DATASET_CLASS}.pth" \
    --log_dir "logs_fedper" \
    --use_amp
done
