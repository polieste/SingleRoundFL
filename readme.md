
# SingleRoundFL Baseline

This repository contains a baseline setup for medical image segmentation on the PolypGen dataset.
The current baseline supports:

- local-only training per center
- centralized training on all training centers
- post-hoc checkpoint aggregation
- evaluation with Dice and IoU

## Dataset

Download the PolypGen dataset from:

- https://www.synapse.org/Synapse:syn45200214

Expected dataset layout used by the commands below:

```text
IoT/
├── PolypGen2021_MultiCenterData_v3/
│   └── PolypGen2021_MultiCenterData_v3/
└── SingleRoundFL/
```

The split file used by the code is:

- `polypgen_split.csv`

In the current setup:

- centers `1..5` are used for training and in-domain testing
- center `6` is used as a held-out test center

## Environment Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Baseline Workflow

Recommended baseline order:

1. Train a centralized U-Net model with `center=all`
2. Train local-only U-Net models for centers `1..5`
3. Evaluate all checkpoints on the test split
4. Evaluate aggregated checkpoints with `eval_aggregate.py`
5. Compare Dice and IoU across methods

## Training

Run the following commands from the `SingleRoundFL` folder.

### Centralized baseline

```bash
python train_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center all --model_name Unet --epochs 50 --batch_size 8 --lr 1e-4 --save_dir weights_baseline --save_name Unet_PolypGenFLDataset_Call.pth --use_amp
```

### Local-only baselines

```bash
python train_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 1 --model_name Unet --epochs 50 --batch_size 8 --lr 1e-4 --save_dir weights_baseline --save_name Unet_PolypGenFLDataset_C1.pth --use_amp
python train_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 2 --model_name Unet --epochs 50 --batch_size 8 --lr 1e-4 --save_dir weights_baseline --save_name Unet_PolypGenFLDataset_C2.pth --use_amp
python train_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 3 --model_name Unet --epochs 50 --batch_size 8 --lr 1e-4 --save_dir weights_baseline --save_name Unet_PolypGenFLDataset_C3.pth --use_amp
python train_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 4 --model_name Unet --epochs 50 --batch_size 8 --lr 1e-4 --save_dir weights_baseline --save_name Unet_PolypGenFLDataset_C4.pth --use_amp
python train_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 5 --model_name Unet --epochs 50 --batch_size 8 --lr 1e-4 --save_dir weights_baseline --save_name Unet_PolypGenFLDataset_C5.pth --use_amp
```

For a quick sanity check before full training, reduce `--epochs` to `2`.

## Evaluation

### Evaluate the centralized model

```bash
python eval_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 1 --model_name Unet --weight_path weights_baseline\Unet_PolypGenFLDataset_Call.pth
python eval_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 2 --model_name Unet --weight_path weights_baseline\Unet_PolypGenFLDataset_Call.pth
python eval_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 3 --model_name Unet --weight_path weights_baseline\Unet_PolypGenFLDataset_Call.pth
python eval_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 4 --model_name Unet --weight_path weights_baseline\Unet_PolypGenFLDataset_Call.pth
python eval_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 5 --model_name Unet --weight_path weights_baseline\Unet_PolypGenFLDataset_Call.pth
python eval_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 6 --model_name Unet --weight_path weights_baseline\Unet_PolypGenFLDataset_Call.pth
```

### Evaluate local-only models

```bash
python eval_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 1 --model_name Unet --weight_path weights_baseline\Unet_PolypGenFLDataset_C1.pth
python eval_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 2 --model_name Unet --weight_path weights_baseline\Unet_PolypGenFLDataset_C2.pth
python eval_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 3 --model_name Unet --weight_path weights_baseline\Unet_PolypGenFLDataset_C3.pth
python eval_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 4 --model_name Unet --weight_path weights_baseline\Unet_PolypGenFLDataset_C4.pth
python eval_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 5 --model_name Unet --weight_path weights_baseline\Unet_PolypGenFLDataset_C5.pth
```

### Evaluate aggregated checkpoints

```bash
python eval_aggregate.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 6 --model_name Unet --weight_folder_path weights_baseline --agg_mode average
python eval_aggregate.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 6 --model_name Unet --weight_folder_path weights_baseline --agg_mode fedavg
```

## Current Scope

This repository currently provides a baseline experiment setup.
It does not yet implement a full multi-round federated training pipeline.

The intended interpretation of the current scripts is:

- `train_baselines.py`: train one model
- `eval_baselines.py`: evaluate one checkpoint on the test split
- `eval_aggregate.py`: aggregate multiple local checkpoints and evaluate the aggregated model

## Suggested Reporting

For the first baseline, report:

- local-only performance per center
- centralized performance per center
- aggregated performance
- mean Dice across centers `1..5`
- worst-client Dice
- held-out performance on center `6`
