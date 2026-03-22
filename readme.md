# SingleRoundFL Baseline

This repository provides a baseline setup for medical image segmentation on the PolypGen dataset.

Current scope:

- local-only training per center
- centralized training on all training centers
- post-hoc checkpoint aggregation
- evaluation with Dice and IoU

It does not yet implement a full multi-round federated training pipeline.

## Dataset

Download PolypGen from:

- https://www.synapse.org/Synapse:syn45200214

Expected directory layout:

```text
IoT/
├── PolypGen2021_MultiCenterData_v3/
│   └── PolypGen2021_MultiCenterData_v3/
└── SingleRoundFL/
```

The split file used by the code is `polypgen_split.csv`.

Current split design:

- centers `1..5`: training clients and in-domain test centers
- center `6`: held-out test center

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run all commands below from the `SingleRoundFL` directory.

## Training

### Centralized baseline

```bash
python train_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center all --model_name Unet --epochs 50 --batch_size 8 --lr 1e-4 --save_dir weights_baseline --save_name Unet_PolypGenFLDataset_Call.pth --use_amp
```

### Local-only baselines

Run one command per center:

```bash
python train_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 1 --model_name Unet --epochs 50 --batch_size 8 --lr 1e-4 --save_dir weights_baseline --save_name Unet_PolypGenFLDataset_C1.pth --use_amp
python train_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 2 --model_name Unet --epochs 50 --batch_size 8 --lr 1e-4 --save_dir weights_baseline --save_name Unet_PolypGenFLDataset_C2.pth --use_amp
python train_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 3 --model_name Unet --epochs 50 --batch_size 8 --lr 1e-4 --save_dir weights_baseline --save_name Unet_PolypGenFLDataset_C3.pth --use_amp
python train_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 4 --model_name Unet --epochs 50 --batch_size 8 --lr 1e-4 --save_dir weights_baseline --save_name Unet_PolypGenFLDataset_C4.pth --use_amp
python train_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 5 --model_name Unet --epochs 50 --batch_size 8 --lr 1e-4 --save_dir weights_baseline --save_name Unet_PolypGenFLDataset_C5.pth --use_amp
```

For a quick sanity check before full training, reduce `--epochs` to `2`.

## Evaluation

### Centralized model

Evaluate the centralized checkpoint on each test center:

```bash
python eval_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 1 --model_name Unet --weight_path weights_baseline\Unet_PolypGenFLDataset_Call.pth
python eval_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 2 --model_name Unet --weight_path weights_baseline\Unet_PolypGenFLDataset_Call.pth
python eval_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 3 --model_name Unet --weight_path weights_baseline\Unet_PolypGenFLDataset_Call.pth
python eval_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 4 --model_name Unet --weight_path weights_baseline\Unet_PolypGenFLDataset_Call.pth
python eval_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 5 --model_name Unet --weight_path weights_baseline\Unet_PolypGenFLDataset_Call.pth
python eval_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 6 --model_name Unet --weight_path weights_baseline\Unet_PolypGenFLDataset_Call.pth
```

### Local-only models

Evaluate each local checkpoint on its corresponding test center:

```bash
python eval_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 1 --model_name Unet --weight_path weights_baseline\Unet_PolypGenFLDataset_C1.pth
python eval_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 2 --model_name Unet --weight_path weights_baseline\Unet_PolypGenFLDataset_C2.pth
python eval_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 3 --model_name Unet --weight_path weights_baseline\Unet_PolypGenFLDataset_C3.pth
python eval_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 4 --model_name Unet --weight_path weights_baseline\Unet_PolypGenFLDataset_C4.pth
python eval_baselines.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 5 --model_name Unet --weight_path weights_baseline\Unet_PolypGenFLDataset_C5.pth
```

### Aggregated checkpoints

```bash
python eval_aggregate.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 6 --model_name Unet --weight_folder_path weights_baseline --agg_mode average
python eval_aggregate.py --dataset_class PolypGenFLDataset --data_path ..\PolypGen2021_MultiCenterData_v3\PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --center 6 --model_name Unet --weight_folder_path weights_baseline --agg_mode fedavg
```

## What Each Script Does

- `train_baselines.py`: trains a single segmentation model
- `eval_baselines.py`: evaluates one checkpoint on the test split
- `eval_aggregate.py`: aggregates multiple local checkpoints and evaluates the aggregated model

## Suggested Baseline Reporting

For the first report, include:

- local-only performance per center
- centralized performance per center
- aggregated performance
- mean Dice across centers `1..5`
- worst-client Dice across centers `1..5`
- held-out performance on center `6`

## FedAvg

Run FedAvg with:

```bash
bash run_train_fedavg.sh
```

Or run it directly:

```bash
python train_fedavg.py --dataset_class PolypGenFLDataset --data_path ../PolypGen2021_MultiCenterData_v3/PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --model_name Unet --rounds 20 --local_epochs 1 --batch_size 8 --lr 1e-4 --save_dir weights_fedavg --save_name FedAvg_Unet_PolypGenFLDataset.pth --log_dir logs_fedavg --use_amp
```
## FedProx
Run FedProx with:

```bash
bash run_train_fedprox.sh
```

Or run it directly:

```bash
python train_fedprox.py --dataset_class PolypGenFLDataset --data_path ../PolypGen2021_MultiCenterData_v3/PolypGen2021_MultiCenterData_v3 --csv_path polypgen_split.csv --model_name Unet --rounds 20 --local_epochs 1 --batch_size 8 --lr 1e-4 --prox_mu 1e-2 --save_dir weights_fedprox --save_name FedProx_Unet_PolypGenFLDataset.pth --log_dir logs_fedprox --use_amp
```
