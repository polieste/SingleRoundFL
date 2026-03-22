import argparse
import json
import os
from copy import deepcopy

import torch
from torch.utils.data import DataLoader

from train_baselines import (
    CombinedLoss,
    build_dataset,
    build_model,
    compute_dice,
    get_autocast_context,
    get_grad_scaler,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a FedAvg segmentation model")

    # dataset
    parser.add_argument("--dataset_class", type=str, default="PolypGenFLDataset")
    parser.add_argument("--data_path", type=str, default="/home/khoi.ho/ML709/PolypGen2021_MultiCenterData_v3")
    parser.add_argument("--csv_path", type=str, default="polypgen_split.csv")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_centers", type=str, default="1,2,3,4,5")
    parser.add_argument("--eval_centers", type=str, default="1,2,3,4,5,6")

    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="Unet",
        choices=["Unet", "UnetPlusPlus", "DeepLabV3", "FPN", "PAN", "Segformer"],
    )

    # train
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_amp", action="store_true")

    # save
    parser.add_argument("--save_dir", type=str, default="weights_fedavg")
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="logs_fedavg")

    return parser.parse_args()


def parse_centers(centers):
    return [center.strip() for center in centers.split(",") if center.strip()]


def build_dataset_for_center(args, center, train=True):
    args.center = center
    return build_dataset(args, train=train)


def build_loader(dataset, batch_size, num_workers, device, shuffle, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=drop_last,
    )


def train_local_epochs(model, train_loader, criterion, optimizer, device, local_epochs, scaler=None, use_amp=False):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0

    for _ in range(local_epochs):
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with get_autocast_context(device, use_amp):
                logits = model(images)
                loss = criterion(logits, masks)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_dice += compute_dice(logits.detach(), masks.detach())
            num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "dice": total_dice / max(num_batches, 1),
    }


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, use_amp=False):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()

        with get_autocast_context(device, use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        total_loss += loss.item()
        total_dice += compute_dice(logits, masks)
        num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "dice": total_dice / max(num_batches, 1),
    }


def average_state_dicts(state_dicts, client_weights):
    averaged_state = deepcopy(state_dicts[0])
    normalized_weights = [weight / sum(client_weights) for weight in client_weights]

    for key in averaged_state.keys():
        if averaged_state[key].dtype in [torch.int32, torch.int64, torch.uint8, torch.bool]:
            averaged_state[key] = state_dicts[0][key].clone()
            continue

        averaged_state[key] = torch.zeros_like(averaged_state[key], dtype=torch.float32)
        for state_dict, weight in zip(state_dicts, normalized_weights):
            averaged_state[key] += state_dict[key].float() * weight
        averaged_state[key] = averaged_state[key].to(state_dicts[0][key].dtype)

    return averaged_state


def append_log(log_path, record):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def format_round_summary(round_idx, rounds, train_summary, eval_summary):
    return (
        f"Round [{round_idx:03d}/{rounds:03d}] | "
        f"train_loss={train_summary['loss']:.4f} | "
        f"train_dice={train_summary['dice']:.4f} | "
        f"eval_loss={eval_summary['loss']:.4f} | "
        f"eval_dice={eval_summary['dice']:.4f} | "
        f"worst_client_dice={eval_summary['worst_client_dice']:.4f}"
    )


def main():
    args = parse_args()
    seed_everything(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device(args.device)
    criterion = CombinedLoss()

    train_centers = parse_centers(args.train_centers)
    eval_centers = parse_centers(args.eval_centers)

    client_train_loaders = {}
    client_train_sizes = {}
    eval_loaders = {}

    for center in train_centers:
        train_dataset = build_dataset_for_center(args, center=center, train=True)
        client_train_sizes[center] = len(train_dataset)
        client_train_loaders[center] = build_loader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            shuffle=True,
            drop_last=(len(train_dataset) >= args.batch_size),
        )

    for center in eval_centers:
        test_dataset = build_dataset_for_center(args, center=center, train=False)
        eval_loaders[center] = build_loader(
            dataset=test_dataset,
            batch_size=1,
            num_workers=args.num_workers,
            device=device,
            shuffle=False,
            drop_last=False,
        )

    global_model = build_model(args.model_name).to(device)

    if args.save_name.strip():
        save_name = args.save_name
    else:
        save_name = f"FedAvg_{args.model_name}_{args.dataset_class}.pth"

    save_path = os.path.join(args.save_dir, save_name)
    log_path = os.path.join(args.log_dir, save_name.replace(".pth", ".jsonl"))
    text_log_path = os.path.join(args.log_dir, save_name.replace(".pth", ".txt"))

    print(f"Device        : {device}")
    print(f"Dataset class : {args.dataset_class}")
    print(f"Train centers : {train_centers}")
    print(f"Eval centers  : {eval_centers}")
    print(f"Model         : {args.model_name}")
    print(f"Rounds        : {args.rounds}")
    print(f"Local epochs  : {args.local_epochs}")
    print(f"Save path     : {save_path}")
    print("-" * 70)

    best_eval_dice = -1.0

    for round_idx in range(1, args.rounds + 1):
        global_state = deepcopy(global_model.state_dict())
        local_state_dicts = []
        local_metrics = {}

        for center in train_centers:
            local_model = build_model(args.model_name).to(device)
            local_model.load_state_dict(global_state)

            optimizer = torch.optim.AdamW(local_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scaler = get_grad_scaler(device, args.use_amp)

            metrics = train_local_epochs(
                model=local_model,
                train_loader=client_train_loaders[center],
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                local_epochs=args.local_epochs,
                scaler=scaler,
                use_amp=args.use_amp,
            )

            local_metrics[center] = {
                "train_loss": metrics["loss"],
                "train_dice": metrics["dice"],
                "num_samples": client_train_sizes[center],
            }
            local_state_dicts.append(deepcopy(local_model.state_dict()))

        averaged_state = average_state_dicts(
            state_dicts=local_state_dicts,
            client_weights=[client_train_sizes[center] for center in train_centers],
        )
        global_model.load_state_dict(averaged_state)

        per_client_eval = {}
        eval_dice_values = []
        eval_loss_values = []
        for center in eval_centers:
            metrics = evaluate_model(
                model=global_model,
                loader=eval_loaders[center],
                criterion=criterion,
                device=device,
                use_amp=args.use_amp,
            )
            per_client_eval[center] = {
                "eval_loss": metrics["loss"],
                "eval_dice": metrics["dice"],
            }
            eval_loss_values.append(metrics["loss"])
            eval_dice_values.append(metrics["dice"])

        total_train_samples = sum(client_train_sizes.values())
        train_summary = {
            "loss": sum(local_metrics[c]["train_loss"] * local_metrics[c]["num_samples"] for c in train_centers) / max(total_train_samples, 1),
            "dice": sum(local_metrics[c]["train_dice"] * local_metrics[c]["num_samples"] for c in train_centers) / max(total_train_samples, 1),
        }
        eval_summary = {
            "loss": sum(eval_loss_values) / max(len(eval_loss_values), 1),
            "dice": sum(eval_dice_values) / max(len(eval_dice_values), 1),
            "worst_client_dice": min(eval_dice_values) if eval_dice_values else 0.0,
        }

        round_record = {
            "round": round_idx,
            "train_summary": train_summary,
            "eval_summary": eval_summary,
            "local_metrics": local_metrics,
            "per_client_eval": per_client_eval,
        }
        append_log(log_path, round_record)

        round_message = format_round_summary(round_idx, args.rounds, train_summary, eval_summary)
        print(round_message)
        with open(text_log_path, "a", encoding="utf-8") as f:
            f.write(round_message + "\n")

        if eval_summary["dice"] > best_eval_dice:
            best_eval_dice = eval_summary["dice"]
            torch.save(
                {
                    "model_state_dict": global_model.state_dict(),
                    "round": round_idx,
                    "best_eval_dice": best_eval_dice,
                    "train_centers": train_centers,
                    "eval_centers": eval_centers,
                },
                save_path,
            )
            print(f"Saved best global model to: {save_path}")

    print("FedAvg training finished.")


if __name__ == "__main__":
    main()
