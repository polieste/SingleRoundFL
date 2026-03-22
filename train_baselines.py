import argparse
import importlib
import inspect
import json
import os
import random
from contextlib import nullcontext

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def forward(self, logits, targets):
        return 0.5 * self.bce(logits, targets) + 0.5 * self.dice(logits, targets)


def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation model")

    # dataset
    parser.add_argument("--dataset_class", type=str, default="PolypGenFLDataset")
    parser.add_argument("--data_path", type=str, default="/home/khoi.ho/ML709/PolypGen2021_MultiCenterData_v3")
    parser.add_argument("--csv_path", type=str, default="polypgen_split.csv")
    parser.add_argument("--center", type=str, default="all", choices=["all", "1", "2", "3", "4", "5", "6"])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="Unet",
        choices=["Unet", "UnetPlusPlus", "DeepLabV3", "FPN", "PAN", "Segformer"],
    )

    # train
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_amp", action="store_true")

    # save
    parser.add_argument("--save_dir", type=str, default="weights2")
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="logs")

    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_image_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def build_model(model_name):
    model = smp.create_model(
        model_name,
    )
    return model


def build_dataset(args, train = True):
    module = importlib.import_module('data')
    dataset_class = getattr(module, args.dataset_class)

    image_transform = build_image_transform(args.image_size)

    base_kwargs = {
        "data_path": args.data_path,
        "csv_path": args.csv_path,
        "center": args.center,
        "transform": image_transform,
        "split": "train" if train else "test",
    }
    dataset = dataset_class(**base_kwargs)
    return dataset


def get_autocast_context(device, use_amp):
    if not use_amp or device.type != "cuda":
        return nullcontext()

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda")

    return torch.cuda.amp.autocast()


def get_grad_scaler(device, use_amp):
    if not use_amp or device.type != "cuda":
        return None

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda")
        except TypeError:
            return torch.amp.GradScaler()

    return torch.cuda.amp.GradScaler()


@torch.no_grad()
def compute_dice(logits, masks):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    inter = (preds * masks).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    dice = (2 * inter + 1e-7) / (denom + 1e-7)
    return dice.mean().item()


def train_one_epoch(model, train_loader, test_loader, criterion, optimizer, device, scaler=None, use_amp=False):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, masks in pbar:
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

        dice = compute_dice(logits.detach(), masks.detach())

        total_loss += loss.item()
        total_dice += dice
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{dice:.4f}"
        })
    
    model.eval()
    total_val_dice = 0.0
    num_val_batches = 0

    pbar = tqdm(test_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).float()

            with get_autocast_context(device, use_amp):
                logits = model(images)

            dice = compute_dice(logits, masks)
            total_val_dice += dice
            num_val_batches += 1

    model.train()

    return {
        "loss": total_loss / max(num_batches, 1),
        "dice": total_dice / max(num_batches, 1),
        "val_dice": total_val_dice / max(num_val_batches, 1),
    }

def main():
    args = parse_args()
    seed_everything(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device(args.device)

    train_dataset = build_dataset(args, train=True)
    test_dataset = build_dataset(args, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(args.model_name).to(device)
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = get_grad_scaler(device, args.use_amp)

    if args.save_name.strip():
        save_name = args.save_name
    else:
        save_name = f"{args.model_name}_{args.dataset_class}_C{args.center}.pth"

    save_path = os.path.join(args.save_dir, save_name)

    print(f"Device         : {device}")
    print(f"Dataset class  : {args.dataset_class}")
    print(f"Dataset size   : {len(train_dataset)} train / {len(test_dataset)} test")
    print(f"Model          : {args.model_name}")
    print(f"Save path      : {save_path}")
    print("-" * 70)

    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=args.use_amp,
        )

        if train_metrics["val_dice"] > best_dice:
            best_dice = train_metrics["val_dice"]
            torch.save(model.state_dict(), save_path)

            print(
                f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"train_dice={train_metrics['dice']:.4f} | "
                f"val_dice={train_metrics['val_dice']:.4f} | "
                f"saved: {save_path}" 
            )
            with open(os.path.join(args.log_dir, save_name.replace(".pth", ".txt")), "a") as f:
                f.write(
                    f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
                    f"train_loss={train_metrics['loss']:.4f} | "
                    f"train_dice={train_metrics['dice']:.4f} | "
                    f"val_dice={train_metrics['val_dice']:.4f} | "
                    f"saved: {save_path}\n"
                )
        else:
            print(
                f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"train_dice={train_metrics['dice']:.4f} | "
                f"val_dice={train_metrics['val_dice']:.4f}"
            )
        

    print("Training finished.")


if __name__ == "__main__":
    main()