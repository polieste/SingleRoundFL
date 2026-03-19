import argparse
import importlib
import inspect
import json
import os

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def forward(self, logits, targets):
        return 0.5 * self.bce(logits, targets) + 0.5 * self.dice(logits, targets)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")

    parser.add_argument("--dataset_class", type=str, default="PolypGenFLDataset")
    parser.add_argument("--data_path", type=str, default="/home/khoi.ho/ML709/PolypGen2021_MultiCenterData_v3")
    parser.add_argument("--center", type=str, default="6")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--model_name",
        type=str,
        default="Unet",
        choices=["Unet", "UnetPlusPlus", "DeepLabV3", "FPN", "PAN", "Segformer"],
    )
    
    parser.add_argument("--weight_folder_path", type=str, default="/home/khoi.ho/ML709/SingleRoundFL/weights")
    parser.add_argument("--agg_mode", type=str, default="average", choices=["average", "fedavg"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_pred_dir", type=str, default="")
    parser.add_argument("--threshold", type=float, default=0.5)

    return parser.parse_args()


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


def build_dataset(args):
    module = importlib.import_module('data')
    dataset_class = getattr(module, args.dataset_class)

    image_transform = build_image_transform(args.image_size)

    base_kwargs = {
        "data_path": args.data_path,
        "center": args.center,
        "transform": image_transform,
    }

    dataset = dataset_class(**base_kwargs)
    return dataset


@torch.no_grad()
def compute_metrics(logits, masks, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    inter = (preds * masks).sum(dim=(1, 2, 3))
    pred_sum = preds.sum(dim=(1, 2, 3))
    mask_sum = masks.sum(dim=(1, 2, 3))
    union = pred_sum + mask_sum - inter

    dice = ((2 * inter + 1e-7) / (pred_sum + mask_sum + 1e-7)).mean().item()
    iou = ((inter + 1e-7) / (union + 1e-7)).mean().item()
    return dice, iou, preds

def aggregate_checkpoints(weight_paths, client_weights, model_name, device, agg_mode="average"):
    client_weights = torch.tensor(client_weights, dtype=torch.float32, device=device)
    client_weights = client_weights / client_weights.sum()

    global_model = build_model(model_name).to(device)
    global_dict = global_model.state_dict()

    local_state_dicts = []
    for weight_path in weight_paths:
        ckpt = torch.load(weight_path, weights_only=True, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            local_state_dicts.append(ckpt["model_state_dict"])
        else:
            local_state_dicts.append(ckpt)

    for key in global_dict.keys():
        if global_dict[key].dtype in [torch.int32, torch.int64, torch.uint8, torch.bool]:
            global_dict[key] = local_state_dicts[0][key].clone()
        else:
            global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
            for i in range(len(local_state_dicts)):
                global_dict[key] += local_state_dicts[i][key].float() * client_weights[i]
            global_dict[key] = global_dict[key].to(local_state_dicts[0][key].dtype)

    global_model.load_state_dict(global_dict)
    return global_model

def main():
    args = parse_args()
    device = torch.device(args.device)

    dataset = build_dataset(args)
    if len(dataset) == 0:
        raise ValueError("Dataset rỗng.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    
    if args.dataset_class == 'PolypGenFLDataset':
        if args.agg_mode != "average":
            weight_count = np.array([256, 301, 457, 227, 208])
        elif args.agg_mode == "average":
            weight_count = np.ones(5)

    weight_paths = []
    for filename in os.listdir(args.weight_folder_path):
        if f'{args.model_name}_' in filename and 'Call' not in filename and filename.endswith('.pth'):
            weight_path = os.path.join(args.weight_folder_path, filename)
            print(f"Loading weights from: {weight_path}")
            weight_paths.append(weight_path)
              
    model = aggregate_checkpoints(weight_paths, weight_count, args.model_name, device, agg_mode=args.agg_mode)
    model.eval()
    criterion = CombinedLoss()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0
    sample_idx = 0

    pbar = tqdm(loader, desc="Evaluating")
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).float()

            logits = model(images)
            loss = criterion(logits, masks)
            dice, iou, preds = compute_metrics(logits, masks, threshold=args.threshold)

            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{dice:.4f}",
                "iou": f"{iou:.4f}",
            })

    print("-" * 60)
    print(f"Weight path : {args.weight_folder_path}")
    print(f"Dataset     : {args.dataset_class}")
    print(f"Center      : {args.center}")
    print(f"Samples     : {len(dataset)}")
    print(f"Loss        : {total_loss / max(num_batches, 1):.4f}")
    print(f"Dice        : {total_dice / max(num_batches, 1):.4f}")
    print(f"IoU         : {total_iou / max(num_batches, 1):.4f}")
    print("-" * 60)


if __name__ == "__main__":
    main()