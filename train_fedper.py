import argparse
import os
from copy import deepcopy

import torch

from train_baselines import CombinedLoss, build_model, get_grad_scaler, seed_everything
from train_fedavg import (
    append_log,
    build_dataset_for_center,
    build_loader,
    evaluate_model,
    parse_centers,
    train_local_epochs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a FedPer segmentation model")

    parser.add_argument("--dataset_class", type=str, default="PolypGenFLDataset")
    parser.add_argument("--data_path", type=str, default="/home/khoi.ho/ML709/PolypGen2021_MultiCenterData_v3")
    parser.add_argument("--csv_path", type=str, default="polypgen_split.csv")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_centers", type=str, default="1,2,3,4,5")
    parser.add_argument("--eval_centers", type=str, default="1,2,3,4,5")

    parser.add_argument(
        "--model_name",
        type=str,
        default="Unet",
        choices=["Unet", "UnetPlusPlus", "DeepLabV3", "FPN", "PAN", "Segformer"],
    )
    parser.add_argument(
        "--personalized_prefixes",
        type=str,
        default="decoder.,segmentation_head.,classification_head.",
        help="Comma-separated state_dict prefixes kept local per client",
    )

    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_amp", action="store_true")

    parser.add_argument("--save_dir", type=str, default="weights_fedper")
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="logs_fedper")

    return parser.parse_args()


def parse_prefixes(prefixes):
    return [prefix.strip() for prefix in prefixes.split(",") if prefix.strip()]


def get_personalized_state_keys(model, prefixes):
    personalized_keys = set()
    for key in model.state_dict().keys():
        if any(key.startswith(prefix) for prefix in prefixes):
            personalized_keys.add(key)
    return personalized_keys


def merge_global_and_local_state(global_state, local_state, local_keys):
    merged_state = deepcopy(global_state)
    for key in local_keys:
        if key in local_state:
            merged_state[key] = local_state[key].clone()
    return merged_state


def average_state_dicts_excluding(state_dicts, client_weights, excluded_keys):
    averaged_state = deepcopy(state_dicts[0])
    normalized_weights = [weight / sum(client_weights) for weight in client_weights]

    for key in averaged_state.keys():
        if key in excluded_keys:
            continue

        if averaged_state[key].dtype in [torch.int32, torch.int64, torch.uint8, torch.bool]:
            averaged_state[key] = state_dicts[0][key].clone()
            continue

        averaged_state[key] = torch.zeros_like(averaged_state[key], dtype=torch.float32)
        for state_dict, weight in zip(state_dicts, normalized_weights):
            averaged_state[key] += state_dict[key].float() * weight
        averaged_state[key] = averaged_state[key].to(state_dicts[0][key].dtype)

    return averaged_state


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
    personalized_prefixes = parse_prefixes(args.personalized_prefixes)

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
    personalized_state_keys = get_personalized_state_keys(global_model, personalized_prefixes)
    client_states = {
        center: deepcopy(global_model.state_dict())
        for center in train_centers
    }

    if args.save_name.strip():
        save_name = args.save_name
    else:
        save_name = f"FedPer_{args.model_name}_{args.dataset_class}.pth"

    save_path = os.path.join(args.save_dir, save_name)
    log_path = os.path.join(args.log_dir, save_name.replace(".pth", ".jsonl"))
    text_log_path = os.path.join(args.log_dir, save_name.replace(".pth", ".txt"))

    print(f"Device              : {device}")
    print(f"Dataset class       : {args.dataset_class}")
    print(f"Train centers       : {train_centers}")
    print(f"Eval centers        : {eval_centers}")
    print(f"Model               : {args.model_name}")
    print(f"Rounds              : {args.rounds}")
    print(f"Local epochs        : {args.local_epochs}")
    print(f"Personalized prefix : {personalized_prefixes}")
    print(f"Personalized keys   : {len(personalized_state_keys)}")
    print(f"Save path           : {save_path}")
    print("-" * 70)

    best_eval_dice = -1.0

    for round_idx in range(1, args.rounds + 1):
        global_state = deepcopy(global_model.state_dict())
        local_state_dicts = []
        local_metrics = {}

        for center in train_centers:
            local_model = build_model(args.model_name).to(device)
            local_model.load_state_dict(
                merge_global_and_local_state(global_state, client_states[center], personalized_state_keys)
            )

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

            client_states[center] = deepcopy(local_model.state_dict())
            local_state_dicts.append(client_states[center])
            local_metrics[center] = {
                "train_loss": metrics["loss"],
                "train_dice": metrics["dice"],
                "num_samples": client_train_sizes[center],
            }

        averaged_state = average_state_dicts_excluding(
            state_dicts=local_state_dicts,
            client_weights=[client_train_sizes[center] for center in train_centers],
            excluded_keys=personalized_state_keys,
        )
        global_model.load_state_dict(averaged_state, strict=False)

        per_client_eval = {}
        eval_dice_values = []
        eval_loss_values = []
        for center in eval_centers:
            if center in client_states:
                eval_model = build_model(args.model_name).to(device)
                eval_model.load_state_dict(
                    merge_global_and_local_state(global_model.state_dict(), client_states[center], personalized_state_keys)
                )
            else:
                eval_model = build_model(args.model_name).to(device)
                eval_model.load_state_dict(global_model.state_dict())

            metrics = evaluate_model(
                model=eval_model,
                loader=eval_loaders[center],
                criterion=criterion,
                device=device,
                use_amp=args.use_amp,
            )
            per_client_eval[center] = {
                "eval_loss": metrics["loss"],
                "eval_dice": metrics["dice"],
                "personalized": center in client_states,
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
                    "client_states": client_states,
                    "round": round_idx,
                    "best_eval_dice": best_eval_dice,
                    "train_centers": train_centers,
                    "eval_centers": eval_centers,
                    "personalized_prefixes": personalized_prefixes,
                    "personalized_state_keys": sorted(personalized_state_keys),
                },
                save_path,
            )
            print(f"Saved best personalized global state to: {save_path}")

    print("FedPer training finished.")


if __name__ == "__main__":
    main()
