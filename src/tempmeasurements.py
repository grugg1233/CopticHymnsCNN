#!/usr/bin/env python
# coding: utf-8
"""
Neural Collapse NC1 measurement for HymnCNN checkpoints.

Usage:
    python measurements.py --checkpoint output/checkpoints/checkpoint1.pt \
                           --batch-size 32

This will:
- load the dataset (all snippets) using loadTheData()
- load the HymnCNN model from the given checkpoint
- extract penultimate-layer features (fc1 output, pre-ReLU)
- compute NC1 = trace(S_w) / trace(S_b)
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader

from config import Config
from model import HymnCNN
from data import loadTheData


def get_features_and_labels(model, dataloader, device):
    """
    Run the model over all batches, capturing penultimate features and labels.
    We hook on fc1, so we get a (B, 64) tensor per forward call.
    """
    model.eval()
    features_list = []
    labels_list = []

    def hook_fn(module, input, output):
        # output: (B, 64)
        features_list.append(output.detach().cpu())

    handle = model.fc1.register_forward_hook(hook_fn)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels_list.append(labels.clone().cpu())
            _ = model(inputs)  # triggers hook, fills features_list

    handle.remove()

    features = torch.cat(features_list, dim=0)  # (N, D)
    labels = torch.cat(labels_list, dim=0)      # (N,)
    return features, labels


def compute_nc1(features: torch.Tensor, labels: torch.Tensor):
    """
    Compute NC1 = trace(S_w) / trace(S_b).

    S_w = within-class scatter
    S_b = between-class scatter (class means vs global mean)
    """
    # ensure float32
    features = features.float()
    labels = labels.long()

    unique_labels = labels.unique(sorted=True)
    num_classes = unique_labels.numel()
    N, D = features.shape

    print(f"Total samples: {N}, feature dim: {D}, num_classes: {num_classes}")

    class_means = []
    class_counts = []

    Sw = 0.0

    for c in unique_labels:
        mask = (labels == c)
        Xc = features[mask]  # (n_c, D)
        n_c = Xc.shape[0]

        if n_c == 0:
            continue

        mu_c = Xc.mean(dim=0)
        class_means.append(mu_c)
        class_counts.append(n_c)

        # within-class scatter (sum of squared distances to class mean)
        Sw += ((Xc - mu_c) ** 2).sum().item()

    class_means = torch.stack(class_means, dim=0)          # (C, D)
    class_counts = torch.tensor(class_counts, dtype=torch.float32)  # (C,)

    # global mean (weighted by class counts)
    total_count = class_counts.sum()
    global_mean = (class_means * class_counts.unsqueeze(1)).sum(dim=0) / total_count

    # between-class scatter
    Sb = 0.0
    for k in range(class_means.shape[0]):
        diff = class_means[k] - global_mean
        Sb += class_counts[k].item() * (diff * diff).sum().item()

    nc1 = Sw / Sb if Sb > 0 else float("inf")

    return {
        "Sw": Sw,
        "Sb": Sb,
        "NC1": nc1,
        "num_classes": num_classes,
        "total_samples": int(N),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute Neural Collapse NC1 for a HymnCNN checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint .pt file (with model_state_dict).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=Config.NUM_WORKERS,
        help="Number of DataLoader workers.",
    )
    args = parser.parse_args()

    torch.manual_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")
    dataset = loadTheData()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Dataset size (snippets): {len(dataset)}")

    print("Building model...")
    model = HymnCNN(n_classes=Config.N_CLASSES).to(device)

    print(f"Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # in case you saved directly model.state_dict()
        model.load_state_dict(ckpt)
    model.eval()

    start = time.time()
    print("Extracting features...")
    features, labels = get_features_and_labels(model, dataloader, device)
    print(f"Got features: {features.shape}, labels: {labels.shape}")

    print("Computing NC1...")
    stats = compute_nc1(features, labels)

    elapsed = time.time() - start
    print("\n=== Neural Collapse NC1 Stats ===")
    print(f"Total samples      : {stats['total_samples']}")
    print(f"Num classes        : {stats['num_classes']}")
    print(f"Within-class Sw    : {stats['Sw']:.4f}")
    print(f"Between-class Sb   : {stats['Sb']:.4f}")
    print(f"NC1 = Sw / Sb      : {stats['NC1']:.6f}")
    print(f"Elapsed time       : {elapsed/60:.2f} min")


if __name__ == "__main__":
    main()
