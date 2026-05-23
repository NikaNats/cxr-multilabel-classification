from __future__ import annotations

import os
import numpy as np
import torch

EMBEDDING_DIR: str = "/workspace/Extraction/cxr_embeddings_10percent"


def analyze_split_cardinality(
    file_path: str | os.PathLike, split_name: str = "Train"
) -> None:
    """Loads a serialized embedding dataset and logs its clinical label cardinality distribution."""
    if not os.path.exists(file_path):
        print(f"[!] File not found: {file_path}")
        return

    data = torch.load(file_path, map_location="cpu", weights_only=True)
    labels = data["labels"].numpy().astype(np.int32)  # Shape: (N, 14)
    
    total_samples = labels.shape[0]
    
    cardinalities = np.sum(labels, axis=1)
    unique_card, counts = np.unique(cardinalities, return_counts=True)
    
    border = "=" * 55
    divider = "-" * 55
    print(f"\n{border}")
    print(f"  GROUND TRUTH CARDINALITY DISTRIBUTION: {split_name.upper()} SPLIT")
    print(f"  Total Samples (N) = {total_samples:,}")
    print(border)
    print(f"{'Pathologies':<15} | {'Samples Count':<15} | {'Percentage (%)':<15}")
    print(divider)
    
    for card, cnt in zip(unique_card, counts):
        percentage = (cnt / total_samples) * 100
        print(f"{int(card):<15} | {cnt:<15,} | {percentage:.2f}%")
        
    print(divider)
    print(f"  Empirical Mean Cardinality: {cardinalities.mean():.4f}")
    print(f"{border}\n")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        path = os.path.join(EMBEDDING_DIR, f"{split}_embeddings.pt")
        analyze_split_cardinality(path, split_name=split)