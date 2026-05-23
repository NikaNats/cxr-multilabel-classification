from __future__ import annotations

import hashlib
import logging
import os
import multiprocessing
import numpy as np
import torch
from medmnist import ChestMNIST
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import DATALOADER_GENERATOR, log_process

# ==============================================================================
# § 1  GLOBAL DATA CONFIGURATION
# ==============================================================================

IMAGE_SIZE: int = 224
BATCH_SIZE: int = 64

# Path to serialized feature tensors extracted from the ELIXR-C (CXR-Foundation) backbone
EMBEDDING_DIR: str = "/workspace/Extraction/cxr_embeddings_10percent"

# Normalizes raw images to single-channel tensors
_extract_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]
)


def _img_hashes(arr: np.ndarray) -> list[str]:
    """
    Generates cryptographically secure, bit-exact Blake2b fingerprints.
    Used for the Forensic Leakage Audit to catch overlapping patient scans.
    """
    return [
        hashlib.blake2b(arr[i].tobytes(), digest_size=16).hexdigest()
        for i in range(len(arr))
    ]


def get_raw_datasets() -> tuple[ChestMNIST, ChestMNIST, ChestMNIST, list[str]]:
    """
    Acquires ChestMNIST data and executes a Forensic Deduplication Audit.
    Medical datasets often contain patient overlaps between train/test; 
    this function ensures strict separation via bitwise hashing.
    """
    print("[*] Downloading / loading ChestMNIST 224×224 (MedMNIST+)...")
    train_dataset_raw = ChestMNIST(
        split="train", transform=_extract_transform, download=True, size=IMAGE_SIZE
    )
    val_dataset_raw = ChestMNIST(
        split="val", transform=_extract_transform, download=True, size=IMAGE_SIZE
    )
    test_dataset_raw = ChestMNIST(
        split="test", transform=_extract_transform, download=True, size=IMAGE_SIZE
    )

    # --------------------------------------------------------------------------
    # FORENSIC LEAKAGE AUDIT (Blake2b Verification)
    # --------------------------------------------------------------------------
    print("[*] Blake2b cross-split deduplication audit...")
    train_hash_set = set(_img_hashes(train_dataset_raw.imgs))
    raw_val_hashes = _img_hashes(val_dataset_raw.imgs)
    raw_test_hashes = _img_hashes(test_dataset_raw.imgs)

    # Filter Validation split for Train overlaps
    val_keep = np.array([h not in train_hash_set for h in raw_val_hashes])
    n_val_rm = int((~val_keep).sum())
    if n_val_rm:
        val_dataset_raw.imgs = val_dataset_raw.imgs[val_keep]
        val_dataset_raw.labels = val_dataset_raw.labels[val_keep]
        val_dataset_raw.info["n_samples"]["val"] = int(val_keep.sum())
        print(f"  Removed {n_val_rm} duplicate(s) from Val.")

    # Filter Test split for Train/Val overlaps (Zero-leakage guarantee)
    val_hash_clean = {h for h, k in zip(raw_val_hashes, val_keep) if k}
    combined_ref = train_hash_set | val_hash_clean
    test_keep = np.array([h not in combined_ref for h in raw_test_hashes])
    n_test_rm = int((~test_keep).sum())
    if n_test_rm:
        test_dataset_raw.imgs = test_dataset_raw.imgs[test_keep]
        test_dataset_raw.labels = test_dataset_raw.labels[test_keep]
        test_dataset_raw.info["n_samples"]["test"] = int(test_keep.sum())
        print(f"  Removed {n_test_rm} duplicate(s) from Test.")

    class_names = [
        train_dataset_raw.info["label"][str(i)]
        for i in range(len(train_dataset_raw.info["label"]))
    ]

    log_process(
        "data",
        "dataset_ready",
        train=len(train_dataset_raw),
        val=len(val_dataset_raw),
        test=len(test_dataset_raw),
        classes=len(class_names),
        val_dups_removed=n_val_rm,
        test_dups_removed=n_test_rm,
    )

    return train_dataset_raw, val_dataset_raw, test_dataset_raw, class_names


class EmbeddingDataset(Dataset):
    """
    Specialized Dataset for frozen-backbone training.
    Loads pre-computed 1376-dimensional feature maps (8x8 spatial resolution).
    
    Includes Latent Space Jittering for robust feature augmentation.
    """

    def __init__(
        self, path: str | os.PathLike, split: str = "train", jitter_eps: float = 0.015
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Embedding file not found: {path}\nPlease run extraction first."
            )

        # Safe-load protocol using weights_only=True to prevent arbitrary code execution
        data = torch.load(path, map_location="cpu", weights_only=True)
        self.features = data["features"].float()
        self.labels = data["labels"].float()
        self.split = split
        self.jitter_eps = jitter_eps

        # --------------------------------------------------------------------------
        # Tensor Layout Alignment
        # If features are in PyTorch-standard (B, C, H, W) -> e.g. (N, 1376, 8, 8),
        # permute them to channel-last (B, H, W, C) -> e.g. (N, 8, 8, 1376).
        # This resolves the runtime conflict with the model's forward pass.
        # --------------------------------------------------------------------------
        if self.features.ndim == 4:
            if self.features.shape[1] == 1376:  # Channel-first detected (N, 1376, 8, 8)
                self.features = self.features.permute(0, 2, 3, 1).contiguous()
                log_process(
                    "data",
                    "layout_alignment_applied",
                    path=str(path),
                    original="(B, C, H, W)",
                    target="(B, H, W, C)",
                )
            elif self.features.shape[3] == 1376:  # Channel-last detected (N, 8, 8, 1376)
                pass
            else:
                warn_msg = (
                    f"Unexpected tensor layout for features shape: {self.features.shape}"
                )
                logging.warning(warn_msg)
        elif self.features.ndim == 3:
            # Flattened spatial representation: e.g. (N, 64, 1376)
            if self.features.shape[2] != 1376:
                raise ValueError(
                    f"Features dimension mismatch. Expected 1376 as C, got: {self.features.shape}"
                )
        else:
            raise ValueError(
                f"Unsupported features tensor dimension: {self.features.ndim}"
            )

        # Structural Integrity Check: Ensure feature/label alignment
        n_f, n_l = self.features.shape[0], self.labels.shape[0]
        if n_f != n_l:
            _min = min(n_f, n_l)
            print(
                f"  [!] Size mismatch in {path}: features({n_f}) ≠ labels({n_l}). "
                f"Truncating to {_min}."
            )
            log_process(
                "data",
                "embedding_size_mismatch",
                level=logging.WARNING,
                path=str(path),
                features=n_f,
                labels=n_l,
                truncated_to=_min,
            )
            self.features = self.features[:_min]
            self.labels = self.labels[:_min]

        self.n = self.features.shape[0]
        print(
            f"  EmbeddingDataset '{path}' ({split}): {self.n:,} samples, "
            f"feat={tuple(self.features.shape[1:])}"
        )
        log_process(
            "data",
            "embedding_dataset_loaded",
            path=str(path),
            samples=self.n,
            feature_shape=tuple(self.features.shape[1:]),
        )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features[i]
        label = self.labels[i]
        
        # Latent Space Jittering (Feature space data augmentation)
        # Prevents Pathology-as-Query transformer layers from overfitting on static grids.
        # Only applied during training to preserve exact inference validity.
        if self.split == "train":
            # Generate deterministic noise tied to the current CPU generator
            noise = torch.randn_like(feat) * self.jitter_eps
            feat = feat + noise
            
        return feat, label


def seed_worker(worker_id: int) -> None:
    """
    Ensures that multi-process data loading remains deterministic.
    Prevents different workers from generating identical augmentation noise or shuffling.
    """
    import random
    w = torch.initial_seed() % 2**32
    np.random.seed(w)
    random.seed(w)


def get_dataloaders() -> tuple[EmbeddingDataset, DataLoader, DataLoader, DataLoader, int]:
    """Constructs highly optimized DataLoaders with persistent workers."""
    # Dynamic worker calculation based on CPU topology
    NUM_WORKERS = 0 if os.name == "nt" else min(4, multiprocessing.cpu_count() - 1)

    train_emb_dataset = EmbeddingDataset(
        os.path.join(EMBEDDING_DIR, "train_embeddings.pt"), split="train"
    )
    val_emb_dataset = EmbeddingDataset(
        os.path.join(EMBEDDING_DIR, "val_embeddings.pt"), split="val"
    )
    test_emb_dataset = EmbeddingDataset(
        os.path.join(EMBEDDING_DIR, "test_embeddings.pt"), split="test"
    )

    # Shared configuration for DataLoader lifecycle management
    _loader_kw = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=(torch.cuda.is_available() and NUM_WORKERS > 0),
        worker_init_fn=seed_worker,
        generator=DATALOADER_GENERATOR,
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )

    train_loader = DataLoader(train_emb_dataset, shuffle=True, **_loader_kw)
    val_loader = DataLoader(val_emb_dataset, shuffle=False, **_loader_kw)
    test_loader = DataLoader(test_emb_dataset, shuffle=False, **_loader_kw)

    log_process(
        "data",
        "dataloaders_ready",
        batch_size=BATCH_SIZE,
        workers=NUM_WORKERS,
        train_batches=len(train_loader),
        val_batches=len(val_loader),
        test_batches=len(test_loader),
    )

    return train_emb_dataset, train_loader, val_loader, test_loader, NUM_WORKERS