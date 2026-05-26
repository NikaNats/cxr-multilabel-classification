from __future__ import annotations

import hashlib
import logging
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from medmnist import ChestMNIST
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import DATALOADER_GENERATOR, log_process

IMAGE_SIZE: int = 224
BATCH_SIZE: int = 64

EMBEDDING_DIR: str = "/workspace/Extraction/cxr_embeddings_10percent"

_extract_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]
)


def _hash_chunk(chunk: np.ndarray) -> list[str]:
    """Hashes a chunk of images using Blake2b."""
    return [
        hashlib.blake2b(img.tobytes(), digest_size=16).hexdigest()
        for img in chunk
    ]


def _img_hashes_parallel(arr: np.ndarray) -> list[str]:
    """
    Generates cryptographically secure, bit-exact Blake2b fingerprints
    in parallel across all available CPU cores.
    """
    num_workers = min(multiprocessing.cpu_count(), 8)
    chunks = np.array_split(arr, num_workers)
    
    hashes = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(_hash_chunk, chunks)
        for chunk_hashes in results:
            hashes.extend(chunk_hashes)
    return hashes


def get_raw_datasets() -> tuple[ChestMNIST, ChestMNIST, ChestMNIST, list[str]]:
    """
    Acquires ChestMNIST data and executes a parallelized Forensic Deduplication Audit.
    Ensures zero-leakage via parallelized bitwise hashing.
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

    print("[*] Parallelized Blake2b cross-split deduplication audit...")
    train_hash_set = set(_img_hashes_parallel(train_dataset_raw.imgs))
    raw_val_hashes = _img_hashes_parallel(val_dataset_raw.imgs)
    raw_test_hashes = _img_hashes_parallel(test_dataset_raw.imgs)

    val_keep = np.array([h not in train_hash_set for h in raw_val_hashes])
    n_val_rm = int((~val_keep).sum())
    if n_val_rm:
        val_dataset_raw.imgs = val_dataset_raw.imgs[val_keep]
        val_dataset_raw.labels = val_dataset_raw.labels[val_keep]
        val_dataset_raw.info["n_samples"]["val"] = int(val_keep.sum())
        print(f"  Removed {n_val_rm} duplicate(s) from Val.")

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
    """

    def __init__(
        self, path: str | os.PathLike, split: str = "train", jitter_eps: float = 0.015
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Embedding file not found: {path}\nPlease run extraction first."
            )

        data = torch.load(path, map_location="cpu", weights_only=True)
        self.features = data["features"].float()
        self.labels = data["labels"].float()
        self.split = split
        self.jitter_eps = jitter_eps

        if self.features.ndim == 4:
            if self.features.shape[1] == 1376:  # Channel-first detected
                self.features = self.features.permute(0, 2, 3, 1).contiguous()
                log_process(
                    "data",
                    "layout_alignment_applied",
                    path=str(path),
                    original="(B, C, H, W)",
                    target="(B, H, W, C)",
                )
            elif self.features.shape[3] == 1376:
                pass
            else:
                warn_msg = f"Unexpected tensor layout: {self.features.shape}"
                logging.warning(warn_msg)
        elif self.features.ndim == 3:
            if self.features.shape[2] != 1376:
                raise ValueError(f"Features dimension mismatch. Expected 1376, got: {self.features.shape}")
        else:
            raise ValueError(f"Unsupported features tensor dimension: {self.features.ndim}")

        n_f, n_l = self.features.shape[0], self.labels.shape[0]
        if n_f != n_l:
            _min = min(n_f, n_l)
            print(f"  [!] Size mismatch in {path}: features({n_f}) ≠ labels({n_l}). Truncating to {_min}.")
            self.features = self.features[:_min]
            self.labels = self.labels[:_min]

        self.n = self.features.shape[0]
        print(f"  EmbeddingDataset '{path}' ({split}): {self.n:,} samples, feat={tuple(self.features.shape[1:])}")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[i], self.labels[i]


def seed_worker(worker_id: int) -> None:
    import random
    w = torch.initial_seed() % 2**32
    np.random.seed(w)
    random.seed(w)


def get_dataloaders() -> tuple[EmbeddingDataset, DataLoader, DataLoader, DataLoader, int]:
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