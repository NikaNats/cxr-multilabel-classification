import os
import hashlib
import multiprocessing
import numpy as np
import torch
import logging
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from medmnist import ChestMNIST
from config import GLOBAL_SEED, DATALOADER_GENERATOR, log_process

IMAGE_SIZE = 224
BATCH_SIZE = 64
EMBEDDING_DIR = "/workspace/Extraction/cxr_embeddings_10percent"

_extract_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]
)

def _img_hashes(arr: np.ndarray) -> list:
    """MD5 hash of each image's raw uint8 bytes."""
    return [hashlib.md5(arr[i].tobytes()).hexdigest() for i in range(len(arr))]

def get_raw_datasets():
    print("[*] Downloading / loading ChestMNIST 224×224 (MedMNIST+)...")
    train_dataset_raw = ChestMNIST(split="train", transform=_extract_transform, download=True, size=IMAGE_SIZE)
    val_dataset_raw = ChestMNIST(split="val", transform=_extract_transform, download=True, size=IMAGE_SIZE)
    test_dataset_raw = ChestMNIST(split="test", transform=_extract_transform, download=True, size=IMAGE_SIZE)

    print("[*] MD5 cross-split deduplication audit...")
    train_hash_set = set(_img_hashes(train_dataset_raw.imgs))
    raw_val_hashes = _img_hashes(val_dataset_raw.imgs)
    raw_test_hashes = _img_hashes(test_dataset_raw.imgs)

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
    
    class_names = [train_dataset_raw.info["label"][str(i)] for i in range(len(train_dataset_raw.info["label"]))]
    
    log_process("data", "dataset_ready", train=len(train_dataset_raw), val=len(val_dataset_raw), test=len(test_dataset_raw), classes=len(class_names), val_dups_removed=n_val_rm, test_dups_removed=n_test_rm)
    
    return train_dataset_raw, val_dataset_raw, test_dataset_raw, class_names

class EmbeddingDataset(Dataset):
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding file not found: {path}\nPlease run extraction first.")
        data = torch.load(path, map_location="cpu", weights_only=True)
        self.features = data["features"].float()
        self.labels = data["labels"].float()
        n_f, n_l = self.features.shape[0], self.labels.shape[0]
        if n_f != n_l:
            _min = min(n_f, n_l)
            print(f"  [!] Size mismatch in {path}: features({n_f}) ≠ labels({n_l}). Truncating to {_min}.")
            log_process("data", "embedding_size_mismatch", level=logging.WARNING, path=path, features=n_f, labels=n_l, truncated_to=_min)
            self.features = self.features[:_min]
            self.labels = self.labels[:_min]
        self.n = self.features.shape[0]
        print(f"  EmbeddingDataset '{path}': {self.n:,} samples, feat={tuple(self.features.shape[1:])}")
        log_process("data", "embedding_dataset_loaded", path=path, samples=self.n, feature_shape=tuple(self.features.shape[1:]))

    def __len__(self): return self.n
    def __getitem__(self, i): return self.features[i], self.labels[i]

def seed_worker(worker_id):
    import random
    w = torch.initial_seed() % 2**32
    np.random.seed(w)
    random.seed(w)

def get_dataloaders():
    NUM_WORKERS = 0 if os.name == "nt" else min(4, multiprocessing.cpu_count() - 1)
    train_emb_dataset = EmbeddingDataset(os.path.join(EMBEDDING_DIR, "train_embeddings.pt"))
    val_emb_dataset   = EmbeddingDataset(os.path.join(EMBEDDING_DIR, "val_embeddings.pt"))
    test_emb_dataset  = EmbeddingDataset(os.path.join(EMBEDDING_DIR, "test_embeddings.pt"))
    
    _loader_kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=(torch.cuda.is_available() and NUM_WORKERS > 0), 
                      worker_init_fn=seed_worker, generator=DATALOADER_GENERATOR, persistent_workers=(NUM_WORKERS > 0), drop_last=False)
    
    train_loader = DataLoader(train_emb_dataset, shuffle=True,  **_loader_kw)
    val_loader   = DataLoader(val_emb_dataset,   shuffle=False, **_loader_kw)
    test_loader  = DataLoader(test_emb_dataset,  shuffle=False, **_loader_kw)
    
    log_process("data", "dataloaders_ready", batch_size=BATCH_SIZE, workers=NUM_WORKERS, train_batches=len(train_loader), val_batches=len(val_loader), test_batches=len(test_loader))
    
    return train_emb_dataset, train_loader, val_loader, test_loader, NUM_WORKERS
