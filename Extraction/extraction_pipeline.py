"""
CXR-Synapse: 10% Decile Extraction Pipeline with Checkpointing
Optimized for long-running processes and interruption recovery.
"""
import argparse
import datetime
import gc
import hashlib
import io
import json
import logging
import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", message=".*Protobuf gencode version.*")

import numpy as np
import cv2
import torch
import tensorflow as tf
import tensorflow_text
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

FRACTION = 0.10
RANDOM_SEED = 42
BATCH_SAVE_SIZE = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("synapse_resumable.log"), logging.StreamHandler()],
)

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

SCHEMA_VERSION = "v8_checkpoint_v1"
FEATURE_DESC = {
    "elixr_input": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([14], tf.int64),
    "hash": tf.io.FixedLenFeature([], tf.string),
}


def encode_sample(args):
    idx, img_np, label = args
    img = img_np[:, :, 0] if img_np.ndim == 3 else img_np
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-5) * 65535.0
    ok, buf = cv2.imencode(".png", img.astype(np.uint16))
    png_bytes = buf.tobytes() if ok else b""
    inner = tf.train.Example(features=tf.train.Features(feature={
        "image/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[png_bytes])),
        "image/format": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"png"])),
    }))
    return idx, inner.SerializeToString(), label.tolist(), hashlib.md5(img_np.tobytes()).hexdigest()


def build_tfrecords(cache_dir, force_rebuild=False):
    from medmnist import ChestMNIST
    os.makedirs(cache_dir, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    for split in ["train", "val", "test"]:
        tfr_path = os.path.join(cache_dir, f"{split}_10percent.tfrecord")
        ver_path = os.path.join(cache_dir, f"{split}_10percent_schema.txt")

        if not force_rebuild and os.path.exists(tfr_path) and \
                os.path.exists(ver_path) and open(ver_path).read().strip() == SCHEMA_VERSION:
            logging.info(f"[SKIP] {split} cache exists.");
            continue

        dataset = ChestMNIST(split=split, download=True, size=224)
        limit = int(len(dataset.imgs) * FRACTION)
        indices = np.arange(len(dataset.imgs))
        np.random.shuffle(indices)
        selected_indices = indices[:limit]

        args = [(i, dataset.imgs[idx], dataset.labels[idx]) for i, idx in enumerate(selected_indices)]
        results = [None] * len(args)

        with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 12)) as ex:
            futures = {ex.submit(encode_sample, a): a[0] for a in args}
            for f in tqdm(as_completed(futures), total=len(args), desc=f"Enc {split}"):
                idx, *rest = f.result()
                results[idx] = rest

        with tf.io.TFRecordWriter(tfr_path) as writer:
            for elixr_bytes, label, img_hash in results:
                ex = tf.train.Example(features=tf.train.Features(feature={
                    "elixr_input": tf.train.Feature(bytes_list=tf.train.BytesList(value=[elixr_bytes])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                    "hash": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_hash.encode()])),
                }))
                writer.write(ex.SerializeToString())
        open(ver_path, "w").write(SCHEMA_VERSION)


def parse_tfrecord(s):
    return tf.io.parse_single_example(s, FEATURE_DESC)


def extract_with_resume(tfr_path, split, model, out_key, output_dir):
    infer = model.signatures["serving_default"]

    ckpt_dir = os.path.join(output_dir, f"checkpoints_{split}")
    os.makedirs(ckpt_dir, exist_ok=True)
    final_path = os.path.join(output_dir, f"{split}_embeddings.pt")

    if os.path.exists(final_path):
        logging.info(f"[DONE] {split} already fully processed. Skipping.")
        return final_path

    processed_hashes = set()
    checkpoint_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    for cf in checkpoint_files:
        try:
            data = torch.load(os.path.join(ckpt_dir, cf))
            processed_hashes.update(data["hashes"])
        except:
            continue

    logging.info(f"[*] Resuming {split}: {len(processed_hashes)} samples already done.")

    ds = tf.data.TFRecordDataset(tfr_path).map(parse_tfrecord)
    total_samples = sum(1 for _ in tf.data.TFRecordDataset(tfr_path))

    feats, labels, hashes = [], [], []

    with tqdm(total=total_samples, desc=f"Extract {split}", initial=len(processed_hashes)) as pbar:
        for record in ds:
            hsh_str = record["hash"].numpy().decode("ascii")

            if hsh_str in processed_hashes:
                continue

            elixr_str = record["elixr_input"]
            res = infer(input_example=tf.expand_dims(elixr_str, 0))

            feats.append(res[out_key].numpy().astype(np.float16)[0])
            labels.append(record["label"].numpy())
            hashes.append(hsh_str)
            pbar.update(1)

            if len(feats) >= BATCH_SAVE_SIZE:
                save_checkpoint(ckpt_dir, feats, labels, hashes)
                feats, labels, hashes = [], [], []

        if feats:
            save_checkpoint(ckpt_dir, feats, labels, hashes)

    # ფინალური გაერთიანება
    return finalize_split(ckpt_dir, final_path, split)


def save_checkpoint(ckpt_dir, feats, labels, hashes):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ckpt_path = os.path.join(ckpt_dir, f"batch_{timestamp}.pt")
    torch.save({
        "features": np.stack(feats),
        "labels": np.stack(labels),
        "hashes": hashes
    }, ckpt_path)


def finalize_split(ckpt_dir, final_path, split):
    logging.info(f"Merging checkpoints for {split}...")
    all_feats, all_labels, all_hashes = [], [], []

    ckpt_files = sorted([os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt")])

    for f in ckpt_files:
        data = torch.load(f)
        all_feats.append(data["features"])
        all_labels.append(data["labels"])
        all_hashes.extend(data["hashes"])

    if not all_feats:
        return None

    torch.save({
        "features": torch.from_numpy(np.concatenate(all_feats, axis=0)),
        "labels": torch.from_numpy(np.concatenate(all_labels, axis=0)),
        "hashes": all_hashes,
        "model_version": "ELIXR-C_v2",
        "extraction_date": datetime.datetime.now().isoformat(),
    }, final_path)

    logging.info(f"✓ {split.upper()} finalized → {final_path}")
    return final_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--cache_dir", default="./tfrecord_cache_10percent")
    p.add_argument("--out_dir", default="./cxr_embeddings_10percent")
    args = p.parse_args()

    build_tfrecords(args.cache_dir)

    model = tf.saved_model.load(args.model_dir)
    out_key = next(k for k in model.signatures["serving_default"].structured_outputs if "feature" in k.lower())

    os.makedirs(args.out_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        tfr = os.path.join(args.cache_dir, f"{split}_10percent.tfrecord")
        extract_with_resume(tfr, split, model, out_key, args.out_dir)
        tf.keras.backend.clear_session()
        gc.collect()

    logging.info("PIPELINE COMPLETE.")
