from __future__ import annotations

import gc
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.swa_utils import SWALR, AveragedModel, update_bn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Any

from config import log_process, DEVICE
from evaluators import validate
from model import CXR_Synapse_Foundation  # AsymmetricLoss იმპორტი აღარ არის საჭირო
from utils import ensure_radlex_embeddings, RADLEX_PATHOLOGIES, EarlyStopping

CFG: dict[str, Any] = {
    "num_epochs": 35,
    "swa_start": 25,
    "batch_size": 64,
    "base_lr": 2e-4,
    "weight_decay": 0.05,
    "warmup_steps": 200,
    "patience": 10,
    "max_grad_norm": 1.0,
}


# ==============================================================================
# § 1  STANDALONE CLASS-BALANCED ASYMMETRIC LOSS (CB-ASL)
# ==============================================================================
class ClassBalancedAsymmetricLoss(nn.Module):
    """
    Numerically stable Class-Balanced Asymmetric Loss (CB-ASL).
    Dynamically scales loss gradients using inverse Effective Number of Samples (ENS).
    """

    def __init__(
            self,
            gamma_neg: float = 4.0,
            gamma_pos: float = 1.0,
            clip: float = 0.05,
            eps: float = 1e-8,
            class_weights: torch.Tensor | None = None
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, x: torch.Tensor, y: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        log_xs_pos = F.logsigmoid(x)
        log_xs_neg = F.logsigmoid(-x)

        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg_clipped = (xs_neg + self.clip).clamp(max=1.0)
            log_xs_neg = torch.log(xs_neg_clipped)
        else:
            xs_neg_clipped = xs_neg

        loss_pos = y * log_xs_pos
        loss_neg = (1.0 - y) * log_xs_neg

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            if self.gamma_pos > 0:
                factors_pos = (1.0 - xs_pos) ** self.gamma_pos
                loss_pos *= factors_pos
            if self.gamma_neg > 0:
                factors_neg = (1.0 - xs_neg_clipped) ** self.gamma_neg
                loss_neg *= factors_neg

        loss = loss_pos + loss_neg

        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0)

        return -loss.mean()


# ==============================================================================
# § 2  TRAINING ENGINE
# ==============================================================================

def train_single_model(
        seed: int,
        adj_norm_np: np.ndarray,
        train_emb_dataset: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_workers: int,
) -> str:
    """Trains a single instance of the foundation network under a specified random seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ckpt_path = f"CXR_Synapse_Foundation_Seed_{seed}.pth"

    model = CXR_Synapse_Foundation(num_classes=14).to(DEVICE)
    model.set_adjacency_mask(adj_norm_np)

    radlex = ensure_radlex_embeddings(
        "radlex_embeddings_14.pth",
        RADLEX_PATHOLOGIES,
        "microsoft/BiomedVLP-BioViL-T",
        DEVICE,
    )
    model.set_radlex_embeddings(radlex.detach())

    train_labels_np = train_emb_dataset.labels.numpy().astype(np.float32)
    priors = np.mean(train_labels_np, axis=0)
    priors = np.clip(priors, 1e-4, 1.0 - 1e-4)
    model.set_priors(priors)

    beta = 0.9999
    class_counts = np.clip(np.sum(train_labels_np, axis=0), 1.0, None)
    ens = (1.0 - np.power(beta, class_counts)) / (1.0 - beta)
    raw_weights = 1.0 / ens
    normalized_weights = raw_weights / np.mean(raw_weights)
    class_weights_tensor = torch.from_numpy(normalized_weights).to(DEVICE)

    loss_fn = ClassBalancedAsymmetricLoss(
        gamma_neg=4.0,
        gamma_pos=1.0,
        clip=0.05,
        class_weights=class_weights_tensor
    ).to(DEVICE)

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 0 or len(param.shape) == 1 or name.endswith(".bias") or "pos_embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": CFG["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=CFG["base_lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    with torch.no_grad():
        radlex_ref = radlex.detach().to(DEVICE)
        semantic_anchor = F.cosine_similarity(
            radlex_ref.unsqueeze(1), radlex_ref.unsqueeze(0), dim=-1
        )
        p_target = F.softmax(semantic_anchor / 0.1, dim=-1)

    total_steps = CFG["num_epochs"] * len(train_loader)
    warmup_steps = CFG["warmup_steps"]

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=5e-5, anneal_epochs=3)
    early = EarlyStopping(patience=CFG["patience"], path=ckpt_path)

    for epoch in range(1, CFG["num_epochs"] + 1):
        model.train()
        run_loss = 0.0
        valid_steps = 0
        pbar = tqdm(
            train_loader,
            desc=f"  Ep {epoch:>2}/{CFG['num_epochs']}",
            leave=False,
        )

        for feats, lbls in pbar:
            feats = feats.to(DEVICE, non_blocking=True)
            lbls = lbls.to(DEVICE, non_blocking=True).float()

            if train_emb_dataset.split == "train" and train_emb_dataset.jitter_eps > 0:
                noise = torch.randn_like(feats) * train_emb_dataset.jitter_eps
                feats = feats + noise

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                z_v = model(feats)
                classification_loss = loss_fn(z_v, lbls, epoch)

                current_queries = model.pathology_router.text_proj(model.radlex_emb)
                norm_queries = F.normalize(current_queries, p=2, dim=-1)
                current_topology = torch.matmul(norm_queries, norm_queries.t())

                q_current = F.log_softmax(current_topology / 0.1, dim=-1)
                anchor_loss = F.kl_div(q_current, p_target, reduction="batchmean")

                loss = classification_loss + (0.5 * anchor_loss)

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()

            if epoch < CFG["swa_start"]:
                scheduler.step()

            run_loss += loss.item()
            valid_steps += 1
            pbar.set_postfix(
                {
                    "total": f"{loss.item():.3f}",
                    "asl_loss": f"{classification_loss.item():.3f}",
                    "anchor_loss": f"{anchor_loss.item():.3f}",
                }
            )

        if epoch >= CFG["swa_start"]:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            update_bn(train_loader, swa_model, device=DEVICE)

        eval_m = swa_model if epoch >= CFG["swa_start"] else model
        v_auc, v_f1, _, _, _, _ = validate(eval_m, val_loader, DEVICE, priors=priors)

        model_to_save = eval_m.module if isinstance(eval_m, AveragedModel) else eval_m
        saved = early(v_auc, model_to_save)

        swa_t = "[SWA]" if epoch >= CFG["swa_start"] else "     "
        mean_loss = run_loss / max(valid_steps, 1)
        print(
            f"  {swa_t} Ep {epoch:>2} | Loss {mean_loss:.4f} | "
            f"AUC {v_auc:.4f} | (F1@0.5: {v_f1:.4f}){' ✓' if saved else ''}"
        )

        log_process(
            "train",
            f"epoch_{epoch}_metrics",
            total_loss=f"{mean_loss:.4f}",
            gpu_mem=f"{torch.cuda.max_memory_allocated() / 1e9:.2f}GB",
        )
        if early.early_stop:
            break

    del model, optimizer, scaler, swa_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return ckpt_path


def train_ensemble(
        seeds: list[int] | np.ndarray,
        adj_norm_np: np.ndarray,
        train_emb_dataset: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_workers: int,
) -> list[str]:
    ensemble_checkpoints = []
    for seed in seeds:
        ckpt = train_single_model(
            seed,
            adj_norm_np,
            train_emb_dataset,
            train_loader,
            val_loader,
            num_workers,
        )
        ensemble_checkpoints.append(ckpt)
    return ensemble_checkpoints
