"""
train.py — CXR-Synapse Training Pipeline (Decoupled Weight Decay Edition)
═══════════════════════════════════════════════════════════════════════════════
Orchestrates single-model and ensemble training with:
  • SOTA Asymmetric Focal Evidential Loss (Gamma = 2.0).
  • Class-Conditional Evidential Annealing.
  • SOTA Decoupled Weight Decay (Excludes Biases & LayerNorms from L2).
  • Manifold-Preserving LISA (MP-LISA) regularization.
  • Stochastic Weight Averaging (SWA) and AdamW optimization.
"""

import logging
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.swa_utils import SWALR, AveragedModel
from tqdm.auto import tqdm

from config import log_process, DEVICE
from evaluators import validate
from model import CXR_Synapse_Foundation, AsymmetricFocalEvidentialLoss
from utils import ensure_radlex_embeddings, compute_logit_adjustment, RADLEX_PATHOLOGIES, EarlyStopping

CFG = {
    "num_epochs": 35, "swa_start": 25, "batch_size": 64, "base_lr": 2e-4,
    "weight_decay": 0.05, "warmup_steps": 200, "patience": 10, "max_grad_norm": 1.0
}

def train_single_model(seed, adj_norm_np, train_emb_dataset, train_loader, val_loader, num_workers):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ckpt_path = f"CXR_Synapse_Foundation_Seed_{seed}.pth"
    
    model = CXR_Synapse_Foundation(num_classes=14).to(DEVICE)
    model.set_adjacency_mask(adj_norm_np)

    radlex = ensure_radlex_embeddings("radlex_embeddings_14.pth", RADLEX_PATHOLOGIES, "microsoft/BiomedVLP-BioViL-T", DEVICE)
    model.set_radlex_embeddings(radlex.detach())

    _train_labels_np = train_emb_dataset.labels.numpy().astype(np.int32)
    logit_adj_vec = compute_logit_adjustment(_train_labels_np, tau=1.0).to(DEVICE)
    model.set_logit_prior(logit_adj_vec.cpu().numpy())

    # SOTA: Instantiate the newly engineered Asymmetric Focal Evidential Loss
    loss_fn = AsymmetricFocalEvidentialLoss(annealing_epochs=20, gamma=2.0).to(DEVICE)
    
    # SOTA FIX: Decoupled Weight Decay Parameter Grouping
    # We split parameters so that we do NOT apply L2 regularization to biases, 
    # positional encodings, or LayerNorm scale/shift parameters.
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Keep 1D parameters (biases, layer norms, and 2D sincos pos encodings) decay-free
        if len(param.shape) == 1 or name.endswith(".bias") or "pos_embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": CFG["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0} # Absolute 0.0 decay for structural layers
    ]
    
    optimizer = torch.optim.AdamW(optim_groups, lr=CFG["base_lr"])
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == "cuda"))

    # LISA: Calculate the Semantic Anchor (Original RadLex Topology)
    with torch.no_grad():
        radlex_ref = radlex.detach().to(DEVICE)
        semantic_anchor = F.cosine_similarity(radlex_ref.unsqueeze(1), radlex_ref.unsqueeze(0), dim=-1)

    total_steps = CFG["num_epochs"] * len(train_loader)
    warmup_steps = CFG["warmup_steps"]
    
    def _lr_lambda(step):
        if step < warmup_steps: return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=5e-5, anneal_epochs=3)
    early = EarlyStopping(patience=CFG["patience"], path=ckpt_path)

    for epoch in range(1, CFG["num_epochs"] + 1):
        model.train()
        run_loss = valid_steps = 0
        pbar = tqdm(train_loader, desc=f"  Ep {epoch:>2}/{CFG['num_epochs']}", leave=False)

        for feats, lbls in pbar:
            feats, lbls = feats.to(DEVICE), lbls.to(DEVICE).float()
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda")):
                z_posterior, z_v = model(feats)
                evidential_loss = loss_fn(z_posterior, z_v, lbls, epoch)

                # LISA: Language-Invariant Semantic Anchoring
                current_queries = model.pathology_router.text_proj(model.radlex_emb)
                norm_queries = F.normalize(current_queries, p=2, dim=-1)
                current_topology = torch.matmul(norm_queries, norm_queries.t())
                
                # SOTA: Manifold-Preserving LISA (MP-LISA) via KL-Divergence
                p_target = F.softmax(semantic_anchor / 0.1, dim=-1)
                q_current = F.log_softmax(current_topology / 0.1, dim=-1)
                
                # Kullback-Leibler Divergence is mathematically flexible and SOTA
                anchor_loss = F.kl_div(q_current, p_target, reduction="batchmean")
                
                loss = evidential_loss + (0.5 * anchor_loss)
                
            if not torch.isfinite(loss): continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()

            if epoch < CFG["swa_start"]: scheduler.step()
            run_loss += loss.item()
            valid_steps += 1
            pbar.set_postfix({
                "total": f"{loss.item():.3f}",
                "focal_edl": f"{evidential_loss.item():.3f}", 
                "lisa": f"{anchor_loss.item():.3f}"    
            })

        if epoch >= CFG["swa_start"]:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        eval_m = swa_model if epoch >= CFG["swa_start"] else model
        v_auc, v_f1, _, _, _, _ = validate(eval_m, val_loader, DEVICE)
        saved = early(v_auc, eval_m)

        swa_t = "[SWA]" if epoch >= CFG["swa_start"] else "     "
        print(f"  {swa_t} Ep {epoch:>2} | Loss {run_loss/max(valid_steps,1):.4f} | AUC {v_auc:.4f} | (F1@0.5: {v_f1:.4f}){' ✓' if saved else ''}")
        log_process("train", f"epoch_{epoch}_metrics", 
            total_loss=f"{run_loss/valid_steps:.4f}",
            gpu_mem=f"{torch.cuda.max_memory_allocated()/1e9:.2f}GB")
        if early.early_stop: break

    # Clean Memory Management (No C++ Segfaults)
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    import gc; gc.collect()
    
    return ckpt_path

def train_ensemble(seeds, adj_norm_np, train_emb_dataset, train_loader, val_loader, num_workers):
    ensemble_checkpoints = []
    for _seed in seeds:
        ckpt = train_single_model(_seed, adj_norm_np, train_emb_dataset, train_loader, val_loader, num_workers)
        ensemble_checkpoints.append(ckpt)
    return ensemble_checkpoints