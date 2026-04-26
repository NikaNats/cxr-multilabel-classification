import time
import math
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.swa_utils import SWALR, AveragedModel
from tqdm.auto import tqdm

from config import log_process, DEVICE
from utils import ensure_radlex_embeddings, compute_logit_adjustment, RADLEX_PATHOLOGIES, EarlyStopping
from model import CXR_Synapse_Foundation, LogitAdjustedAsymmetricEvidentialLoss, target_distribution, batch_hard_triplet_loss
from evaluators import validate
import logging

# ============================================================
# HYPERPARAMETER CONFIGURATION (Clinical Learner Priors)
# ============================================================
CFG = {
    "num_epochs": 35,
    "swa_start": 25,       # Stochastic Weight Averaging starts at the tail of training
    "batch_size": 64,
    "base_lr": 2e-4,       # Initial learning rate for the Pro-head
    "weight_decay": 0.05,  # Decoupled weight decay for AdamW
    "warmup_steps": 200,   # Linear LR warmup to stabilize early gradients
    "trip_lambda": 0.05,   # Weight for Batch-Hard Triplet Loss (Latent separation)
    "trip_margin": 0.3,    # Margin for Triplet space separation
    "cluster_lambda": 0.10, # Weight for Student-t Clustering (Manifold alignment)
    "patience": 10,        # Early stopping patience based on Validation AUC
    "max_grad_norm": 1.0,  # Gradient clipping to prevent exploding gradients
    "nan_abort_ratio": 0.3, # Abort threshold for numerical instability
}

ENSEMBLE_CKPT_TMPL = "CXR_Synapse_Foundation_Seed_{seed}.pth"

def train_single_model(seed, adj_norm_np, train_emb_dataset, train_loader, val_loader, num_workers):
    """
    Trains a single CXR-Synapse instance with absolute reproducibility.
    
    This function orchestrates the entire learning lifecycle:
    1. Knowledge Graph Injection (RadLex Adjacency)
    2. Long-tail correction (Logit Adjustment)
    3. Evidential Sparsity Optimization
    """
    # Force bit-exact determinism for this specific seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    ckpt_path = ENSEMBLE_CKPT_TMPL.format(seed=seed)
    log_process("train", "seed_started", seed=seed, checkpoint=ckpt_path)
    
    # ----------------------------------------------------
    # MODEL & CLINICAL KNOWLEDGE SETUP
    # ----------------------------------------------------
    model = CXR_Synapse_Foundation(num_classes=14).to(DEVICE)
    model.set_adjacency_mask(adj_norm_np)
    
    # Inject BioViL-T clinical embeddings into the GraphGPS classifier
    radlex = ensure_radlex_embeddings(path="radlex_embeddings_14.pth", pathologies=RADLEX_PATHOLOGIES, model_name="microsoft/BiomedVLP-BioViL-T", device_=DEVICE)
    model.set_radlex_embeddings(radlex.detach())

    # Pre-calculate logit adjustment vector based on training label distribution
    _train_labels_np = train_emb_dataset.labels.numpy().astype(np.int32)
    logit_adj_vec = compute_logit_adjustment(_train_labels_np, tau=1.0).to(DEVICE)
    
    # Unified Diagnostic Loss: Handles imbalance, asymmetry, and evidential uncertainty
    loss_fn = LogitAdjustedAsymmetricEvidentialLoss(
        logit_adj=logit_adj_vec,
        gamma_pos=0.0,
        gamma_neg=2.0, 
        clip=0.05,
        annealing_epochs=20, # Delay KL penalty to allow base classification learning
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["base_lr"], weight_decay=CFG["weight_decay"])
    
    # ----------------------------------------------------
    # SCHEDULING & PRECISION PROTOCOLS
    # ----------------------------------------------------
    total_steps = CFG["num_epochs"] * len(train_loader)
    warmup_steps = CFG["warmup_steps"]
    
    # Cosine Annealing with Linear Warmup
    def _lr_lambda(step):
        if step < warmup_steps: return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    
    # Modern Mixed Precision (AMP) for optimal GPU utilization
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == "cuda"))
    
    # SWA Setup: Finds wider local minima for better out-of-distribution calibration
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=5e-5, anneal_epochs=3)
    
    early = EarlyStopping(patience=CFG["patience"], path=ckpt_path)
    t0 = time.time()
    
    # ----------------------------------------------------
    # MAIN EPOCH LOOP
    # ----------------------------------------------------
    for epoch in range(1, CFG["num_epochs"] + 1):
        # Gradual clustering activation
        kl_w = CFG["cluster_lambda"] if epoch > 5 else 0.0
        
        model.train()
        run_loss = nan_steps = valid_steps = 0
        total_b = len(train_loader)
        pbar = tqdm(train_loader, desc=f"  Ep {epoch:>2}/{CFG['num_epochs']}", leave=False)
        
        for feats, lbls in pbar:
            feats, lbls = feats.to(DEVICE), lbls.to(DEVICE).float()
            optimizer.zero_grad(set_to_none=True)
            
            # AUTOMATIC MIXED PRECISION (AMP)
            with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda")):
                # Forward pass returns diagnostic logits and latent representations
                feat_logit, fused, q = model(feats)
                
                # Primary Evidential Classification Loss
                total_clf_loss = loss_fn(feat_logit.float(), lbls.float(), epoch)
                
                # Auxiliary Task 1: Student-t Manifold Clustering
                p_lbl = torch.argmax(q, 1)
                p_dist = target_distribution(q.detach())
                kl_loss = F.kl_div((q + 1e-12).log(), p_dist, reduction="batchmean")
                
                # Auxiliary Task 2: Batch-Hard Triplet Separation
                trip = batch_hard_triplet_loss(fused, p_lbl, CFG["trip_margin"])
                
                # Multi-objective total loss
                loss = total_clf_loss + CFG["trip_lambda"] * trip + kl_w * kl_loss
            
            # Numerical Stability Guard
            if not torch.isfinite(loss):
                nan_steps += 1
                if nan_steps > total_b * CFG["nan_abort_ratio"]:
                    log_process("train", "epoch_aborted_nan_ratio", level=logging.WARNING, seed=seed, epoch=epoch)
                    break
                continue

            # Scaled backpropagation
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # Gradient clipping: Crucial for stable latent clustering
            gn = nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
            
            if torch.isfinite(gn):
                scaler.step(optimizer)
            else:
                nan_steps += 1
            scaler.update()
            
            if epoch < CFG["swa_start"]: scheduler.step()
            run_loss += loss.item()
            valid_steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "clf": f"{total_clf_loss.item():.4f}"})

        # Stochastic Weight Averaging update logic
        if epoch >= CFG["swa_start"]:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        
        avg_l = run_loss / max(valid_steps, 1)
        eval_m = swa_model if epoch >= CFG["swa_start"] else model
        
        # CLINICAL VALIDATION
        # We prioritize AUC as the primary stopping metric due to label imbalance.
        v_auc, v_f1, _, _, _ = validate(eval_m, val_loader, DEVICE)
        saved = early(v_auc, eval_m) 
        
        swa_t = "[SWA]" if epoch >= CFG["swa_start"] else "     "
        tag = " ✓" if saved else ""
        print(f"  {swa_t} Ep {epoch:>2} | Loss {avg_l:.4f} | AUC {v_auc:.4f} | (F1@0.5: {v_f1:.4f}){tag}")
        
        log_process("train", "epoch_completed", seed=seed, epoch=epoch, loss=f"{avg_l:.4f}", val_auc=f"{v_auc:.4f}", best_saved=saved, swa=(epoch >= CFG["swa_start"]))

        if early.early_stop:
            log_process("train", "early_stopping_triggered", level=logging.WARNING, seed=seed, epoch=epoch, best_auc=f"{early.best_score:.4f}")
            break
            
    elapsed = time.time() - t0
    print(f"  ✓ Seed {seed} done ({elapsed//60:.0f}m {elapsed%60:.0f}s) | Best AUC: {early.best_score:.4f}")
    return ckpt_path

def train_ensemble(seeds, adj_norm_np, train_emb_dataset, train_loader, val_loader, num_workers):
    """
    Orchestrates the training of multiple ensemble members.
    Ensembling is the gold standard for quantifying Epistemic Uncertainty.
    """
    ensemble_checkpoints = []
    log_process("train", "ensemble_training_started", members=len(seeds), epochs=CFG["num_epochs"])
    for _i, _seed in enumerate(seeds):
        log_process("train", "ensemble_member_started", index=_i + 1, total=len(seeds), seed=_seed)
        ckpt = train_single_model(_seed, adj_norm_np, train_emb_dataset, train_loader, val_loader, num_workers)
        ensemble_checkpoints.append(ckpt)
    return ensemble_checkpoints