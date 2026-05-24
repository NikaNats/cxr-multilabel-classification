from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model import get_evidential_metrics
from utils import expected_calibration_error


class DeepEnsembleTTAEvaluator:
    """
    Bayesian Deep Ensemble Evaluator utilizing Test-Time Augmentation (TTA).
    Calculates Predictive Shannon Entropy natively on the GPU to avoid
    costly host-device memory synchronizations in the inner loops.
    """
    def __init__(
        self,
        model_class: type[nn.Module],
        checkpoint_paths: list[str],
        device_: torch.device,
        adj_norm_np: np.ndarray | None = None,
        num_mc_passes: int = 10,
        priors: torch.Tensor | np.ndarray | None = None,
        temperature: np.ndarray | None = None,
    ):
        self.device = device_
        self.T = num_mc_passes
        self.models: list[nn.Module] = []
        self.priors = priors
        self.temperature = (
            torch.as_tensor(temperature, device=device_) if temperature is not None else None
        )
        
        for ckpt in checkpoint_paths:
            m = model_class(num_classes=14).to(device_)
            if adj_norm_np is not None:
                m.set_adjacency_mask(adj_norm_np)
            if priors is not None:
                m.set_priors(priors)
            
            raw_sd = torch.load(ckpt, map_location=device_, weights_only=True)
            clean = {
                k.replace("module.", ""): v
                for k, v in raw_sd.items()
                if k != "n_averaged"
            }
            m.load_state_dict(clean, strict=False)
            self.models.append(m)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, thresholds: np.ndarray | None = None) -> dict[str, Any]:
        """Runs GPU-accelerated Bayesian inference over the dataloader."""
        all_probs_list, all_lbls, all_entropies = [], [], []
        
        for feats, lbls in tqdm(loader, desc="  Bayesian Inference", leave=False):
            feats = feats.to(self.device, non_blocking=True)
            batch_probs = []

            for m in self.models:
                m.eval()
                for module in m.modules():
                    if isinstance(module, nn.Dropout): 
                        module.train()

                for _ in range(self.T):
                    z_v = m(feats)
                    
                    if self.temperature is not None:
                        z_v = z_v / self.temperature
                        
                    probs_pass = torch.sigmoid(z_v)
                    batch_probs.append(probs_pass)

            # Stack along evaluation dimension on the GPU. Shape: (M * T, B, C)
            batch_probs_tensor = torch.stack(batch_probs, dim=0)
            
            # Compute ensemble average in GPU space
            mean_probs = torch.mean(batch_probs_tensor, dim=0) # Shape: (B, C)
            
            # Compute Predictive Shannon Entropy on the GPU
            eps = 1e-8
            predictive_entropy = - (
                mean_probs * torch.log2(mean_probs + eps) + 
                (1.0 - mean_probs) * torch.log2(1.0 - mean_probs + eps)
            ) # Shape: (B, C)
            
            all_probs_list.append(mean_probs.cpu())
            all_lbls.append(lbls)
            all_entropies.append(predictive_entropy.cpu())

        # Concatenate arrays once at the end
        predictive_mean = torch.cat(all_probs_list, dim=0).numpy()
        lbls = torch.cat(all_lbls, dim=0).numpy()
        predictive_entropies = torch.cat(all_entropies, dim=0).numpy()
        
        total_epistemic_uncertainty = predictive_entropies.mean(axis=1)

        if thresholds is None: 
            thresholds = np.full(lbls.shape[1], 0.5)
        preds_binary = (predictive_mean >= thresholds).astype(int)

        auc = roc_auc_score(lbls, predictive_mean, average="macro")
        f1 = f1_score(lbls, preds_binary, average="macro")
        ece = expected_calibration_error(predictive_mean, lbls)

        return {
            "auc": auc, 
            "f1": f1, 
            "per_class_ece": ece, 
            "predictive_mean": predictive_mean, 
            "labels": lbls,
            "epistemic_variance": total_epistemic_uncertainty
        }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device_: torch.device,
    priors: torch.Tensor | np.ndarray | None = None,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float, float, np.ndarray, np.ndarray, list[float]]:
    """Runs a standard validation pass and returns primary metrics."""
    model.eval()
    all_p, all_l = [], []
    active_priors = model.priors if hasattr(model, "priors") else priors
    
    for feats, lbls in tqdm(loader, desc="  Val", leave=False):
        z_v = model(feats.to(device_, non_blocking=True))
        probs, _, _ = get_evidential_metrics(z_v, active_priors)
        all_p.append(probs.cpu().numpy())
        all_l.append(lbls.numpy())

    P = np.vstack(all_p)
    L = np.vstack(all_l)
    n_cls = P.shape[1]
    
    class_aucs = []
    for i in range(n_cls):
        try:
            if len(np.unique(L[:, i])) > 1:
                class_aucs.append(roc_auc_score(L[:, i], P[:, i]))
            else:
                class_aucs.append(0.5)
        except Exception:
            class_aucs.append(0.5)
            
    macro_auc = float(np.mean(class_aucs))
    binary_thresh = thresholds if thresholds is not None else 0.5
    f1 = float(f1_score(L, (P >= binary_thresh).astype(int), average="macro"))
    mAP = float(average_precision_score(L, P, average="macro"))
    
    return macro_auc, f1, mAP, P, L, class_aucs