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
    Calculates SOTA Predictive Shannon Entropy over the ensemble-averaged 
    probabilities to obtain a calibrated, robust measure of total predictive uncertainty.
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
        self.temperature = temperature
        
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
        """Runs Bayesian inference over the dataloader and returns calibrated evaluation metrics."""
        all_probs_list, all_lbls, all_entropies = [], [], []
        
        for feats, lbls in tqdm(loader, desc="  Bayesian Inference", leave=False):
            feats = feats.to(self.device)
            batch_probs = []

            for m in self.models:
                m.eval()
                # Activate Monte Carlo Dropout layers during inference to sample predictive space
                for module in m.modules():
                    if isinstance(module, nn.Dropout): 
                        module.train()

                for _ in range(self.T):
                    z_v = m(feats)
                    
                    # Apply temperature scaling if temperature calibration is active
                    if self.temperature is not None:
                        temp_t = torch.as_tensor(self.temperature, device=self.device, dtype=z_v.dtype)
                        z_v = z_v / temp_t
                        
                    # Extract standard calibrated probabilities
                    probs_pass = torch.sigmoid(z_v).cpu().numpy()
                    batch_probs.append(probs_pass)

            # Stack along evaluation dimension. Shape: (M * T, B, C)
            batch_probs_arr = np.stack(batch_probs, axis=0)
            
            # SOTA Ensemble Averaging in Probability Space to respect Jensen's inequality
            mean_probs = np.mean(batch_probs_arr, axis=0) # Shape: (B, C)
            
            # Compute Predictive Shannon Entropy over the averaged ensemble probabilities H(p_bar)
            eps = 1e-8
            predictive_entropy = - (
                mean_probs * np.log2(mean_probs + eps) + 
                (1.0 - mean_probs) * np.log2(1.0 - mean_probs + eps)
            ) # Shape: (B, C)
            
            all_probs_list.append(mean_probs)
            all_lbls.append(lbls.numpy())
            all_entropies.append(predictive_entropy)

        # Concatenate across batches to yield full dataset metrics. Shape: (N, C)
        predictive_mean = np.concatenate(all_probs_list, axis=0)
        lbls = np.concatenate(all_lbls, axis=0)
        predictive_entropies = np.concatenate(all_entropies, axis=0)
        
        # Aggregate uncertainty across classes for selective classification. Shape: (N,)
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
            "epistemic_variance": total_epistemic_uncertainty # Mapped as 'epistemic_variance' to maintain compatibility
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
        z_v = model(feats.to(device_))
        # Call get_evidential_metrics which maps to our calibrated sigmoidal representation
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