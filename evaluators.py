from __future__ import annotations

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
    Bayesian Deep Ensemble Evaluator using Test-Time Augmentation (TTA).
    Averages predictions directly in logit space (z_v) to preserve linear 
    relations prior to Prior-Aware Subjective Logic probability mapping.
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
        all_logits, all_lbls = [], []
        
        for feats, lbls in tqdm(loader, desc="  Bayesian Inference", leave=False):
            feats = feats.to(self.device)
            batch_logits = []

            for m in self.models:
                m.eval()
                # Activate Monte Carlo Dropout layers during inference
                for module in m.modules():
                    if isinstance(module, nn.Dropout): 
                        module.train()

                for _ in range(self.T):
                    z_v = m(feats)
                    batch_logits.append(z_v.cpu().numpy())

            # Shape: (M * T, B, C) -> Average directly in logit space to protect Jensen's inequality
            mean_batch_logits = np.mean(batch_logits, axis=0)
            all_logits.append(mean_batch_logits)
            all_lbls.append(lbls.numpy())

        # Shape: (N, C)
        test_logits = np.concatenate(all_logits, axis=0)
        lbls = np.concatenate(all_lbls, axis=0)
        
        # Apply optimal calibration temperature scaling directly to raw logits
        test_logits_t = torch.from_numpy(test_logits).to(self.device)
        if self.temperature is not None:
            temp_t = torch.as_tensor(
                self.temperature, device=self.device, dtype=test_logits_t.dtype
            )
            test_logits_t = test_logits_t / temp_t
            
        # Convert calibrated logit spaces to probabilities via Prior-Aware Subjective Logic
        active_priors = self.models[0].priors if len(self.models) > 0 else self.priors
        prob_t, epistemic_t, _ = get_evidential_metrics(test_logits_t, active_priors)
        
        predictive_mean = prob_t.cpu().numpy()
        total_epistemic_uncertainty = epistemic_t.cpu().numpy().mean(axis=1)

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
        z_v = model(feats.to(device_))
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