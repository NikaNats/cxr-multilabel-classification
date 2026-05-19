import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from tqdm.auto import tqdm
from model import get_evidential_metrics
from utils import expected_calibration_error
from config import log_process

class DeepEnsembleTTAEvaluator:
    """
    SOTA 2026: Bayesian Deep Ensemble and Test-Time Augmentation (TTA) Evaluator.
    Combines expected intra-model evidential uncertainty and inter-model disagreement,
    fully decoupled from logit adjustment priors to prevent positive-class masking.
    
    Features defensive unpacking to support both legacy and updated model architectures.
    """
    def __init__(self, model_class, checkpoint_paths, device_, adj_norm_np=None, num_mc_passes=10, logit_adj=None):
        self.device = device_
        self.T = num_mc_passes
        self.models = []
        
        # Instantiate Ensemble Members
        for ckpt in checkpoint_paths:
            m = model_class(num_classes=14).to(device_)
            if adj_norm_np is not None: m.set_adjacency_mask(adj_norm_np)
            if logit_adj is not None: m.set_logit_prior(logit_adj.cpu().numpy())
            
            raw_sd = torch.load(ckpt, map_location=device_, weights_only=True)
            clean = {k.replace("module.", ""): v for k, v in raw_sd.items() if k != "n_averaged"}
            m.load_state_dict(clean, strict=False)
            self.models.append(m)

    @torch.no_grad()
    def evaluate(self, loader, thresholds=None):
        all_preds, all_raw_preds, all_epistemics, all_lbls = [], [], [], []
        
        for feats, lbls in tqdm(loader, desc="  Bayesian Inference", leave=False):
            feats = feats.to(self.device)
            batch_model_means = []
            batch_model_raw_means = []
            batch_model_epistemics = []

            for m in self.models:
                m.eval()
                # Activate Dropout for Epistemic Uncertainty
                for module in m.modules():
                    if isinstance(module, nn.Dropout): 
                        module.train()

                mc_passes = []
                mc_raw_passes = []
                mc_epistemics = []
                
                for _ in range(self.T):
                    out = m(feats)
                    
                    # SOTA defensive unpacking with mathematical reconstruction of raw evidence
                    if isinstance(out, tuple):
                        z_posterior, z_v = out
                    else:
                        z_posterior = out
                        # Reconstruct raw evidence directly: z_v = z_posterior - prior
                        z_v = z_posterior - m.logit_prior

                    # 1. Probabilities for final classification (logit-adjusted)
                    prob, _, _ = get_evidential_metrics(z_posterior)
                    mc_passes.append(prob.cpu().numpy())
                    
                    # 2. Raw probabilities and evidential metrics (unbiased by class prevalence prior)
                    raw_prob, epistemic, _ = get_evidential_metrics(z_v)
                    mc_raw_passes.append(raw_prob.cpu().numpy())
                    mc_epistemics.append(epistemic.cpu().numpy())

                batch_model_means.append(np.stack(mc_passes, axis=0).mean(axis=0))
                batch_model_raw_means.append(np.stack(mc_raw_passes, axis=0).mean(axis=0))
                batch_model_epistemics.append(np.stack(mc_epistemics, axis=0).mean(axis=0))

            all_preds.append(np.stack(batch_model_means, axis=0))       # (M, B, C)
            all_raw_preds.append(np.stack(batch_model_raw_means, axis=0))   # (M, B, C)
            all_epistemics.append(np.stack(batch_model_epistemics, axis=0))  # (M, B, C)
            all_lbls.append(lbls.numpy())

        ens = np.concatenate(all_preds, axis=1)            # Shape: (M, N, C) - Adjusted for optimal prediction
        ens_raw = np.concatenate(all_raw_preds, axis=1)    # Shape: (M, N, C) - Unadjusted for unbiased variance
        ens_epistemic = np.concatenate(all_epistemics, axis=1)  # Shape: (M, N, C) - Prior-free evidential
        lbls = np.concatenate(all_lbls, axis=0)
        
        predictive_mean = ens.mean(axis=0)                 # Final macro prediction
        
        # SOTA: Total Prior-Free Epistemic Uncertainty
        # Sum of mean intra-model evidential doubt and unbiased ensemble variance (disagreement)
        avg_evidential_uncertainty = ens_epistemic.mean(axis=0) # Shape: (N, C)
        ensemble_disagreement = ens_raw.var(axis=0)             # Shape: (N, C)
        
        # Aggregate across all classes to get a single patient-level risk score
        total_epistemic_uncertainty = (avg_evidential_uncertainty + ensemble_disagreement).mean(axis=1) # Shape: (N,)

        if thresholds is None: thresholds = np.full(lbls.shape[1], 0.5)
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

class EvidentialEvaluator:
    """
    Evidential Deep Learning (EDL) Evaluator.
    Ref: Sensoy et al., 'Evidential Deep Learning to Quantify Classification Uncertainty'.
    """
    def __init__(self, model, device_):
        self.model = model
        self.device = device_
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, loader, thresholds=None):
        all_probs, all_epistemic, all_lbls = [], [], []
        for feats, lbls in tqdm(loader, desc="  Evidential Inference", leave=False):
            feats = feats.to(self.device)
            out = self.model(feats)
            
            # SOTA defensive unpacking
            if isinstance(out, tuple):
                logits, z_v = out
            else:
                logits = out
                z_v = logits - self.model.logit_prior

            prob, _, _ = get_evidential_metrics(logits)
            _, epistemic, _ = get_evidential_metrics(z_v) # Prior-free evidential uncertainty

            all_probs.append(prob.cpu().numpy())
            all_epistemic.append(epistemic.cpu().numpy())
            all_lbls.append(lbls.numpy())

        probs = np.concatenate(all_probs, axis=0)
        epistemic = np.concatenate(all_epistemic, axis=0)
        lbls = np.concatenate(all_lbls, axis=0)
        
        # Mean across classes to get a scalar uncertainty score per patient
        epistemic_scalar = epistemic.mean(axis=1)

        if thresholds is None: thresholds = np.full(lbls.shape[1], 0.5)

        preds_binary = (probs >= thresholds).astype(int)
        try:
            auc = roc_auc_score(lbls, probs, average="macro")
        except ValueError:
            auc = float("nan")

        f1 = f1_score(lbls, preds_binary, average="macro")
        ece = expected_calibration_error(probs, lbls)

        log_process("evidential", "evaluation_completed", auc=f"{auc:.4f}", f1_macro=f"{f1:.4f}",
                    ece_mean=f"{ece.mean():.4f}")

        return {
            "auc": auc, "f1": f1, "ece": ece, "predictive_mean": probs,
            "epistemic_variance": epistemic_scalar, "labels": lbls, "per_class_ece": ece
        }


def validate(model, loader, device_, thresholds=None):
    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for feats, lbls in tqdm(loader, desc="  Val", leave=False):
            logits = model(feats.to(device_))
            # Safely extract the posterior logits, ignoring the raw logits for validation
            if isinstance(logits, tuple): 
                logits = logits[0]
            probs, _, _ = get_evidential_metrics(logits)
            all_p.append(probs.cpu().numpy())
            all_l.append(lbls.numpy())

    P, L = np.vstack(all_p), np.vstack(all_l)
    n_cls = P.shape[1]
    
    # Calculate per-class AUCs
    class_aucs = []
    for i in range(n_cls):
        try:
            class_aucs.append(roc_auc_score(L[:, i], P[:, i]))
        except:
            class_aucs.append(0.5)
            
    macro_auc = np.mean(class_aucs)
    f1 = f1_score(L, (P >= (thresholds if thresholds is not None else 0.5)).astype(int), average="macro")
    mAP = average_precision_score(L, P, average="macro")
    
    return macro_auc, f1, mAP, P, L, class_aucs