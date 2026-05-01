import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from tqdm.auto import tqdm
from model import get_evidential_metrics
from utils import expected_calibration_error

class DeepEnsembleTTAEvaluator:
    def __init__(self, model_class, checkpoint_paths, device_, adj_norm_np=None, num_mc_passes=10, logit_adj=None):
        self.device = device_
        self.T = num_mc_passes
        self.models =[]
        
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
        all_preds, all_lbls = [],[]
        for feats, lbls in tqdm(loader, desc="  Bayesian Inference", leave=False):
            feats = feats.to(self.device)
            batch_model_means =[]

            for m in self.models:
                m.eval()
                # Activate Dropout for Epistemic Uncertainty
                for module in m.modules():
                    if isinstance(module, nn.Dropout): module.train()

                mc_passes =[]
                for _ in range(self.T):
                    z_posterior = m(feats)
                    prob, _, _ = get_evidential_metrics(z_posterior)
                    mc_passes.append(prob.cpu().numpy())

                batch_model_means.append(np.stack(mc_passes, axis=0).mean(axis=0))

            all_preds.append(np.stack(batch_model_means, axis=0))
            all_lbls.append(lbls.numpy())

        ens = np.concatenate(all_preds, axis=1) 
        lbls = np.concatenate(all_lbls, axis=0)
        predictive_mean = ens.mean(axis=0)  

        if thresholds is None: thresholds = np.full(lbls.shape[1], 0.5)
        preds_binary = (predictive_mean >= thresholds).astype(int)

        auc = roc_auc_score(lbls, predictive_mean, average="macro")
        f1 = f1_score(lbls, preds_binary, average="macro")
        ece = expected_calibration_error(predictive_mean, lbls)

        return {"auc": auc, "f1": f1, "per_class_ece": ece, "predictive_mean": predictive_mean, "labels": lbls}

def validate(model, loader, device_, thresholds=None, logit_adj=None):
    model.eval()
    all_p, all_l = [],[]
    with torch.no_grad():
        for feats, lbls in tqdm(loader, desc="  Val", leave=False):
            z_posterior = model(feats.to(device_))
            probs, _, _ = get_evidential_metrics(z_posterior)
            all_p.append(probs.cpu().numpy())
            all_l.append(lbls.numpy())

    P, L = np.vstack(all_p), np.vstack(all_l)
    if thresholds is None: thresholds = np.full(P.shape[1], 0.5)
    
    auc = roc_auc_score(L, P, average="macro")
    f1 = f1_score(L, (P >= thresholds).astype(int), average="macro")
    mAP = average_precision_score(L, P, average="macro")
    return auc, f1, mAP, P, L
    """
    Evidential Deep Learning (EDL) Evaluator.
    Ref: Sensoy et al., 'Evidential Deep Learning to Quantify Classification Uncertainty'.
    
    Provides single-pass uncertainty estimation by treating model outputs 
    as parameters of a Dirichlet distribution, capturing 'Lack of Evidence'.
    """

    def __init__(self, model, device_):
        self.model = model
        self.device = device_
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, loader, thresholds=None):
        """
        Calculates subjective logic parameters (Evidence, Alpha, S) 
        directly from the output logits in a single forward pass.
        """
        all_probs, all_epistemic, all_lbls = [], [], []
        for feats, lbls in tqdm(loader, desc="  Evidential Inference", leave=False):
            feats = feats.to(self.device)
            logits = self.model(feats)
            if isinstance(logits, tuple): logits = logits[0]

            # Decompose logits into Dirichlet-based probability and Epistemic variance
            prob, epistemic, _ = get_evidential_metrics(logits)

            all_probs.append(prob.cpu().numpy())
            all_epistemic.append(epistemic.cpu().numpy())
            all_lbls.append(lbls.numpy())

        probs = np.concatenate(all_probs, axis=0)
        epistemic = np.concatenate(all_epistemic, axis=0)
        lbls = np.concatenate(all_lbls, axis=0)

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
            "epistemic_variance": epistemic, "labels": lbls, "per_class_ece": ece
        }


def validate(model, loader, device_, thresholds=None, logit_adj=None):
    """
    Clinical Validation Engine.
    Executes a high-fidelity evaluation during the training loop.
    
    Returns:
        - AUC: Predictive ranking quality.
        - F1: Harmonic mean of precision and recall.
        - mAP: Area under the Precision-Recall curve (Crucial for long-tail medical labels).
        - P/L: Raw probabilities and ground truth labels for calibration audits.
    """
    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for feats, lbls in tqdm(loader, desc="  Val", leave=False):
            logits = model(feats.to(device_))
            if isinstance(logits, tuple): logits = logits[0]

            # Add the logit adjustment before metrics!
            if logit_adj is not None:
                logits = logits + logit_adj

            # Use strict evidential probabilities to mirror real-world inference
            probs, _, _ = get_evidential_metrics(logits)

            all_p.append(probs.cpu().numpy())
            all_l.append(lbls.numpy())

    P, L = np.vstack(all_p), np.vstack(all_l)

    if thresholds is None: thresholds = np.full(P.shape[1], 0.5)

    try:
        auc = roc_auc_score(L, P, average="macro")
    except ValueError:
        auc = 0.5

    try:
        # ----------------------------------------------------
        # mAP (Nature Fix): Critical for rare pathology recall
        # ----------------------------------------------------
        mAP = average_precision_score(L, P, average="macro")
    except ValueError:
        mAP = 0.0

    f1 = f1_score(L, (P >= thresholds).astype(int), average="macro")

    return auc, f1, mAP, P, L
