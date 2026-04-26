import warnings
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from config import log_process
from utils import CHESTMNIST_CLASS_NAMES, RADLEX_PATHOLOGIES, ensure_radlex_embeddings, expected_calibration_error
from model import get_evidential_metrics
import logging

class DeepEnsembleTTAEvaluator:
    def __init__(self, model_class, checkpoint_paths, device_, adj_norm_np=None, radlex_path="radlex_embeddings_14.pth", num_mc_passes=10):
        self.device = device_
        self.T = num_mc_passes
        self.models = []
        radlex = ensure_radlex_embeddings(path=radlex_path, pathologies=RADLEX_PATHOLOGIES, model_name="microsoft/BiomedVLP-BioViL-T", device_=device_)
        
        log_process("ensemble", "initialization_started", members=len(checkpoint_paths), mc_passes=self.T)

        for ckpt in checkpoint_paths:
            m = model_class(num_classes=14, cxr_dim=1376, feat_dim=384).to(device_)
            if adj_norm_np is not None: 
                m.set_adjacency_mask(adj_norm_np)
            m.set_radlex_embeddings(radlex.detach())
            raw_sd = torch.load(ckpt, map_location=device_, weights_only=True)
            clean = {k.replace("module.", ""): v for k, v in raw_sd.items() if k != "n_averaged"}
            m.load_state_dict(clean, strict=False)
            self.models.append(m)
            
        log_process("ensemble", "initialization_completed", members=len(self.models), mc_passes=self.T)

    @torch.no_grad()
    def evaluate(self, loader, thresholds=None):
        all_preds, all_mc_vars, all_lbls = [], [], []
        for feats, lbls in tqdm(loader, desc="  Bayesian Inference", leave=False):
            feats = feats.to(self.device)
            batch_model_means, batch_model_vars = [], []
            for m in self.models:
                m.eval()
                for module in m.modules():
                    if isinstance(module, nn.Dropout): module.train()
                
                mc_passes = []
                for _ in range(self.T):
                    out = m(feats)
                    logits = out[0] if isinstance(out, tuple) else out
                    prob, _, _ = get_evidential_metrics(logits)
                    mc_passes.append(prob.cpu().numpy())
                
                mc_arr = np.stack(mc_passes, axis=0)
                batch_model_means.append(mc_arr.mean(axis=0))
                batch_model_vars.append(mc_arr.var(axis=0, ddof=1))
            
            all_preds.append(np.stack(batch_model_means, axis=0))
            all_mc_vars.append(np.stack(batch_model_vars, axis=0))
            all_lbls.append(lbls.numpy())

        ens = np.concatenate(all_preds, axis=1)
        mc_vars_all = np.concatenate(all_mc_vars, axis=1)
        lbls = np.concatenate(all_lbls, axis=0)
        
        predictive_mean = ens.mean(axis=0)
        epistemic_var = ens.var(axis=0) 
        aleatoric_var = mc_vars_all.mean(axis=0) 
        total_var = epistemic_var + aleatoric_var

        if thresholds is None:
            thresholds = np.full(lbls.shape[1], 0.5)
        
        preds_binary = (predictive_mean >= thresholds).astype(int)
        
        try: 
            auc = roc_auc_score(lbls, predictive_mean, average="macro")
        except ValueError: 
            auc = float("nan")
            
        f1 = f1_score(lbls, preds_binary, average="macro")
        ece = expected_calibration_error(predictive_mean, lbls)

        log_process("ensemble", "evaluation_completed", auc=f"{auc:.4f}", f1_macro=f"{f1:.4f}", 
                    ece_mean=f"{ece.mean():.4f}", total_var_mean=f"{total_var.mean():.6f}")

        return {
            "auc": auc, "f1": f1, "ece": ece, "predictive_mean": predictive_mean,
            "epistemic_variance": epistemic_var, "aleatoric_variance": aleatoric_var,
            "total_var": total_var, "labels": lbls, "per_class_ece": ece,
        }


class EvidentialEvaluator:
    def __init__(self, model, device_):
        self.model = model
        self.device = device_
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, loader, thresholds=None):
        all_probs, all_epistemic, all_lbls = [], [], []
        for feats, lbls in tqdm(loader, desc="  Evidential Inference", leave=False):
            logits = self.model(feats.to(self.device))
            if isinstance(logits, tuple): logits = logits[0]
            prob, epistemic, _ = get_evidential_metrics(logits)
            
            all_probs.append(prob.cpu().numpy())
            all_epistemic.append(epistemic.cpu().numpy())
            all_lbls.append(lbls.numpy())
        
        probs = np.concatenate(all_probs, axis=0)
        epistemic = np.concatenate(all_epistemic, axis=0)
        lbls = np.concatenate(all_lbls, axis=0)

        if thresholds is None: thresholds = np.full(lbls.shape[1], 0.5)
        
        preds_binary = (probs >= thresholds).astype(int)
        try: auc = roc_auc_score(lbls, probs, average="macro")
        except ValueError: auc = float("nan")
            
        f1 = f1_score(lbls, preds_binary, average="macro")
        ece = expected_calibration_error(probs, lbls)

        return {
            "auc": auc, "f1": f1, "ece": ece, "predictive_mean": probs,
            "epistemic_variance": epistemic, "labels": lbls, "per_class_ece": ece
        }


def validate(model, loader, device_, thresholds=None):
    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for feats, lbls in tqdm(loader, desc="  Val", leave=False):
            logits = model(feats.to(device_))
            if isinstance(logits, tuple): logits = logits[0]
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
        mAP = average_precision_score(L, P, average="macro")
    except ValueError:
        mAP = 0.0
        
    f1 = f1_score(L, (P >= thresholds).astype(int), average="macro")
    
    return auc, f1, mAP, P, L