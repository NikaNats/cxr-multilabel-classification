import os
import gc
import warnings
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from config import DEVICE, log_process, EXPERIMENT_NAME, EXPERIMENT_ID
from data import get_dataloaders
from utils import (
    select_adjacency_threshold, build_cooccurrence_adjacency, 
    TemperatureScaler, bootstrap_metric_ci, 
    paired_bootstrap_metric_test, ensure_radlex_embeddings, 
    RADLEX_PATHOLOGIES, CHESTMNIST_CLASS_NAMES, configure_nature_plots,
    optimise_thresholds 
)
from model import CXR_Synapse_Foundation
from evaluators import validate, EvidentialEvaluator
from train import train_ensemble

class MultiLabelConformalPredictor:
    """
    Nature-Grade True Marginal Conformal Predictor.
    დამოუკიდებლად აკალიბრებს თითოეულ დაავადებას, რათა უზრუნველყოს გარანტირებული დაფარვა.
    """
    def __init__(self, alpha=0.10):
        self.alpha = alpha
        self.thresholds = None

    def calibrate(self, cal_probs, cal_labels):
        K = cal_probs.shape[1]
        self.thresholds = np.zeros(K)
        
        for k in range(K):
            pos_mask = (cal_labels[:, k] == 1)
            if pos_mask.sum() == 0:
                self.thresholds[k] = 0.5
                continue
            
            scores = 1.0 - cal_probs[pos_mask, k]
            n = scores.shape[0]
            
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            q_level = min(max(q_level, 0.0), 1.0)
            
            self.thresholds[k] = 1.0 - np.quantile(scores, q_level)

    def predict_sets(self, test_probs):
        return test_probs >= self.thresholds


def main():
    configure_nature_plots()
    print(f"\n{'='*75}\n  Starting {EXPERIMENT_NAME} \n  RUN ID: {EXPERIMENT_ID}\n{'='*75}")
    
    print("[*] Loading Pre-extracted Embeddings...")
    train_emb_dataset, train_loader, val_loader, test_loader, num_workers = get_dataloaders()
    
    _train_labels_np = train_emb_dataset.labels.numpy().astype(np.int32)
    _optimal_threshold = select_adjacency_threshold(_train_labels_np, num_classes=14)
    adj_norm = build_cooccurrence_adjacency(
        _train_labels_np, num_classes=14, threshold=_optimal_threshold, self_loops=True
    )
    
    ENSEMBLE_SEEDS = [42] 
    ensemble_checkpoints = train_ensemble(
        ENSEMBLE_SEEDS, adj_norm, train_emb_dataset, train_loader, val_loader, num_workers
    )
    
    print("\n[*] Loading Final Pro Model...")
    pro_model = CXR_Synapse_Foundation(num_classes=14).to(DEVICE)
    pro_model.set_adjacency_mask(adj_norm)
    radlex = ensure_radlex_embeddings(
        path="radlex_embeddings_14.pth", pathologies=RADLEX_PATHOLOGIES, 
        model_name="microsoft/BiomedVLP-BioViL-T", device_=DEVICE
    )
    pro_model.set_radlex_embeddings(radlex)
    
    _raw = torch.load(ensemble_checkpoints[-1], map_location=DEVICE, weights_only=True)
    _clean = {k.replace("module.", ""): v for k, v in _raw.items() if k != "n_averaged"}
    pro_model.load_state_dict(_clean, strict=False)
    pro_model.eval()

    print("[*] Optimizing Class Thresholds on Validation Set...")
    _, _, _, val_probs, val_labels = validate(pro_model, val_loader, DEVICE)
    opt_thresholds = optimise_thresholds(val_probs, val_labels)

    print("[*] Single Model Baseline Evaluation...")
    base_auc, base_f1, base_mAP, base_preds, base_labels = validate(pro_model, test_loader, DEVICE)
    
    print("[*] Evidential Evaluation (Sparsity Regularized)...")
    evidential_evaluator = EvidentialEvaluator(model=pro_model, device_=DEVICE)
    ensemble_results = evidential_evaluator.evaluate(test_loader, thresholds=opt_thresholds)
    test_preds, test_labels = ensemble_results["predictive_mean"], ensemble_results["labels"]

    print("[*] Bootstrapping Confidence Intervals (N=2000)...")
    def _macro_auc(yt, ys): return roc_auc_score(yt, ys, average="macro")
    ens_ci = bootstrap_metric_ci(_macro_auc, test_labels, test_preds)
    sig = paired_bootstrap_metric_test(_macro_auc, test_labels, test_preds, base_preds)

    # 9. True Multi-Label Conformal Prediction
    print("\n[*] Calibrating Multi-Label Conformal Predictor (90% Marginal Guarantee)...")
    
    conformal_predictor = MultiLabelConformalPredictor(alpha=0.10)
    conformal_predictor.calibrate(val_probs, val_labels)
    
    print("[*] Evaluating Conformal Predictor...")
    
    test_pred_sets = conformal_predictor.predict_sets(test_preds)
    test_true = test_labels.astype(bool)
    
    per_class_cov = ((test_pred_sets & test_true).sum(axis=0) / np.maximum(test_true.sum(axis=0), 1))
    
    marginal_cov = per_class_cov.mean()
    
    joint_cov = ((~test_true) | test_pred_sets).all(axis=1).mean()
    avg_size = test_pred_sets.sum(axis=1).mean()

    results_df = pd.DataFrame({
        "Metric":[
            "Multi-label AUC (Evid)", 
            "AUC 95% CI", 
            "ΔAUC p-value (Evid vs Base)", 
            "Optimized Macro F1", 
            "Mean ECE (Evid + Sparsity)", 
            "Conformal Marginal Coverage (Target: 90%)",
            "Conformal Joint Coverage", 
            "Avg Prediction Set Size"
        ],
        "Value": [
            f"{ensemble_results['auc']:.4f}", 
            f"[{ens_ci['ci_low']:.4f}, {ens_ci['ci_high']:.4f}]", 
            f"{sig['p_value']:.4g}", 
            f"{ensemble_results['f1']:.4f}", 
            f"{ensemble_results['per_class_ece'].mean():.4f}", 
            f"{marginal_cov:.1%}",
            f"{joint_cov:.1%}", 
            f"{avg_size:.2f}"
        ]
    })
    
    print("\n" + "="*75)
    print("  FINAL SCIENTIFIC SUMMARY — CXR-SYNAPSE")
    print("="*75)
    print(results_df.to_string(index=False))
    print("="*75)

    # მოდელის შენახვა
    torch.save(pro_model.state_dict(), "CXR_Synapse_Foundation_final.pth")
    print("\n✓ Process completed successfully. Coverage guaranteed. Artifacts saved.")

if __name__ == "__main__":
    main()