import os
import gc
import warnings
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.exceptions import UndefinedMetricWarning

# ── NATURE-GRADE FIX: გავთიშოთ ზედმეტი გაფრთხილებები იშვიათ კლასებზე ──
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from config import DEVICE, log_process, EXPERIMENT_NAME, EXPERIMENT_ID
from data import get_dataloaders
from utils import (
    select_adjacency_threshold, build_cooccurrence_adjacency, 
    TemperatureScaler, bootstrap_metric_ci, 
    paired_bootstrap_metric_test, ensure_radlex_embeddings, 
    RADLEX_PATHOLOGIES, CHESTMNIST_CLASS_NAMES, configure_nature_plots
)
from model import CXR_Synapse_Foundation
from evaluators import validate, EvidentialEvaluator
from train import train_ensemble

from torchcp.classification.score import EntmaxScore
from torchcp.classification.predictor import ClassConditionalPredictor

def main():
    # 0. გარემოს ინიციალიზაცია
    configure_nature_plots()
    print(f"\n{'='*70}\n  Starting {EXPERIMENT_NAME} \n  RUN ID: {EXPERIMENT_ID}\n{'='*70}")
    
    # 1. მონაცემების მომზადება
    print("[*] Loading Pre-extracted Embeddings...")
    train_emb_dataset, train_loader, val_loader, test_loader, num_workers = get_dataloaders()
    
    # 2. დინამიური გრაფის აგება (Nature Fix 1)
    _train_labels_np = train_emb_dataset.labels.numpy().astype(np.int32)
    _optimal_threshold = select_adjacency_threshold(_train_labels_np, num_classes=14)
    adj_norm = build_cooccurrence_adjacency(
        _train_labels_np, num_classes=14, threshold=_optimal_threshold, self_loops=True
    )
    
    # 3. ანსამბლის ტრენინგი (Baseline: M=1)
    ENSEMBLE_SEEDS = [42] 
    ensemble_checkpoints = train_ensemble(
        ENSEMBLE_SEEDS, adj_norm, train_emb_dataset, train_loader, val_loader, num_workers
    )
    
    # 4. საბოლოო მოდელის ჩატვირთვა
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

    # 5. საბაზისო ვალიდაცია და ტემპერატურული სკალირება
    print("[*] Single Model Baseline Evaluation...")
    base_auc, base_f1, base_mAP, base_preds, base_labels = validate(pro_model, test_loader, DEVICE)
    
    print("[*] Post-hoc Temperature Scaling...")
    ts_model = TemperatureScaler(pro_model).to(DEVICE)
    T_opt = ts_model.calibrate(val_loader, DEVICE)
    
    # 6. ევინდენციალური (EDL) შეფასება და ბუტსტრაპი
    print("[*] Evidential Evaluation (Single Pass)...")
    evidential_evaluator = EvidentialEvaluator(model=pro_model, device_=DEVICE)
    ensemble_results = evidential_evaluator.evaluate(test_loader, class_names=CHESTMNIST_CLASS_NAMES)
    test_preds, test_labels = ensemble_results["predictive_mean"], ensemble_results["labels"]

    print("[*] Bootstrapping Confidence Intervals...")
    def _macro_auc(yt, ys): return roc_auc_score(yt, ys, average="macro")
    ens_ci = bootstrap_metric_ci(_macro_auc, test_labels, test_preds)
    sig = paired_bootstrap_metric_test(_macro_auc, test_labels, test_preds, base_preds)

    # 7. TorchCP კონფორმული პროგნოზირება
    print("\n[*] Calibrating Conformal Predictor via TorchCP...")
    score_fn = EntmaxScore(gamma=1.5)
    
    class _TorchCPLogitAdapter(nn.Module):
        def __init__(self, base_model): super().__init__(); self.base_model = base_model
        def forward(self, x): 
            out = self.base_model(x)
            return out[0] if isinstance(out, tuple) else out
    
    torchcp_model = _TorchCPLogitAdapter(pro_model).to(DEVICE)
    torchcp_model.eval()
    torchcp_predictor = ClassConditionalPredictor(score_function=score_fn, model=torchcp_model, alpha=0.10)
    
    def _collect_logits_and_labels(loader, model, device_):
        all_logits, all_labels = [], []
        with torch.no_grad():
            for xb, yb in loader:
                logits = model(xb.to(device_))
                all_logits.append(logits.cpu())
                all_labels.append(yb.cpu())
        return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)

    val_logits_t, val_labels_t = _collect_logits_and_labels(val_loader, torchcp_model, DEVICE)
    pos_pairs = (val_labels_t > 0.5).nonzero(as_tuple=False)
    
    # თავსებადობის ბლოკი ძველი/ახალი TorchCP ვერსიებისთვის
    try:
        torchcp_predictor.calculate_threshold(val_logits_t[pos_pairs[:, 0]], pos_pairs[:, 1].long())
    except TypeError:
        torchcp_predictor.calculate_threshold(val_logits_t[pos_pairs[:, 0]].to(DEVICE), pos_pairs[:, 1].to(DEVICE).long())

    print("[*] Evaluating TorchCP Conformal Predictor...")
    test_logits_t, test_labels_t = _collect_logits_and_labels(test_loader, torchcp_model, DEVICE)
    test_sets_raw = torchcp_predictor.predict_with_logits(test_logits_t)
    
    test_pred_sets = np.zeros(test_labels_t.shape, dtype=bool)
    for i, idxs in enumerate(test_sets_raw):
        if torch.is_tensor(idxs): 
            idxs = idxs.cpu().numpy()
        test_pred_sets[i, np.asarray(idxs, dtype=int)] = True
    
    test_true = test_labels_t.numpy().astype(bool)
    per_class_cov = ((test_pred_sets & test_true).sum(axis=0) / np.maximum(test_true.sum(axis=0), 1))
    joint_cov = ((~test_true) | test_pred_sets).all(axis=1).mean()
    avg_size = test_pred_sets.sum(axis=1).mean()

    # 8. საბოლოო ანგარიში (Final Report)
    results_df = pd.DataFrame({
        "Metric": [
            "Multi-label AUC (Evid)", 
            "AUC 95% CI", 
            "ΔAUC p-value (Evid vs Base)", 
            "Post-hoc T (Calib)", 
            "Mean ECE (Evid)", 
            "Conformal Joint Coverage", 
            "Avg Prediction Set Size"
        ],
        "Value": [
            f"{ensemble_results['auc']:.4f}", 
            f"[{ens_ci['ci_low']:.4f}, {ens_ci['ci_high']:.4f}]", 
            f"{sig['p_value']:.4g}", 
            f"{T_opt:.4f}", 
            f"{ensemble_results['per_class_ece'].mean():.4f}", 
            f"{joint_cov:.1%}", 
            f"{avg_size:.2f}"
        ]
    })
    
    print("\n" + "="*70)
    print("  FINAL RESULTS SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70)

    # მოდელის შენახვა
    torch.save(pro_model.state_dict(), "CXR_Synapse_Foundation_final.pth")
    print("\n✓ Process completed successfully. Artifacts saved.")

if __name__ == "__main__":
    main()