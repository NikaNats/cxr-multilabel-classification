import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import logging
from config import DEVICE, log_process, EXPERIMENT_NAME, EXPERIMENT_ID
from data import get_dataloaders, get_raw_datasets
from audit import execute_forensic_audit
from utils import (
    select_adjacency_threshold, build_cooccurrence_adjacency, 
    TemperatureScaler, optimise_thresholds, bootstrap_metric_ci, 
    paired_bootstrap_metric_test, expected_calibration_error, 
    brier_score_multilabel, ensure_radlex_embeddings, 
    RADLEX_PATHOLOGIES, CHESTMNIST_CLASS_NAMES, configure_nature_plots
)
from model import CXR_Synapse_Foundation
from evaluators import validate, EvidentialEvaluator, DeepEnsembleTTAEvaluator
from train import train_ensemble
from sklearn.metrics import roc_auc_score, f1_score
from torchcp.classification.score import EntmaxScore
from torchcp.classification.predictor import ClassConditionalPredictor
import gc

def main():
    configure_nature_plots()
    print(f"Starting {EXPERIMENT_NAME} (ID: {EXPERIMENT_ID})")
    
    # 0. Forensic Dataset Audit (Raw Images)
    print("\n[*] Loading RAW images for Forensic Audit...")
    train_raw, val_raw, test_raw, class_names_raw = get_raw_datasets()
    execute_forensic_audit(train_raw, val_raw, test_raw)
    
    # Free memory after audit
    del train_raw, val_raw, test_raw
    gc.collect()

    # 1. Data Preparation (Embeddings for Training)
    print("\n[*] Loading Pre-extracted Embeddings for Training...")
    train_emb_dataset, train_loader, val_loader, test_loader, num_workers = get_dataloaders()
    
    # 2. Build Adjacency Matrix
    _train_labels_np = train_emb_dataset.labels.numpy().astype(np.int32)
    _optimal_threshold = select_adjacency_threshold(_train_labels_np, num_classes=14)
    adj_norm = build_cooccurrence_adjacency(_train_labels_np, num_classes=14, threshold=_optimal_threshold, self_loops=True)
    
    # 3. Training
    ENSEMBLE_SEEDS = [42]
    ensemble_checkpoints = train_ensemble(ENSEMBLE_SEEDS, adj_norm, train_emb_dataset, train_loader, val_loader, num_workers)
    
    # 4. Load Trained Model
    pro_model = CXR_Synapse_Foundation(num_classes=14).to(DEVICE)
    pro_model.set_adjacency_mask(adj_norm)
    radlex = ensure_radlex_embeddings(path="radlex_embeddings_14.pth", pathologies=RADLEX_PATHOLOGIES, model_name="microsoft/BiomedVLP-BioViL-T", device_=DEVICE)
    pro_model.set_radlex_embeddings(radlex)
    
    _raw = torch.load(ensemble_checkpoints[-1], map_location=DEVICE, weights_only=True)
    _clean = {k.removeprefix("module."): v for k, v in _raw.items() if k != "n_averaged"}
    pro_model.load_state_dict(_clean, strict=False)
    pro_model.eval()
    log_process("model", "pro_model_loaded", checkpoint=ensemble_checkpoints[-1])

    # 5. Evaluation
    # [A] Single-model baseline
    print("[*] Single-model baseline evaluation...")
    base_auc, base_f1, base_mAP, base_preds, base_labels = validate(pro_model, test_loader, DEVICE)
    print(f"  AUC: {base_auc:.4f}  |  F1: {base_f1:.4f}  |  mAP: {base_mAP:.4f}")
    
    # [B] Temperature Scaling
    print("\n[*] Post-hoc temperature scaling on validation set...")
    ts_model = TemperatureScaler(pro_model).to(DEVICE)
    T_opt = ts_model.calibrate(val_loader, DEVICE)
    ts_auc, ts_f1, ts_mAP, ts_preds, ts_labels = validate(ts_model, test_loader, DEVICE)
    ts_ece = expected_calibration_error(ts_preds, ts_labels)
    print(f"  Post-calibration AUC: {ts_auc:.4f}  |  F1: {ts_f1:.4f}  |  mAP: {ts_mAP:.4f}  |  ECE: {ts_ece.mean():.4f}")

    # [C] Binary threshold optimization
    print("[*] Optimising binary F1 threshold on validation set...")
    _, _, _, val_base_preds, val_base_labels = validate(pro_model, val_loader, DEVICE)
    val_bin_y = (val_base_labels.sum(1) > 0).astype(int)
    val_bin_p = val_base_preds.max(1)
    _bin_grid = np.linspace(0.001, 0.5, 200)
    _val_f1s = [f1_score(val_bin_y, (val_bin_p >= t).astype(int), zero_division=0) for t in _bin_grid]
    bin_thr = float(_bin_grid[int(np.argmax(_val_f1s))])
    
    true_bin = (base_labels.sum(1) > 0).astype(int)
    pred_bin_p = base_preds.max(1)
    f1_bin = f1_score(true_bin, (pred_bin_p >= bin_thr).astype(int), zero_division=0)
    print(f"  Binary threshold (val-opt): {bin_thr:.3f}  |  AUC: {roc_auc_score(true_bin, pred_bin_p):.4f}  |  F1: {f1_bin:.4f}")

    # [D] Evidential Evaluator
    print("\n[*] Initialising EvidentialEvaluator...")
    evidential_evaluator = EvidentialEvaluator(model=pro_model, device_=DEVICE)
    ensemble_results = evidential_evaluator.evaluate(test_loader, class_names=CHESTMNIST_CLASS_NAMES)
    test_preds, test_labels = ensemble_results["predictive_mean"], ensemble_results["labels"]

    # Bootstrap
    def _macro_auc(yt, ys): return roc_auc_score(yt, ys, average="macro")
    base_ci = bootstrap_metric_ci(_macro_auc, base_labels, base_preds)
    ens_ci = bootstrap_metric_ci(_macro_auc, test_labels, test_preds)
    sig = paired_bootstrap_metric_test(_macro_auc, test_labels, test_preds, base_preds)
    print(f"  Single AUC 95% CI   : [{base_ci['ci_low']:.4f}, {base_ci['ci_high']:.4f}]")
    print(f"  Evidential AUC 95% CI : [{ens_ci['ci_low']:.4f}, {ens_ci['ci_high']:.4f}]")
    print(f"  ΔAUC (Evidential−Single) : {sig['delta']:.4f} [{sig['ci_low']:.4f}, {sig['ci_high']:.4f}] p={sig['p_value']:.4g}")

    # [D.1] Per-Class AUC
    print(f"\n{'='*70}\n  TABLE — Per-Class AUC (Evidential single-pass)\n{'='*70}")
    for k in range(14):
        ak = roc_auc_score(test_labels[:, k], test_preds[:, k])
        print(f"  {CHESTMNIST_CLASS_NAMES[k]:<25}: {ak:.4f}")

    # [D.2] Uncertainty Decomposition Table
    print(f"\n{'='*70}\n  TABLE — Evidential Uncertainty & Calibration (Per-Class)\n{'='*70}")
    unc_df = pd.DataFrame({
        "Class": CHESTMNIST_CLASS_NAMES,
        "Epistemic Var": [f"{v:.6f}" for v in ensemble_results["epistemic_variance"].mean(0)],
        "Aleatoric Var": [f"{v:.6f}" for v in ensemble_results["aleatoric_variance"].mean(0)],
        "ECE": [f"{e:.4f}" for e in ensemble_results["per_class_ece"]]
    })
    print(unc_df.to_string(index=False))

    # [D.3] Detailed Calibration Comparison Table
    print(f"\n{'='*70}\n  TABLE — Calibration Comparison (Per-Class ECE)\n{'='*70}")
    ece_single = expected_calibration_error(base_preds, base_labels)
    calib_df = pd.DataFrame({
        "Class": CHESTMNIST_CLASS_NAMES,
        "ECE (Single)": [f"{e:.4f}" for e in ece_single],
        "ECE (Single+TS)": [f"{e:.4f}" for e in ts_ece],
        "ECE (Evidential)": [f"{e:.4f}" for e in ensemble_results["per_class_ece"]],
        "Brier (Evid)": [f"{b:.4f}" for b in brier_score_multilabel(test_preds, test_labels)]
    })
    print(calib_df.to_string(index=False))

    # [E] TorchCP Conformal Prediction
    print("\n[*] Calibrating Conformal Predictor via TorchCP...")
    score_fn = EntmaxScore(gamma=1.5)
    class _TorchCPLogitAdapter(nn.Module):
        def __init__(self, base_model): super().__init__(); self.base_model = base_model
        def forward(self, x): out = self.base_model(x); return out[0] if isinstance(out, tuple) else out
    torchcp_model = _TorchCPLogitAdapter(pro_model).to(DEVICE)
    torchcp_predictor = ClassConditionalPredictor(score_function=score_fn, model=torchcp_model, alpha=0.10)
    
    def _collect_logits_and_labels(loader, model, device_):
        all_logits, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device_)
                logits = model(xb)
                all_logits.append(logits.detach())
                all_labels.append(yb.to(logits.device))
        return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)

    val_logits_t, val_labels_t = _collect_logits_and_labels(val_loader, torchcp_model, DEVICE)
    pos_pairs = (val_labels_t > 0.5).nonzero(as_tuple=False)
    torchcp_predictor.calculate_threshold(val_logits_t[pos_pairs[:, 0]], pos_pairs[:, 1].long())

    print("[*] Evaluating TorchCP Conformal Predictor on test set...")
    test_logits_t, test_labels_t = _collect_logits_and_labels(test_loader, torchcp_model, DEVICE)
    test_sets_raw = torchcp_predictor.predict_with_logits(test_logits_t)
    
    # Convert to boolean matrix
    test_pred_sets = np.zeros(test_labels_t.shape, dtype=bool)
    for i, idxs in enumerate(test_sets_raw):
        test_pred_sets[i, np.asarray(idxs, dtype=int)] = True
    test_true = test_labels_t.cpu().numpy().astype(bool)

    # Metrics
    per_class_cov = ((test_pred_sets & test_true).sum(0) / np.maximum(test_true.sum(0), 1))
    joint_cov = ((~test_true) | test_pred_sets).all(axis=1).mean()
    avg_size = test_pred_sets.sum(axis=1).mean()
    avg_size_per_label = test_pred_sets.mean()

    print(f"  Target coverage: 90.0% | Empirical mean: {per_class_cov.mean():.1%}")
    print(f"  Joint coverage: {joint_cov:.1%} | Avg set size: {avg_size:.2f}")

    # [F] Final Results Summary
    results_df = pd.DataFrame({
        "Metric": [
            "Multi-label AUC — Single model",
            "Multi-label AUC — Evidential",
            "AUC 95% CI — Evidential",
            "ΔAUC p-value (paired bootstrap)",
            "Binary AUC (Disease vs Normal)",
            "Binary F1 (val-opt τ)",
            "Post-hoc Temperature T",
            "Mean ECE — Single model",
            "Mean ECE — Single + TS",
            "Mean Brier Score — Evidential",
            "Conformal Mean Per-Class Coverage",
            "Avg Prediction Set Size (per-sample)",
            "Avg Prediction Set Size (per-label)"
        ],
        "Value": [
            f"{base_auc:.4f}",
            f"{ensemble_results['auc']:.4f}",
            f"[{ens_ci['ci_low']:.4f}, {ens_ci['ci_high']:.4f}]",
            f"{sig['p_value']:.4g}",
            f"{roc_auc_score(true_bin, pred_bin_p):.4f}",
            f"{f1_bin:.4f}",
            f"{T_opt:.4f}",
            f"{expected_calibration_error(base_preds, base_labels).mean():.4f}",
            f"{ts_ece.mean():.4f}",
            f"{brier_score_multilabel(test_preds, test_labels).mean():.4f}",
            f"{per_class_cov.mean():.1%}",
            f"{avg_size:.2f}",
            f"{avg_size_per_label:.2f}"
        ]
    })
    print("\n" + "="*80 + "\n  FINAL RESULTS SUMMARY\n" + "="*80)
    print(results_df.to_string(index=False))

    # [G] Publication Checklist
    print(f"\n{'='*70}\n  PUBLICATION CHECKLIST\n{'='*70}")
    checklist = [
        "Visual backbone: Google ELIXR-C v2 (frozen)",
        "GLoRI-Lite pooler: Correct GeM (Fix 2)",
        "Data-driven adjacency threshold (Fix 1)",
        "Evidential Deep Learning uncertainty (single-pass)",
        "Conformal prediction (TorchCP, α=0.10)",
        "Bootstrap 95% CI + paired significance test"
    ]
    for item in checklist:
        print(f"  [✓] {item}")

    # Save
    torch.save(pro_model.state_dict(), "CXR_Synapse_Foundation_final.pth")
    print("\n✓ Process completed successfully.")

if __name__ == "__main__":
    main()
