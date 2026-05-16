import os
import gc
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from config import DEVICE, EXPERIMENT_NAME, EXPERIMENT_ID
from data import get_dataloaders
from utils import (
    select_adjacency_threshold, build_cooccurrence_adjacency,
    bootstrap_metric_ci, paired_bootstrap_metric_test, ensure_radlex_embeddings, 
    RADLEX_PATHOLOGIES, configure_nature_plots, optimise_thresholds, 
    compute_logit_adjustment, MultiLabelConformalPredictor, CHESTMNIST_CLASS_NAMES
)
from model import CXR_Synapse_Foundation
from evaluators import validate, DeepEnsembleTTAEvaluator
from train import train_ensemble

# Advanced Visualizer Imports
from visualizer import (
    plot_scientific_results, 
    plot_conformal_tradeoff, 
    plot_semantic_manifold, 
    plot_paq_attention_evidence
)

def safe_macro_auc(y_true, y_pred):
    """Calculates macro AUC safely for sparse labels in 10% subsets."""
    aucs = []
    for i in range(y_true.shape[1]):
        try:
            if len(np.unique(y_true[:, i])) > 1:
                aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
        except ValueError:
            pass 
    return np.mean(aucs) if aucs else 0.5

def main():
    # Set global scientific plotting defaults
    configure_nature_plots()
    
    print(f"\n{'=' * 75}\n  Starting {EXPERIMENT_NAME} \n  RUN ID: {EXPERIMENT_ID}\n{'=' * 75}")

    # =======================================================
    # PHASE 1: DATA PREPARATION & TRAINING
    # =======================================================
    print("[*] Loading Training DataLoaders...")
    train_emb_dataset, train_loader, val_loader, _, num_workers = get_dataloaders()

    _train_labels_np = train_emb_dataset.labels.numpy().astype(np.int32)
    _opt_thresh = select_adjacency_threshold(_train_labels_np, num_classes=14)
    adj_norm = build_cooccurrence_adjacency(_train_labels_np, 14, _opt_thresh, True)

    ENSEMBLE_SEEDS = [42, 43, 44] 
    ensemble_checkpoints = train_ensemble(
        ENSEMBLE_SEEDS, adj_norm, train_emb_dataset, train_loader, val_loader, num_workers
    )

    # Clean up training objects to free GPU memory for deep inference
    del train_loader, val_loader, train_emb_dataset
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # =======================================================
    # PHASE 2: EVALUATION SETUP
    # =======================================================
    print("\n[*] Refreshing DataLoaders for Evaluation...")
    _, _, eval_val_loader, eval_test_loader, _ = get_dataloaders()

    print("[*] Loading Final Pro Model for Calibration...")
    pro_model = CXR_Synapse_Foundation(num_classes=14).to(DEVICE)
    pro_model.set_adjacency_mask(adj_norm)
    radlex = ensure_radlex_embeddings("radlex_embeddings_14.pth", RADLEX_PATHOLOGIES, "microsoft/BiomedVLP-BioViL-T", DEVICE)
    pro_model.set_radlex_embeddings(radlex)

    logit_adj_vec = compute_logit_adjustment(_train_labels_np, tau=1.0).to(DEVICE)
    pro_model.set_logit_prior(logit_adj_vec.cpu().numpy())

    # Load weights from the last ensemble member for baseline checks
    _raw = torch.load(ensemble_checkpoints[-1], map_location=DEVICE, weights_only=True)
    _clean = {k.replace("module.", ""): v for k, v in _raw.items() if k != "n_averaged"}
    pro_model.load_state_dict(_clean, strict=False)
    pro_model.eval()

    # =======================================================
    # PHASE 3: METRICS & CONFORMAL CALIBRATION
    # =======================================================
    print("[*] Optimizing Class Thresholds on Validation Set...")
    _, _, _, val_probs, val_labels, _ = validate(pro_model, eval_val_loader, DEVICE)
    opt_thresholds = optimise_thresholds(val_probs, val_labels)

    print("[*] Single Model Baseline Evaluation...")
    base_auc, base_f1, _, base_preds, base_labels, _ = validate(pro_model, eval_test_loader, DEVICE)

    print("[*] Bayesian Ensemble Evaluation (TTA + MC-Dropout)...")
    ensemble_evaluator = DeepEnsembleTTAEvaluator(
        model_class=CXR_Synapse_Foundation, checkpoint_paths=ensemble_checkpoints, 
        device_=DEVICE, adj_norm_np=adj_norm, num_mc_passes=10, logit_adj=logit_adj_vec
    )
    ensemble_results = ensemble_evaluator.evaluate(eval_test_loader, thresholds=opt_thresholds)
    
    test_preds = ensemble_results["predictive_mean"]
    test_labels = ensemble_results["labels"]
    test_epistemic = ensemble_results["epistemic_variance"]

    print("[*] Bootstrapping Confidence Intervals (N=2000)...")
    ens_ci = bootstrap_metric_ci(safe_macro_auc, test_labels, test_preds)
    sig = paired_bootstrap_metric_test(safe_macro_auc, test_labels, test_preds, base_preds)

    print("\n[*] Calibrating Multi-Label Conformal Predictor (90% Marginal Guarantee)...")
    conformal_predictor = MultiLabelConformalPredictor(alpha=0.10)
    conformal_predictor.calibrate(val_probs, val_labels, opt_thresholds)
    
    test_pred_sets = conformal_predictor.predict_sets(test_preds)["include_pos"]
    test_true = test_labels.astype(bool)

    # Calculate rigorous coverage metrics
    per_class_cov = ((test_pred_sets & test_true).sum(axis=0) / np.maximum(test_true.sum(axis=0), 1))
    marginal_cov = per_class_cov.mean()
    avg_size = test_pred_sets.sum(axis=1).mean()

    # =======================================================
    # PHASE 4: ADVANCED SCIENTIFIC VISUALIZATION
    # =======================================================
    print("\n[*] Generating Nature-Style Publication Graphics Suite...")
    
    # 1. The Master 15-Panel Diagnostic Suite
    plot_scientific_results(test_labels, test_preds, test_pred_sets, test_epistemic, CHESTMNIST_CLASS_NAMES, EXPERIMENT_ID)

    # 2. Conformal Safety-Utility Trade-off Curve
    plot_conformal_tradeoff(val_probs, val_labels, opt_thresholds, EXPERIMENT_ID)

    # 3. LISA Semantic Manifold (t-SNE)
    print("[*] Collecting Latent Features for Manifold Visualization...")
    test_features_list = []
    with torch.no_grad():
        for feats, _ in eval_test_loader:
            B = feats.shape[0]
            # Shape goes from (B, 8, 8, 1376) -> (B, 64, 1376)
            feats_flat = feats.view(B, -1, feats.shape[-1]).to(DEVICE)
            
            # Pass through dimension reduction to get the LISA-constrained space
            proj = pro_model.dim_reduction(feats_flat) # Output shape: (B, 64, 384)
            
            # Pool across the 64 spatial patches to get a single vector per image (B, 384)
            pooled_feats = proj.mean(dim=1).cpu().numpy()
            test_features_list.append(pooled_feats)
            
    test_embeddings = np.concatenate(test_features_list, axis=0)
    # Passed safely as a (N, 384) 2D array, TSNE will not throw an error.
    plot_semantic_manifold(test_embeddings, test_labels, CHESTMNIST_CLASS_NAMES, EXPERIMENT_ID)

    # =======================================================
    # PHASE 5: REPORTING & PERSISTENCE
    # =======================================================
    print("\n[*] PATHOLOGY PERFORMANCE REPORT (Detailed Audit):")
    print("-" * 65)
    print(f"{'Pathology':<25} | {'AUC':<10} | {'Threshold':<10}")
    print("-" * 65)

    for i, name in enumerate(CHESTMNIST_CLASS_NAMES):
        try:
            if len(np.unique(test_labels[:, i])) > 1:
                c_auc = roc_auc_score(test_labels[:, i], test_preds[:, i])
                c_auc_str = f"{c_auc:.4f}"
            else:
                c_auc_str = "Sparse    "
        except ValueError:
            c_auc_str = "NaN       "
        print(f"{name:<25} | {c_auc_str}     | {opt_thresholds[i]:.4f}")

    print("-" * 65)
    print(f"Mean Predictive Uncertainty (Epistemic): {test_epistemic.mean():.6f}")
    print("-" * 65)

    final_summary_df = pd.DataFrame({
        "Metric":["Macro AUC", "AUC 95% CI", "ΔAUC p-value", "Macro F1", "Mean ECE", "Conformal Coverage", "Avg Set Size"],
        "Value": [f"{ensemble_results['auc']:.4f}", f"[{ens_ci['ci_low']:.4f}, {ens_ci['ci_high']:.4f}]",
                  f"{sig['p_value']:.4g}", f"{ensemble_results['f1']:.4f}", f"{ensemble_results['per_class_ece'].mean():.4f}",
                  f"{marginal_cov:.1%}", f"{avg_size:.2f}"]
    })

    print("\n" + "=" * 75 + "\n  FINAL SCIENTIFIC SUMMARY — CXR-SYNAPSE\n" + "=" * 75)
    print(final_summary_df.to_string(index=False))
    print("=" * 75)

    # Save final synchronized weights
    save_path = f"CXR_Synapse_Foundation_final_{EXPERIMENT_ID}.pth"
    torch.save(pro_model.state_dict(), save_path)
    print(f"\n[✓] Process completed. Model: {save_path}")
    print(f"[✓] Figures saved with ID: {EXPERIMENT_ID}")

if __name__ == "__main__":
    main()