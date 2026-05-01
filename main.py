import os, gc, warnings
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
    compute_logit_adjustment, MultiLabelConformalPredictor
)
from model import CXR_Synapse_Foundation
from evaluators import validate
from train import train_ensemble

def main():
    configure_nature_plots()
    print(f"\n{'=' * 75}\n  Starting {EXPERIMENT_NAME} \n  RUN ID: {EXPERIMENT_ID}\n{'=' * 75}")

    # =======================================================
    # PHASE 1: TRAINING
    # =======================================================
    print("[*] Loading Training DataLoaders...")
    train_emb_dataset, train_loader, val_loader, _, num_workers = get_dataloaders()

    _train_labels_np = train_emb_dataset.labels.numpy().astype(np.int32)
    _opt_thresh = select_adjacency_threshold(_train_labels_np, num_classes=14)
    adj_norm = build_cooccurrence_adjacency(_train_labels_np, 14, _opt_thresh, True)

    ENSEMBLE_SEEDS = [42, 43, 44] # 3 Seeds for robust epistemic capture
    ensemble_checkpoints = train_ensemble(
        ENSEMBLE_SEEDS, adj_norm, train_emb_dataset, train_loader, val_loader, num_workers
    )

    del train_loader, val_loader
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

    _raw = torch.load(ensemble_checkpoints[-1], map_location=DEVICE, weights_only=True)
    _clean = {k.replace("module.", ""): v for k, v in _raw.items() if k != "n_averaged"}
    pro_model.load_state_dict(_clean, strict=False)
    pro_model.eval()

    # =======================================================
    # PHASE 3: METRICS & CONFORMAL
    # =======================================================
    print("[*] Optimizing Class Thresholds on Validation Set...")
    _, _, _, val_probs, val_labels = validate(pro_model, eval_val_loader, DEVICE)
    opt_thresholds = optimise_thresholds(val_probs, val_labels)

    print("[*] Single Model Baseline Evaluation...")
    base_auc, base_f1, _, base_preds, base_labels = validate(pro_model, eval_test_loader, DEVICE)

    print("[*] Bayesian Ensemble Evaluation (TTA + MC-Dropout)...")
    from evaluators import DeepEnsembleTTAEvaluator
    ensemble_evaluator = DeepEnsembleTTAEvaluator(
        model_class=CXR_Synapse_Foundation, checkpoint_paths=ensemble_checkpoints, 
        device_=DEVICE, adj_norm_np=adj_norm, num_mc_passes=10, logit_adj=logit_adj_vec
    )
    ensemble_results = ensemble_evaluator.evaluate(eval_test_loader, thresholds=opt_thresholds)
    test_preds, test_labels = ensemble_results["predictive_mean"], ensemble_results["labels"]

    print("[*] Bootstrapping Confidence Intervals (N=2000)...")
    def _m_auc(yt, ys): return roc_auc_score(yt, ys, average="macro")
    ens_ci = bootstrap_metric_ci(_m_auc, test_labels, test_preds)
    sig = paired_bootstrap_metric_test(_m_auc, test_labels, test_preds, base_preds)

    print("\n[*] Calibrating Multi-Label Conformal Predictor (90% Marginal Guarantee)...")
    conformal_predictor = MultiLabelConformalPredictor(alpha=0.10)
    conformal_predictor.calibrate(val_probs, val_labels, opt_thresholds)
    
    test_pred_sets = conformal_predictor.predict_sets(test_preds)["include_pos"]
    test_true = test_labels.astype(bool)

    per_class_cov = ((test_pred_sets & test_true).sum(axis=0) / np.maximum(test_true.sum(axis=0), 1))
    marginal_cov = per_class_cov.mean()
    avg_size = test_pred_sets.sum(axis=1).mean()

    results_df = pd.DataFrame({
        "Metric":["AUC", "AUC 95% CI", "ΔAUC p-value", "Macro F1", "Mean ECE", "Conformal Coverage", "Set Size"],
        "Value": [f"{ensemble_results['auc']:.4f}", f"[{ens_ci['ci_low']:.4f}, {ens_ci['ci_high']:.4f}]",
                  f"{sig['p_value']:.4g}", f"{ensemble_results['f1']:.4f}", f"{ensemble_results['per_class_ece'].mean():.4f}",
                  f"{marginal_cov:.1%}", f"{avg_size:.2f}"]
    })

    print("\n" + "=" * 75 + "\n  FINAL SCIENTIFIC SUMMARY — CXR-SYNAPSE\n" + "=" * 75)
    print(results_df.to_string(index=False))
    print("=" * 75)

    torch.save(pro_model.state_dict(), "CXR_Synapse_Foundation_final.pth")
    print("\n✓ Process completed successfully.")

if __name__ == "__main__":
    main()