"""
main.py — CXR-Synapse Training & Evaluation Pipeline
═══════════════════════════════════════════════════════════════════════════════
Orchestrates five sequential phases:
  1. Data preparation and ensemble training.
  2. Evaluation setup (model loading, calibration, logit adjustment).
  3. Metrics computation and conformal prediction calibration.
  4. Publication-quality figure generation.
  5. Per-pathology performance reporting and model persistence.
"""

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
from evaluators import DeepEnsembleTTAEvaluator, validate
from model import CXR_Synapse_Foundation
from train import train_ensemble
from utils import (
    CHESTMNIST_CLASS_NAMES,
    RADLEX_PATHOLOGIES,
    MultiLabelConformalPredictor,
    bootstrap_metric_ci,
    build_cooccurrence_adjacency,
    compute_logit_adjustment,
    ensure_radlex_embeddings,
    optimise_thresholds,
    paired_bootstrap_metric_test,
    select_adjacency_threshold,
)
from visualizer import (
    configure_nature_style,       # replaces configure_nature_plots from utils
    plot_conformal_tradeoff,
    plot_diagnostic_suite,        # replaces plot_scientific_results
    plot_paq_attention,           # replaces plot_paq_attention_evidence
    plot_semantic_manifold,
)

# Output directory for all generated figures
FIGURE_DIR = f"figures_{EXPERIMENT_ID}"


def _safe_macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Macro-averaged AUROC that skips label-sparse classes.

    Returns 0.5 (chance) if no class has both positive and negative examples,
    which can occur in small bootstrap subsets or rare-pathology slices.
    """
    per_class_aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            try:
                per_class_aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
            except ValueError:
                pass
    return float(np.mean(per_class_aucs)) if per_class_aucs else 0.5


def main() -> None:
    configure_nature_style()   # apply Nature rcParams before any plotting

    print(
        f"\n{'=' * 75}\n"
        f"  Starting {EXPERIMENT_NAME}\n"
        f"  RUN ID: {EXPERIMENT_ID}\n"
        f"{'=' * 75}"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — DATA PREPARATION & ENSEMBLE TRAINING
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[*] Loading training DataLoaders …")
    train_emb_dataset, train_loader, val_loader, _, num_workers = get_dataloaders()

    train_labels_np = train_emb_dataset.labels.numpy().astype(np.int32)
    adj_threshold   = select_adjacency_threshold(train_labels_np, num_classes=14)
    adj_norm        = build_cooccurrence_adjacency(
        train_labels_np, 14, adj_threshold, True   # positional: normalise=True
    )

    ensemble_checkpoints = train_ensemble(
        [42, 43, 44], adj_norm, train_emb_dataset, train_loader, val_loader, num_workers
    )

    # Free GPU memory before deep inference begins
    del train_loader, val_loader, train_emb_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — EVALUATION SETUP
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[*] Refreshing DataLoaders for evaluation …")
    _, _, eval_val_loader, eval_test_loader, _ = get_dataloaders()

    print("[*] Loading model for calibration …")
    pro_model = CXR_Synapse_Foundation(num_classes=14).to(DEVICE)
    pro_model.set_adjacency_mask(adj_norm)

    radlex_embeddings = ensure_radlex_embeddings(
        "radlex_embeddings_14.pth", RADLEX_PATHOLOGIES,
        "microsoft/BiomedVLP-BioViL-T", DEVICE,
    )
    pro_model.set_radlex_embeddings(radlex_embeddings)

    logit_adj_vec = compute_logit_adjustment(train_labels_np, tau=1.0).to(DEVICE)
    pro_model.set_logit_prior(logit_adj_vec.cpu().numpy())

    # Load the last ensemble checkpoint as a single-model baseline
    raw_state  = torch.load(
        ensemble_checkpoints[-1], map_location=DEVICE, weights_only=True
    )
    clean_state = {
        k.replace("module.", ""): v
        for k, v in raw_state.items()
        if k != "n_averaged"
    }
    pro_model.load_state_dict(clean_state, strict=False)
    pro_model.eval()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3 — METRICS & CONFORMAL CALIBRATION
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[*] Optimising per-class thresholds on validation set …")
    _, _, _, val_probs, val_labels, _ = validate(
        pro_model, eval_val_loader, DEVICE
    )
    opt_thresholds = optimise_thresholds(val_probs, val_labels)

    print("[*] Single-model baseline evaluation …")
    base_auc, base_f1, _, base_preds, base_labels, _ = validate(
        pro_model, eval_test_loader, DEVICE
    )

    print("[*] Bayesian ensemble evaluation (TTA + MC-dropout) …")
    ensemble_results = DeepEnsembleTTAEvaluator(
        model_class=CXR_Synapse_Foundation,
        checkpoint_paths=ensemble_checkpoints,
        device_=DEVICE,
        adj_norm_np=adj_norm,
        num_mc_passes=10,
        logit_adj=logit_adj_vec,
    ).evaluate(eval_test_loader, thresholds=opt_thresholds)

    test_preds     = ensemble_results["predictive_mean"]
    test_labels    = ensemble_results["labels"]
    test_epistemic = ensemble_results["epistemic_variance"]

    print("[*] Bootstrapping 95 % confidence intervals (N = 2 000) …")
    ens_ci = bootstrap_metric_ci(_safe_macro_auc, test_labels, test_preds)
    sig    = paired_bootstrap_metric_test(
        _safe_macro_auc, test_labels, test_preds, base_preds
    )

    print("\n[*] Calibrating multi-label conformal predictor (90 % marginal coverage) …")
    conformal_predictor = MultiLabelConformalPredictor(alpha=0.10)
    conformal_predictor.calibrate(val_probs, val_labels, opt_thresholds)

    conformal_sets  = conformal_predictor.predict_sets(test_preds)["include_pos"]
    true_positives  = test_labels.astype(bool)
    per_class_cover = (
        (conformal_sets & true_positives).sum(axis=0)
        / np.maximum(true_positives.sum(axis=0), 1)
    )
    marginal_coverage = per_class_cover.mean()
    mean_set_size     = conformal_sets.sum(axis=1).mean()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 4 — PUBLICATION FIGURES
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n[*] Generating figures → {FIGURE_DIR}/")

    # 4a. 15-panel diagnostic suite (combined PDF + 15 individual PDFs)
    plot_diagnostic_suite(
        test_labels=test_labels,
        test_preds=test_preds,
        conformal_sets=conformal_sets,
        uncertainty=test_epistemic,
        class_names=CHESTMNIST_CLASS_NAMES,
        experiment_id=EXPERIMENT_ID,
        output_dir=FIGURE_DIR,
    )

    # 4b. Conformal safety–efficiency trade-off curve
    plot_conformal_tradeoff(
        val_probs=val_probs,
        val_labels=val_labels,
        opt_thresholds=opt_thresholds,
        experiment_id=EXPERIMENT_ID,
        output_dir=FIGURE_DIR,
    )

    # 4c. LISA latent space manifold (t-SNE)
    print("[*] Collecting latent features for manifold visualisation …")
    feature_batches: list[np.ndarray] = []
    with torch.no_grad():
        for feats, _ in eval_test_loader:
            B = feats.shape[0]
            # (B, 8, 8, 1376) → (B, 64, 1376) → project → (B, 64, 384)
            proj   = pro_model.dim_reduction(feats.view(B, -1, feats.shape[-1]).to(DEVICE))
            pooled = proj.mean(dim=1).cpu().numpy()   # (B, 384)
            feature_batches.append(pooled)

    plot_semantic_manifold(
        embeddings=np.concatenate(feature_batches, axis=0),
        labels=test_labels,
        class_names=CHESTMNIST_CLASS_NAMES,
        experiment_id=EXPERIMENT_ID,
        output_dir=FIGURE_DIR,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 5 — REPORTING & PERSISTENCE
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[*] PATHOLOGY PERFORMANCE REPORT")
    print("-" * 65)
    print(f"{'Pathology':<25} | {'AUROC':<10} | {'Threshold':<10}")
    print("-" * 65)

    for i, name in enumerate(CHESTMNIST_CLASS_NAMES):
        try:
            if len(np.unique(test_labels[:, i])) > 1:
                auroc_str = f"{roc_auc_score(test_labels[:, i], test_preds[:, i]):.4f}"
            else:
                auroc_str = "Sparse"
        except ValueError:
            auroc_str = "NaN"
        print(f"{name:<25} | {auroc_str:<10} | {opt_thresholds[i]:.4f}")

    print("-" * 65)
    print(f"Mean epistemic uncertainty: {test_epistemic.mean():.6f}")

    summary_df = pd.DataFrame({
        "Metric": [
            "Macro AUROC", "AUROC 95 % CI", "ΔAUROC p-value",
            "Macro F1", "Mean ECE", "Conformal coverage", "Mean set size",
        ],
        "Value": [
            f"{ensemble_results['auc']:.4f}",
            f"[{ens_ci['ci_low']:.4f}, {ens_ci['ci_high']:.4f}]",
            f"{sig['p_value']:.4g}",
            f"{ensemble_results['f1']:.4f}",
            f"{ensemble_results['per_class_ece'].mean():.4f}",
            f"{marginal_coverage:.1%}",
            f"{mean_set_size:.2f}",
        ],
    })

    print(f"\n{'=' * 75}")
    print("  FINAL SCIENTIFIC SUMMARY — CXR-SYNAPSE")
    print("=" * 75)
    print(summary_df.to_string(index=False))
    print("=" * 75)

    # Save final synchronised model weights
    save_path = f"CXR_Synapse_Foundation_final_{EXPERIMENT_ID}.pth"
    torch.save(pro_model.state_dict(), save_path)
    print(f"\n[✓] Model saved  → {save_path}")
    print(f"[✓] Figures saved → {FIGURE_DIR}/")


if __name__ == "__main__":
    main()