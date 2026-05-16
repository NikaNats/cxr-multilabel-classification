import os
import cv2
import torch
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, roc_auc_score, 
                             precision_recall_curve, average_precision_score)
from sklearn.manifold import TSNE

# Suppress specific warnings for clean plot generation in production
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# GLOBAL AESTHETIC CONFIGURATION (NATURE / IEEE STANDARD)
# =====================================================================
def set_scientific_style():
    sns.set_theme(style="white", palette="colorblind")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'pdf.fonttype': 42, # Ensures true text in PDFs (no outline)
        'ps.fonttype': 42
    })

# Nature Color Palette
C_BLUE = '#0072B2'
C_ORANGE = '#D55E00'
C_GREEN = '#009E73'
C_SKY = '#56B4E9'
C_GRAY = '#7F8C8D'
C_PURPLE = '#CC79A7'
C_RED = '#E31A1C'
C_YELLOW = '#F0E442'

# =====================================================================
# 1. THE ULTIMATE 15-PANEL DIAGNOSTIC SUITE
# =====================================================================
def plot_scientific_results(test_labels, test_preds, conformal_sets, 
                            uncertainty, class_names, experiment_id):
    """
    15-Panel Comprehensive Diagnostic Plot (5x3 Grid).
    Covers Discrimination, Calibration, Uncertainty, Safety, and Data Forensics.
    """
    print("[*] Rendering 15-Panel Nature-Grade Diagnostic Suite...")
    set_scientific_style()
    
    fig, axes = plt.subplots(5, 3, figsize=(24, 32))
    axes = axes.flatten()

    # --- A. MACRO ROC CURVE ---
    ax = axes[0]
    all_fpr = np.linspace(0, 1, 100)
    tprs = []
    for i in range(test_labels.shape[1]):
        if len(np.unique(test_labels[:, i])) > 1:
            fpr, tpr, _ = roc_curve(test_labels[:, i], test_preds[:, i])
            tprs.append(np.interp(all_fpr, fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color=C_BLUE, lw=2.5, label=f'Macro ROC (AUC = {mean_auc:.3f})')
    ax.fill_between(all_fpr, np.percentile(tprs, 25, axis=0), np.percentile(tprs, 75, axis=0), 
                    color=C_SKY, alpha=0.25, label='IQR')
    ax.plot([0, 1], [0, 1], color=C_GRAY, linestyle='--', lw=1.5)
    ax.set_title("A. Multi-label ROC Performance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", frameon=False)

    # --- B. RELIABILITY DIAGRAM (CALIBRATION) ---
    ax = axes[1]
    n_bins = 15
    p_flat, l_flat = test_preds.flatten(), test_labels.flatten()
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    accuracies, confidences = [], []
    
    for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (p_flat > lower) & (p_flat <= upper)
        accuracies.append(np.mean(l_flat[mask]) if np.any(mask) else np.nan)
        confidences.append(np.mean(p_flat[mask]) if np.any(mask) else np.nan)

    ax.bar(bin_boundaries[:-1], accuracies, width=1/n_bins, align='edge', 
           color=C_ORANGE, edgecolor='white', alpha=0.75, label='Empirical Accuracy')
    ax.plot([0, 1], [0, 1], color=C_GRAY, linestyle='--', lw=1.5, label='Perfect Calibration')
    ax.set_title("B. Global Probability Calibration")
    ax.set_xlabel("Predicted Confidence")
    ax.set_ylabel("Empirical Accuracy")
    ax.legend(loc="upper left", frameon=False)

    # --- C. CONFORMAL SET SIZE VS UNCERTAINTY ---
    ax = axes[2]
    set_sizes = conformal_sets.sum(axis=1)
    u_plot, s_plot = (uncertainty[:2000], set_sizes[:2000]) if len(uncertainty) > 2000 else (uncertainty, set_sizes)
    
    sns.regplot(x=u_plot, y=s_plot, ax=ax, scatter_kws={'alpha':0.2, 's':15, 'color':C_GREEN}, 
                line_kws={'color':C_ORANGE, 'lw':2})
    ax.set_title("C. Uncertainty vs Prediction Set Size")
    ax.set_xlabel("Epistemic Uncertainty")
    ax.set_ylabel("Conformal Set Size")

    # --- D. PER-CLASS AUROC (Lollipop) ---
    ax = axes[3]
    class_aucs = {name: roc_auc_score(test_labels[:, i], test_preds[:, i]) 
                  for i, name in enumerate(class_names) if len(np.unique(test_labels[:, i])) > 1}
    
    sorted_classes = sorted(class_aucs.keys(), key=lambda k: class_aucs[k])
    sorted_aucs = [class_aucs[k] for k in sorted_classes]

    y_pos = np.arange(len(sorted_classes))
    ax.hlines(y=y_pos, xmin=0.5, xmax=sorted_aucs, color=C_GRAY, alpha=0.4, lw=1.5)
    ax.scatter(sorted_aucs, y_pos, color=C_GREEN, s=60, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_classes)
    ax.set_xlim(0.5, 1.0)
    ax.set_title("D. Per-Class Discriminative Performance")
    ax.set_xlabel("AUROC")

    # --- E. CONFORMAL SET SIZE DISTRIBUTION ---
    ax = axes[4]
    sns.histplot(set_sizes, discrete=True, color=C_SKY, alpha=0.8, ax=ax, edgecolor="white")
    ax.axvline(np.mean(set_sizes), color=C_ORANGE, linestyle='--', lw=2, label=f'Mean: {np.mean(set_sizes):.2f}')
    ax.set_title("E. Prediction Set Size Distribution")
    ax.set_xlabel("Number of Pathologies")
    ax.set_ylabel("Patient Count")
    ax.legend(frameon=False)

    # --- F. UNCERTAINTY BY PREDICTION ERROR ---
    ax = axes[5]
    patient_mae = np.mean(np.abs(test_preds - test_labels), axis=1)
    high_error = patient_mae > np.median(patient_mae)
    
    sns.kdeplot(uncertainty[~high_error], ax=ax, color=C_BLUE, fill=True, alpha=0.35, label='Low Error Cases')
    sns.kdeplot(uncertainty[high_error], ax=ax, color=C_ORANGE, fill=True, alpha=0.35, label='High Error Cases')
    ax.set_title("F. Epistemic Uncertainty by Error Profile")
    ax.set_xlabel("Epistemic Uncertainty")
    ax.legend(frameon=False)

    # --- G. MACRO PRECISION-RECALL ---
    ax = axes[6]
    precisions, aps = [], []
    for i in range(test_labels.shape[1]):
        if len(np.unique(test_labels[:, i])) > 1:
            p, r, _ = precision_recall_curve(test_labels[:, i], test_preds[:, i])
            aps.append(average_precision_score(test_labels[:, i], test_preds[:, i]))
            precisions.append(np.interp(all_fpr, r[::-1], p[::-1]))

    ax.plot(all_fpr, np.mean(precisions, axis=0), color=C_PURPLE, lw=2.5, label=f'Macro PR (mAP = {np.mean(aps):.3f})')
    ax.fill_between(all_fpr, np.percentile(precisions, 25, axis=0), np.percentile(precisions, 75, axis=0), color=C_PURPLE, alpha=0.2)
    ax.set_title("G. Macro Precision-Recall")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=False)

    # --- H. CLASS-WISE ECE ---
    ax = axes[7]
    class_eces = {}
    for i, name in enumerate(class_names):
        p_c, l_c = test_preds[:, i], test_labels[:, i]
        ece = sum((np.sum(m)/len(p_c)) * np.abs(np.mean(l_c[m]) - np.mean(p_c[m])) 
                  for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]) 
                  if np.any(m := (p_c > lower) & (p_c <= upper)))
        class_eces[name] = ece

    sort_ece = sorted(class_eces.keys(), key=lambda k: class_eces[k])
    ax.hlines(y=y_pos, xmin=0, xmax=[class_eces[k] for k in sort_ece], color=C_RED, alpha=0.6, lw=2.5)
    ax.scatter([class_eces[k] for k in sort_ece], y_pos, color=C_RED, s=50, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sort_ece)
    ax.set_title("H. Expected Calibration Error (ECE)")
    ax.set_xlabel("ECE (Lower is Better)")

    # --- I. ABSTENTION CURVE ---
    ax = axes[8]
    sort_idx = np.argsort(uncertainty)[::-1]
    rej_rates = np.linspace(0, 0.5, 15)
    ret_aucs = [np.mean([roc_auc_score(test_labels[sort_idx[int(len(sort_idx)*(1-r)):], i], 
                                       test_preds[sort_idx[int(len(sort_idx)*(1-r)):], i]) 
                         for i in range(test_labels.shape[1]) if len(np.unique(test_labels[sort_idx[int(len(sort_idx)*(1-r)):], i])) > 1]) 
                if int(len(sort_idx)*(1-r)) > 10 else np.nan for r in rej_rates]

    ax.plot(rej_rates * 100, ret_aucs, marker='o', color=C_BLUE, lw=2.5)
    ax.set_title("I. Accuracy vs. Uncertainty Abstention")
    ax.set_xlabel("Patients Rejected (%)")
    ax.set_ylabel("Retained Macro-AUROC")
    ax.grid(True, linestyle='--', alpha=0.5)

    # --- J. DECISION CURVE ANALYSIS (DCA) ---
    ax = axes[9]
    thresh = np.linspace(0.01, 0.8, 50)
    tp_all, fp_all, n = np.sum(l_flat == 1), np.sum(l_flat == 0), len(l_flat)
    nb_model = [(np.sum((p_flat >= t) & (l_flat == 1)) - np.sum((p_flat >= t) & (l_flat == 0)) * (t/(1-t))) / n for t in thresh]
    nb_all = [(tp_all - fp_all * (t/(1-t))) / n for t in thresh]

    ax.plot(thresh, nb_model, color=C_BLUE, lw=3, label='CXR-Synapse')
    ax.plot(thresh, nb_all, color=C_GRAY, linestyle='--', label='Treat All')
    ax.axhline(0, color='black', lw=1, label='Treat None')
    ax.set_ylim(-0.02, max(0.1, np.max(nb_model) * 1.2))
    ax.set_title("J. Clinical Net Benefit (DCA)")
    ax.set_xlabel("Probability Threshold")
    ax.set_ylabel("Net Benefit")
    ax.legend(frameon=False)

    # --- K. EPISTEMIC PROFILES ---
    ax = axes[10]
    u_data = [{'Pathology': n, 'Uncertainty': v} for i, n in enumerate(class_names) for v in uncertainty[test_labels[:, i] == 1]]
    if u_data:
        df_u = pd.DataFrame(u_data)
        # BUGFIX: Added hue='Pathology' and legend=False to fix seaborn FutureWarning
        sns.boxenplot(data=df_u, x='Uncertainty', y='Pathology', hue='Pathology', legend=False, ax=ax, palette="viridis", orient='h')
    ax.set_title("K. Epistemic Profiles (True Positives)")
    ax.set_xlabel("Bayesian Variance")
    ax.set_ylabel("")

    # --- L. ERROR CORRELATION ---
    ax = axes[11]
    err_corr = np.nan_to_num(np.corrcoef(test_labels - test_preds, rowvar=False), 0)
    sns.heatmap(err_corr, mask=np.triu(np.ones_like(err_corr, dtype=bool)), cmap='RdBu_r', center=0, 
                ax=ax, xticklabels=[n[:8] for n in class_names], yticklabels=[n[:8] for n in class_names])
    ax.set_title("L. Comorbidity Error Correlation")

    # --- M. DATASET PREVALENCE ---
    ax = axes[12]
    prev = test_labels.mean(axis=0) * 100
    s_idx = np.argsort(prev)
    ax.barh(np.array(class_names)[s_idx], prev[s_idx], color=C_SKY)
    ax.set_title("M. Pathology Prevalence (%)")

    # --- N. EMPIRICAL CO-OCCURRENCE ---
    ax = axes[13]
    co = np.dot(test_labels.T, test_labels)
    sns.heatmap(co / np.maximum(np.diag(co), 1), ax=ax, cmap="YlGnBu", 
                xticklabels=[n[:8] for n in class_names], yticklabels=[n[:8] for n in class_names])
    ax.set_title("N. Empirical Comorbidity P(Row|Col)")

    # --- O. ENTROPY GAP (HEALTHY VS DISEASED) ---
    ax = axes[14]
    has_disease = test_labels.sum(axis=1) > 0
    sns.kdeplot(uncertainty[has_disease], ax=ax, color=C_RED, fill=True, label='Pathological (1+ Disease)')
    sns.kdeplot(uncertainty[~has_disease], ax=ax, color=C_GREEN, fill=True, label='Healthy (No Disease)')
    ax.set_title("O. Epistemic Entropy Gap")
    ax.set_xlabel("Epistemic Uncertainty")
    ax.legend(frameon=False)

    sns.despine(fig)
    plt.tight_layout(pad=3.0) 
    plt.savefig(f"Scientific_Analysis_15Panel_{experiment_id}.pdf", dpi=600, bbox_inches='tight')
    plt.close()
    print("[✓] 15-Panel Suite Saved.")

# =====================================================================
# 2. XAI: PaQ ATTENTION MAP (INTERPRETABILITY)
# =====================================================================
def plot_paq_attention_evidence(image_tensor, attn_weights, pathology_name, save_id):
    """
    Visualizes the Pathology-as-Query (PaQ) attention focus on the raw X-ray.
    Overlays the 8x8 attention grid scaled up to image resolution.
    """
    print(f"[*] Generating PaQ Attention Map for {pathology_name}...")
    set_scientific_style()
    
    if isinstance(image_tensor, torch.Tensor):
        img = image_tensor.cpu().numpy().squeeze()
    else:
        img = image_tensor.squeeze()
        
    if img.ndim == 3: img = img[0]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8) 
    
    attn = attn_weights.reshape(8, 8) if isinstance(attn_weights, np.ndarray) else attn_weights.cpu().numpy().reshape(8, 8)
    attn_resized = cv2.resize(attn, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap='bone')
    im_attn = ax.imshow(attn_resized, cmap='inferno', alpha=0.45) 
    
    ax.axis('off')
    plt.title(f"PaQ Spatial Evidence: {pathology_name}", fontsize=16, fontweight='bold', pad=15)
    cbar = plt.colorbar(im_attn, fraction=0.046, pad=0.04)
    cbar.set_label('PaQ Attention Weight', rotation=270, labelpad=15)
    
    plt.savefig(f"PaQ_Attention_{pathology_name}_{save_id}.png", dpi=300, bbox_inches='tight')
    plt.close()

# =====================================================================
# 3. SEMANTIC MANIFOLD (LISA PROOF)
# =====================================================================
def plot_semantic_manifold(embeddings, labels, class_names, save_id):
    """
    Projects high-dimensional features into 2D via t-SNE to prove LISA constraints.
    """
    print("[*] Computing t-SNE Manifold for LISA Verification...")
    set_scientific_style()
    
    limit = min(3000, len(embeddings))
    emb_subset = embeddings[:limit]
    lbl_subset = labels[:limit]
    
    primary_labels = np.argmax(lbl_subset, axis=1)
    has_disease = lbl_subset.sum(axis=1) > 0
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    proj = tsne.fit_transform(emb_subset)
    
    plt.figure(figsize=(12, 10))
    plt.scatter(proj[~has_disease, 0], proj[~has_disease, 1], c='lightgray', alpha=0.3, label='Normal', s=10)
    
    top_classes = np.argsort(lbl_subset.sum(axis=0))[::-1][:5]
    colors = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE, C_RED]
    
    for idx, color in zip(top_classes, colors):
        mask = (primary_labels == idx) & has_disease
        plt.scatter(proj[mask, 0], proj[mask, 1], c=color, label=class_names[idx], alpha=0.7, s=25)

    plt.title("LISA Latent Topology (t-SNE Projection)", fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(markerscale=2, frameon=False, title="Primary Pathology")
    sns.despine()
    
    plt.savefig(f"LISA_Manifold_{save_id}.png", dpi=300, bbox_inches='tight')
    plt.close()

# =====================================================================
# 4. CONFORMAL SAFETY TRADEOFF
# =====================================================================
def plot_conformal_tradeoff(val_probs, val_labels, opt_thresholds, save_id):
    """
    Shows the mathematical tradeoff between required safety (Coverage)
    and clinical utility (Prediction Set Size).
    """
    print("[*] Calculating Conformal Efficiency Trade-offs...")
    set_scientific_style()
    
    alphas = np.linspace(0.01, 0.30, 15) 
    coverages, sizes = [], []
    val_true = val_labels.astype(bool)
    total_true = max(val_true.sum(), 1)
    
    for alpha in alphas:
        multipliers = np.linspace(2.0, 0.001, 500)
        best_m = 0.001
        for m in multipliers:
            preds = val_probs >= np.clip(opt_thresholds * m, 0.0001, 0.9999)
            if (preds & val_true).sum() / total_true >= (1.0 - alpha):
                best_m = m
                break
        
        final_preds = val_probs >= np.clip(opt_thresholds * best_m, 0.0001, 0.9999)
        coverages.append((final_preds & val_true).sum() / total_true * 100)
        sizes.append(final_preds.sum(axis=1).mean())

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = C_BLUE
    ax1.set_xlabel('Target Failure Rate (α)', fontweight='bold')
    ax1.set_ylabel('Empirical Coverage (%)', color=color1, fontweight='bold')
    ax1.plot(alphas, coverages, marker='o', color=color1, lw=3)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(90, color=color1, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color2 = C_ORANGE
    ax2.set_ylabel('Avg. Prediction Set Size', color=color2, fontweight='bold')
    ax2.plot(alphas, sizes, marker='s', color=color2, lw=3)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title("Conformal Safety vs. Efficiency Trade-off", fontsize=15, fontweight='bold')
    fig.tight_layout()
    plt.savefig(f"Conformal_Tradeoff_{save_id}.png", dpi=300, bbox_inches='tight')
    plt.close()