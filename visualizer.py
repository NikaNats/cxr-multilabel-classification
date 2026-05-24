from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import zoom as nd_zoom
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

# GPU-Accelerated UMAP check (cuml) with graceful CPU fallback
try:
    from cuml.manifold import UMAP as GPU_UMAP
    _HAS_CUML = True
except ImportError:
    import umap
    from umap.utils import disconnected_vertices
    _HAS_CUML = False

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# § 1  STYLE CONSTANTS & NATURE CONFIGURATION
# ==============================================================================

class Colour:
    """Wong (2011) color-blind-safe hexadecimal codes."""
    BLUE   = "#0072B2"
    ORANGE = "#D55E00"
    GREEN  = "#009E73"
    SKY    = "#56B4E9"
    PURPLE = "#CC79A7"
    YELLOW = "#F0E442"
    BLACK  = "#000000"
    GREY   = "#999999"

COLOUR_CYCLE: list[str] = [
    Colour.BLUE, Colour.ORANGE, Colour.GREEN,
    Colour.SKY, Colour.PURPLE, Colour.YELLOW,
]

_MM_TO_IN     = 1.0 / 25.4
SINGLE_COL_IN = 89  * _MM_TO_IN   # 89 mm single-column width
DOUBLE_COL_IN = 183 * _MM_TO_IN   # 183 mm double-column width
MAX_HEIGHT_IN = 230 * _MM_TO_IN   # 230 mm max height
FIGURE_DPI    = 450               # Min 300, 450 for highest resolution

_NATURE_RCPARAMS: dict = {
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":          5,         # Strict 5pt for dense multi-panel grids
    "axes.labelsize":     5,
    "axes.titlesize":     6,
    "axes.titleweight":   "normal",  
    "axes.linewidth":     0.5,       # Stroke weights 0.25 - 1 pt
    "lines.linewidth":    0.75,
    "patch.linewidth":    0.5,
    "xtick.major.size":   2,
    "ytick.major.size":   2,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.labelsize":    5,
    "ytick.labelsize":    5,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.top":          False,
    "ytick.right":        False,
    "axes.grid":          False,     # MUST be False per Nature guidelines
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "legend.fontsize":    4.5,       # Minimal size for legend to save space
    "legend.title_fontsize": 5,
    "legend.frameon":     False,
    "legend.handlelength":1.2,
    "legend.handletextpad":0.4,
    "figure.dpi":         FIGURE_DPI,
    "pdf.fonttype":       42,        # Editable text
    "ps.fonttype":        42,
    "savefig.dpi":        FIGURE_DPI,
    "savefig.bbox":       "tight",
}

_LABEL_KW = dict(fontsize=8, fontweight="bold", va="top", ha="left")
N_CALIB_BINS = 15

def configure_nature_style() -> None:
    """Updates global matplotlib configuration to enforce Nature style sheet."""
    plt.rcParams.update(_NATURE_RCPARAMS)
    sns.set_theme(style="white", palette=COLOUR_CYCLE, rc=_NATURE_RCPARAMS)

def _shorten_name(name: str) -> str:
    """Intelligently abbreviates long pathology names to prevent overlapping."""
    mapping = {
        "Cardiomegaly": "Cardiom.",
        "Atelectasis": "Atelect.",
        "Infiltration": "Infiltr.",
        "Pneumothorax": "Pneumoth.",
        "Consolidation": "Consol.",
        "Pleural_Thickening": "Pleural_Th.",
        "Pleural_Thicken": "Pleural_Th.",
        "Emphysema": "Emphys.",
    }
    return mapping.get(name, name[:10])

# ==============================================================================
# § 2  SHARED DATA CONTAINER
# ==============================================================================

@dataclass
class _DiagnosticData:
    labels:      np.ndarray
    preds:       np.ndarray
    pred_sets:   np.ndarray
    uncertainty: np.ndarray
    class_names: list[str]

    p_flat:      np.ndarray = field(init=False)
    l_flat:      np.ndarray = field(init=False)
    common_fpr:  np.ndarray = field(init=False)
    bin_edges:   np.ndarray = field(init=False)
    set_sizes:   np.ndarray = field(init=False)
    class_auroc: dict[str, float] = field(init=False)
    classes_asc: list[str]        = field(init=False)
    short_names: list[str]        = field(init=False)

    def __post_init__(self) -> None:
        self.p_flat     = self.preds.flatten()
        self.l_flat     = self.labels.flatten()
        self.common_fpr = np.linspace(0, 1, 200)
        self.bin_edges  = np.linspace(0, 1, N_CALIB_BINS + 1)
        self.set_sizes  = self.pred_sets.sum(axis=1)
        self.short_names= [_shorten_name(n) for n in self.class_names]
        
        self.class_auroc = {}
        for i, name in enumerate(self.class_names):
            if len(np.unique(self.labels[:, i])) > 1:
                self.class_auroc[name] = roc_auc_score(self.labels[:, i], self.preds[:, i])
            else:
                self.class_auroc[name] = 0.5
                
        self.classes_asc = sorted(self.class_auroc, key=self.class_auroc.__getitem__)

# ==============================================================================
# § 3  UTILITIES
# ==============================================================================

def _label_panel(ax: plt.Axes, letter: str) -> None:
    """Labels subplots alphabetically."""
    ax.text(-0.25, 1.10, letter, transform=ax.transAxes, **_LABEL_KW)

def _calibration_bins(p: np.ndarray, l: np.ndarray, bin_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorts predictions into confidence intervals for expected calibration error."""
    conf, acc = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if lo == 0.0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p > lo) & (p <= hi)
        conf.append(p[mask].mean() if mask.any() else np.nan)
        acc.append(l[mask].mean() if mask.any() else np.nan)
    return np.array(conf), np.array(acc)

def _ece_single_class(preds: np.ndarray, labels: np.ndarray, bin_edges: np.ndarray) -> float:
    """Calculates Expected Calibration Error (ECE) for a single pathology class."""
    n, ece = len(preds), 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if lo == 0.0:
            mask = (preds >= lo) & (preds <= hi)
        else:
            mask = (preds > lo) & (preds <= hi)
        if mask.any():
            ece += (mask.sum() / n) * abs(labels[mask].mean() - preds[mask].mean())
    return ece

def _legend_patches(labels_colours: list[tuple[str, str]]) -> list[Patch]:
    """Generates custom patch handles for legends."""
    return [Patch(facecolor=c, label=l) for l, c in labels_colours]

# ==============================================================================
# § 4  PANEL DRAWERS (Nature Guidelines Applied)
# ==============================================================================

def _draw_macro_roc(ax: plt.Axes, data: _DiagnosticData) -> None:
    """Plots macro-averaged Receiver Operating Characteristic curve."""
    common_fpr = np.linspace(0, 1, 500) 
    tprs = [np.interp(common_fpr, *roc_curve(data.labels[:, i], data.preds[:, i])[:2])
            for i in range(data.labels.shape[1]) if len(np.unique(data.labels[:, i])) > 1]
    
    mean_tpr = np.array(tprs).mean(axis=0)
    
    auc_val = auc(common_fpr, mean_tpr)
    auc_str = f"{auc_val:.3f}".replace("0.", ".")
    
    ax.plot(common_fpr, mean_tpr, color=Colour.BLUE, linewidth=1.0, 
            label=f"Macro (AUC = {auc_str})")
    
    ax.fill_between(common_fpr, np.percentile(tprs, 25, axis=0), 
                    np.percentile(tprs, 75, axis=0), 
                    color=Colour.SKY, alpha=0.2, linewidth=0)
    
    ax.plot([0, 1], [0, 1], color=Colour.GREY, linestyle="--", linewidth=0.75, zorder=1)
    
    ax.set_xlabel("False positive rate (proportion)")
    ax.set_ylabel("True positive rate (proportion)")
    ax.set_title("Macro-averaged ROC")
    
    ax.tick_params(
        axis='both', 
        which='major', 
        left=True,      
        bottom=True,    
        length=2, 
        width=0.5, 
        labelsize=5,
        pad=2           
    )
    
    ax.legend(loc="lower right", fontsize=4.5, frameon=False)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    sns.despine(ax=ax)
    _label_panel(ax, "a")

def _draw_calibration(ax: plt.Axes, data: _DiagnosticData) -> None:
    """Plots probability calibration curve."""
    _, acc = _calibration_bins(data.p_flat, data.l_flat, data.bin_edges)
    bin_w  = 1.0 / N_CALIB_BINS

    ax.bar(
        data.bin_edges[:-1], acc, width=bin_w, align="edge",
        color=Colour.ORANGE, edgecolor="white", linewidth=0.4, alpha=0.8,
        label="Empirical"
    )
    
    ax.plot([0, 1], [0, 1], color=Colour.GREY, ls="--", lw=0.75, 
            label="Perfect", zorder=1)
    
    ax.set_xlabel("Predicted confidence (probability)")
    ax.set_ylabel("Empirical accuracy (proportion)")
    ax.set_title("Probability calibration")
    
    ax.tick_params(
        axis='both', 
        which='major', 
        left=True,      
        bottom=True,    
        length=2, 
        width=0.5, 
        labelsize=5,
        pad=2           
    )
    
    ax.legend(loc="upper left", fontsize=4.5, frameon=False, borderaxespad=1.0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    sns.despine(ax=ax)
    _label_panel(ax, "b")

def _draw_uncertainty_vs_set_size(ax: plt.Axes, data: _DiagnosticData) -> None:
    """Plots epistemic uncertainty versus conformal set size."""
    n = min(len(data.uncertainty), 1500)
    
    sns.regplot(
        x=data.uncertainty[:n], 
        y=data.set_sizes[:n], 
        ax=ax,
        scatter_kws={
            "alpha": 0.15, 
            "s": 4, 
            "color": Colour.GREEN, 
            "linewidths": 0,    
            "rasterized": True  
        },
        line_kws={
            "color": Colour.ORANGE, 
            "linewidth": 1.2,
            "zorder": 5         
        }
    )
    
    ax.set_xlabel("Epistemic uncertainty (entropy)")
    ax.set_ylabel("Conformal set size (count)")
    ax.set_title("Uncertainty vs. set size")
    
    ax.tick_params(
        axis='both', 
        which='major', 
        left=True,      
        bottom=True,    
        length=2, 
        width=0.5, 
        labelsize=5,
        pad=2
    )
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    sns.despine(ax=ax)
    _label_panel(ax, "c")

def _draw_per_class_auroc(ax: plt.Axes, data: _DiagnosticData) -> None:
    """Plots individual pathology AUROC scores."""
    y_pos  = np.arange(len(data.classes_asc))
    aurocs = [data.class_auroc[c] for c in data.classes_asc]

    ax.hlines(y_pos, xmin=0.5, xmax=aurocs, colors=Colour.GREY, linewidth=0.5, alpha=0.4)
    ax.scatter(aurocs, y_pos, color=Colour.GREEN, s=12, zorder=3, linewidths=0)
    
    ax.set_xlabel("AUROC (score)")
    ax.set_title("Per-class performance")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_shorten_name(n) for n in data.classes_asc])
    
    ax.tick_params(
        axis='both', which='major', 
        left=True, bottom=True,    
        length=2, width=0.5, 
        labelsize=5, pad=1.5       
    )
    
    ax.set_xlim(0.5, 1.0)
    sns.despine(ax=ax)
    _label_panel(ax, "d")

def _draw_set_size_distribution(ax: plt.Axes, data: _DiagnosticData) -> None:
    """Plots prediction set size distribution histogram."""
    mean_sz = data.set_sizes.mean()
    
    sns.histplot(
        data.set_sizes, 
        discrete=True, 
        color=Colour.SKY,
        alpha=0.8, 
        ax=ax, 
        edgecolor="white", 
        linewidth=0.3
    )
    
    ax.axvline(
        mean_sz, 
        color=Colour.ORANGE, 
        linestyle="--", 
        linewidth=0.75,
        label=f"Mean = {mean_sz:.2f}"
    )
    
    ax.set_xlabel("Prediction set size (count)")
    ax.set_ylabel("Frequency (count)")
    ax.set_title("Set size distribution")
    
    ax.tick_params(
        axis='both', 
        which='major', 
        left=True, 
        bottom=True, 
        length=2, 
        width=0.5, 
        labelsize=5,
        pad=1.5
    )
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(loc="upper right", fontsize=4.5, frameon=False)
    sns.despine(ax=ax)
    _label_panel(ax, "e")

def _draw_uncertainty_by_error(ax: plt.Axes, data: _DiagnosticData) -> None:
    """Plots epistemic uncertainty partitioned by clinical error profile."""
    mae = np.abs(data.preds - data.labels).mean(axis=1)
    high_error = mae > np.median(mae)

    sns.kdeplot(
        data.uncertainty[~high_error], 
        ax=ax, color=Colour.BLUE, 
        fill=True, alpha=0.35, linewidth=0.75,
        clip=(0, None), label="Low error"
    )
    sns.kdeplot(
        data.uncertainty[high_error], 
        ax=ax, color=Colour.ORANGE, 
        fill=True, alpha=0.35, linewidth=0.75,
        clip=(0, None), label="High error"
    )
    
    ax.set_xlabel("Epistemic uncertainty (entropy)")
    ax.set_ylabel("Density (proportion)")
    ax.set_title("Uncertainty by error profile")
    
    u_min, u_max = data.uncertainty.min(), data.uncertainty.max()
    ax.set_xlim(u_min * 0.95, min(1.05, u_max * 1.05))

    ax.tick_params(
        axis='both', 
        which='major', 
        left=True,      
        bottom=True,    
        length=2, 
        width=0.5, 
        labelsize=5
    )
    
    ax.legend(
        handles=_legend_patches([
            ("Low error", Colour.BLUE),
            ("High error", Colour.ORANGE)
        ]),
        loc="upper right",
        fontsize=4.5
    )

    sns.despine(ax=ax)
    _label_panel(ax, "f")

def _draw_macro_pr(ax: plt.Axes, data: _DiagnosticData) -> None:
    """Plots macro-averaged Precision-Recall curves."""
    interp_precs, aps = [], []
    for i in range(data.labels.shape[1]):
        if len(np.unique(data.labels[:, i])) > 1:
            prec, rec, _ = precision_recall_curve(
                data.labels[:, i], data.preds[:, i]
            )
            aps.append(
                average_precision_score(data.labels[:, i], data.preds[:, i])
            )
            interp_precs.append(
                np.interp(data.common_fpr, rec[::-1], prec[::-1])
            )

    prec_matrix = np.array(interp_precs)
    mean_prec = prec_matrix.mean(axis=0)
    
    map_val = np.mean(aps)
    map_str = f"{map_val:.3f}".replace("0.", ".")
    
    ax.plot(
        data.common_fpr, mean_prec,
        color=Colour.PURPLE, linewidth=1.0,
        label=f"Macro PR (mAP = {map_str})",
    )
    
    ax.fill_between(
        data.common_fpr,
        np.percentile(prec_matrix, 25, axis=0),
        np.percentile(prec_matrix, 75, axis=0),
        color=Colour.PURPLE, alpha=0.15, linewidth=0
    )
    
    ax.set_xlabel("Recall (proportion)")
    ax.set_ylabel("Precision (proportion)")
    ax.set_title("Macro precision-recall")
    
    ax.tick_params(
        axis='both', 
        which='major', 
        left=True,      
        bottom=True,    
        length=2, 
        width=0.5, 
        labelsize=5,
        pad=2           
    )
    
    ax.legend(loc="upper right", fontsize=4.5, frameon=False)
    sns.despine(ax=ax)
    _label_panel(ax, "g")

def _draw_per_class_ece(ax: plt.Axes, data: _DiagnosticData) -> None:
    """Plots expected calibration error (ECE) per pathology."""
    class_ece = {
        name: _ece_single_class(data.preds[:, i], data.labels[:, i], data.bin_edges) 
        for i, name in enumerate(data.class_names)
    }
    
    sorted_names = sorted(class_ece, key=class_ece.__getitem__)
    ece_vals = [class_ece[c] for c in sorted_names]
    y_pos = np.arange(len(sorted_names))

    ax.hlines(y_pos, xmin=0, xmax=ece_vals, colors=Colour.ORANGE, linewidth=0.6, alpha=0.6)
    ax.scatter(ece_vals, y_pos, color=Colour.ORANGE, s=12, zorder=3, linewidths=0)
    
    ax.set_xlabel("Expected calibration error (score)")
    ax.set_title("Calibration Error (ECE)")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_shorten_name(n) for n in sorted_names])
    
    ax.tick_params(axis='y', which='major', left=True, length=2, width=0.5, pad=2)
    ax.tick_params(axis='x', which='major', bottom=True, length=2, width=0.5)
    ax.tick_params(labelsize=5)
    
    ax.set_xlim(0, max(ece_vals) * 1.1)
    sns.despine(ax=ax)
    _label_panel(ax, "h")

def _draw_abstention_curve(ax: plt.Axes, data: _DiagnosticData) -> None:
    """Plots model macro-AUROC against sample rejection rate."""
    n = len(data.uncertainty)
    rejection_order = np.argsort(data.uncertainty)[::-1] 
    rejection_rates = np.linspace(0, 0.50, 12)
    retained_aucs = []

    for rate in rejection_rates:
        num_rejected = int(n * rate)
        kept_idx = rejection_order[num_rejected:]
        
        if len(kept_idx) < 10:
            retained_aucs.append(np.nan)
            continue
            
        current_aucs = [
            roc_auc_score(data.labels[kept_idx, i], data.preds[kept_idx, i])
            for i in range(data.labels.shape[1])
            if len(np.unique(data.labels[kept_idx, i])) > 1
        ]
        retained_aucs.append(np.mean(current_aucs) if current_aucs else np.nan)

    ax.plot(rejection_rates * 100, retained_aucs, 
            marker="o", markersize=2.0, color=Colour.BLUE, 
            linewidth=1.0, clip_on=False, zorder=3)
    
    ax.set_xlabel("Samples rejected (%)")
    ax.set_ylabel("Retained macro-AUROC (score)") 
    ax.set_title("Accuracy-rejection curve")
    
    ax.tick_params(
        axis='both', which='major', labelsize=5, 
        length=2, width=0.5, left=True, bottom=True, pad=1.5
    )
    
    if not np.isnan(retained_aucs).all():
        valid_vals = np.array(retained_aucs)[~np.isnan(retained_aucs)]
        ymin = np.min(valid_vals) - (np.max(valid_vals) - np.min(valid_vals)) * 0.2
        ymax = np.max(valid_vals) + (np.max(valid_vals) - np.min(valid_vals)) * 0.2
        ax.set_ylim(max(0.5, ymin), min(1.0, ymax))

    ax.set_xlim(-2, 52)
    sns.despine(ax=ax)
    _label_panel(ax, "i")

def _draw_decision_curve(ax: plt.Axes, data: _DiagnosticData) -> None:
    """Plots Decision Curve Analysis (DCA) to measure clinical net benefit."""
    thresholds = np.linspace(0.01, 0.80, 50)
    n = len(data.l_flat)
    n_pos = data.l_flat.sum()
    n_neg = n - n_pos
    
    nb_model = [
        ((np.sum((data.p_flat >= t) & (data.l_flat == 1)) - 
          np.sum((data.p_flat >= t) & (data.l_flat == 0)) * (t / (1.0 - t))) / n) 
        for t in thresholds
    ]
    nb_all = [(n_pos - n_neg * (t / (1.0 - t))) / n for t in thresholds]

    ax.plot(thresholds, nb_model, color=Colour.BLUE, linewidth=1.2, label="CXR-Synapse (Model)")
    ax.plot(thresholds, nb_all, color=Colour.GREY, linewidth=0.75, linestyle="--", label="Treat all")
    ax.axhline(0, color=Colour.BLACK, linewidth=0.75, label="Treat none")
    
    max_val = max(max(nb_model), 0.04) 
    ax.set_ylim(-0.01, max_val * 1.2)
    
    ax.set_xlabel("Probability threshold (probability)")
    ax.set_ylabel("Net benefit (score)")
    ax.set_title("Decision curve analysis")
    
    ax.tick_params(axis='both', which='major', labelsize=5, length=2, width=0.5, left=True, bottom=True)
    ax.legend(loc="upper right", fontsize=4.5, frameon=False)
    sns.despine(ax=ax)
    _label_panel(ax, "j")

def _draw_uncertainty_by_class(ax: plt.Axes, data: _DiagnosticData) -> None:
    """Plots epistemic uncertainty partitioned by medical pathologies."""
    # OPTIMIZATION: Vectorized DataFrame generation avoids slow nested Python loops
    uncertainty_list = []
    pathology_list = []
    
    for i, name in enumerate(data.class_names):
        mask = data.labels[:, i] == 1
        vals = data.uncertainty[mask]
        n_samples = len(vals)
        if n_samples > 0:
            display_name = f"{_shorten_name(name)} (n={n_samples})"
            uncertainty_list.append(vals)
            pathology_list.append(np.full(n_samples, display_name))
            
    if not uncertainty_list:
        ax.set_visible(False)
        return
        
    df = pd.DataFrame({
        "Uncertainty": np.concatenate(uncertainty_list),
        "Pathology": np.concatenate(pathology_list)
    })
    
    sns.stripplot(
        data=df, x="Uncertainty", y="Pathology",
        color=Colour.SKY, alpha=0.3, s=1.5, 
        jitter=0.25, ax=ax, zorder=1, linewidth=0
    )
    
    sns.boxplot(
        data=df, x="Uncertainty", y="Pathology",
        color="white", ax=ax, orient="h",
        width=0.4, linewidth=0.6,
        showfliers=False, 
        boxprops={'alpha': 0.6, 'edgecolor': Colour.BLACK},
        whiskerprops={'color': Colour.BLACK, 'linewidth': 0.5},
        medianprops={'color': Colour.ORANGE, 'linewidth': 1.0}, 
        capprops={'color': Colour.BLACK, 'linewidth': 0.5},
        zorder=2
    )
    
    ax.set_xlabel("Epistemic uncertainty (entropy)")
    ax.set_ylabel("")
    ax.set_title("Epistemic profiles (True Positives)")
    
    ax.tick_params(axis='y', left=True, length=2, width=0.5, pad=2)
    ax.tick_params(axis='both', labelsize=4.5) 
    
    u_min, u_max = df["Uncertainty"].min(), df["Uncertainty"].max()
    
    if u_min == u_max:
        ax.set_xlim(u_min - 0.05, u_min + 0.05)
    else:
        ax.set_xlim(u_min * 0.95, min(1.05, u_max * 1.05))
    _label_panel(ax, "k")

def _draw_error_correlation(ax: plt.Axes, data: _DiagnosticData) -> None:
    """Plots correlation matrix of residual errors."""
    residuals = data.labels.astype(float) - data.preds
    corr = np.nan_to_num(np.corrcoef(residuals, rowvar=False), nan=0.0)
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, 
        mask=mask, 
        cmap="RdBu_r", 
        center=0, 
        vmin=-1, vmax=1, 
        ax=ax,
        xticklabels=data.short_names, 
        yticklabels=data.short_names, 
        linewidths=0, 
        rasterized=True, 
        cbar_kws={
            "shrink": 0.8, 
            "label": "Correlation (score)",
            "ticks": [-1, -0.5, 0, 0.5, 1]
        }
    )
    
    ax.set_xticklabels(
        ax.get_xticklabels(), 
        rotation=45, 
        ha="right", 
        rotation_mode="anchor",
        fontsize=5
    )
    
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=5)
    
    ax.tick_params(
        axis='both', 
        which='major', 
        left=True, 
        bottom=True, 
        length=2, 
        width=0.5, 
        pad=1.5  
    )
    
    ax.set_title("Error correlation")
    _label_panel(ax, "l")

def _draw_prevalence(ax: plt.Axes, data: _DiagnosticData) -> None:
    """Plots empirical pathology prevalence percentages."""
    prev = data.labels.mean(axis=0) * 100
    idx = np.argsort(prev)
    
    sorted_names = np.array(data.short_names)[idx]
    sorted_values = prev[idx]

    ax.barh(sorted_names, sorted_values, color=Colour.SKY, linewidth=0)
    
    ax.set_xlabel("Prevalence (%)")
    ax.set_title("Pathology prevalence")
    
    ax.tick_params(axis='y', which='major', left=True, length=2, width=0.5)
    ax.tick_params(axis='y', pad=2)
    ax.tick_params(axis='x', which='major', bottom=True, length=2, width=0.5)
    ax.tick_params(labelsize=5)
    
    ax.set_xlim(0, max(sorted_values) * 1.1)
    
    sns.despine(ax=ax, top=True, right=True)
    _label_panel(ax, "m")

def _draw_cooccurrence(ax: plt.Axes, data: _DiagnosticData) -> None:
    """Plots empirical comorbidity probabilities between clinical labels."""
    co = data.labels.T @ data.labels
    
    p_row_given_col = co / np.maximum(np.diag(co)[None, :], 1)
    
    sns.heatmap(
        p_row_given_col, 
        ax=ax, 
        cmap="YlGnBu",
        vmin=0, vmax=1,
        xticklabels=data.short_names, 
        yticklabels=data.short_names,
        linewidths=0,       
        rasterized=True,    
        cbar_kws={
            "shrink": 0.8, 
            "label": "P(row | col) (probability)", 
            "ticks": [0, 0.5, 1.0]
        }
    )
    
    ax.set_xticklabels(
        ax.get_xticklabels(), 
        rotation=45, 
        ha="right", 
        rotation_mode="anchor",
        fontsize=5
    )
    
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=5)
    
    ax.tick_params(
        axis='both', 
        which='major', 
        left=True, 
        bottom=True, 
        length=2, 
        width=0.5, 
        pad=1.5
    )
    
    ax.set_title("Empirical comorbidity")
    _label_panel(ax, "n")

def _draw_entropy_gap(ax: plt.Axes, data: _DiagnosticData) -> None:
    """Plots epistemic entropy distribution gaps between diseased and healthy cases."""
    has_disease = data.labels.sum(axis=1) > 0
    
    sns.kdeplot(
        data.uncertainty[has_disease], 
        ax=ax, color=Colour.ORANGE, 
        fill=True, alpha=0.40, linewidth=0.75,
        clip=(0, None), label="Pathological (>=1)"
    )
    sns.kdeplot(
        data.uncertainty[~has_disease], 
        ax=ax, color=Colour.GREEN, 
        fill=True, alpha=0.40, linewidth=0.75,
        clip=(0, None), label="Healthy (0)"
    )
    
    ax.set_xlabel("Epistemic uncertainty (entropy)")
    ax.set_ylabel("Density (proportion)")
    ax.set_title("Epistemic entropy gap")
    
    u_min, u_max = data.uncertainty.min(), data.uncertainty.max()
    ax.set_xlim(u_min * 0.95, min(1.05, u_max * 1.05)) 
    
    ax.legend(
        handles=_legend_patches([
            ("Pathological (>=1)", Colour.ORANGE),
            ("Healthy (0)", Colour.GREEN)
        ]),
        loc="upper right",
        fontsize=4.5
    )
    
    ax.tick_params(
        axis='both', 
        which='major', 
        labelsize=5, 
        length=2, 
        width=0.5,
        left=True,      
        bottom=True     
    )
    
    sns.despine(ax=ax)
    _label_panel(ax, "o")

_PANEL_REGISTRY: list[tuple[str, callable]] = [
    ("a", _draw_macro_roc), ("b", _draw_calibration), ("c", _draw_uncertainty_vs_set_size),
    ("d", _draw_per_class_auroc), ("e", _draw_set_size_distribution), ("f", _draw_uncertainty_by_error),
    ("g", _draw_macro_pr), ("h", _draw_per_class_ece), ("i", _draw_abstention_curve),
    ("j", _draw_decision_curve), ("k", _draw_uncertainty_by_class), ("l", _draw_error_correlation),
    ("m", _draw_prevalence), ("n", _draw_cooccurrence), ("o", _draw_entropy_gap),
]

# ==============================================================================
# § 5  PUBLIC EXPORT API
# ==============================================================================

def _save_figure(fig: plt.Figure, path: Path, fmt: str = "pdf") -> None:
    """Exports generated matplotlib figures to disk."""
    out = path.with_suffix(f".{fmt}")
    fig.savefig(out, format=fmt, dpi=FIGURE_DPI, bbox_inches="tight", transparent=False)
    plt.close(fig)

def plot_diagnostic_suite(
    test_labels: np.ndarray, test_preds: np.ndarray, conformal_sets: np.ndarray,
    uncertainty: np.ndarray, class_names: list[str], experiment_id: str, output_dir: str | Path = "."
) -> None:
    """Generates the multi-panel evaluation diagnostic suite."""
    configure_nature_style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = _DiagnosticData(labels=test_labels, preds=test_preds, pred_sets=conformal_sets, 
                           uncertainty=uncertainty, class_names=class_names)

    fig, axes = plt.subplots(5, 3, figsize=(DOUBLE_COL_IN, MAX_HEIGHT_IN), layout="constrained")
    for ax, (_, drawer) in zip(axes.flatten(), _PANEL_REGISTRY):
        drawer(ax, data)
    sns.despine(fig)
    _save_figure(fig, out / f"diagnostic_suite_{experiment_id}")

    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8}) 
    for letter, drawer in _PANEL_REGISTRY:
        p_fig, p_ax = plt.subplots(figsize=(SINGLE_COL_IN, SINGLE_COL_IN * 0.85), layout="constrained")
        drawer(p_ax, data)
        sns.despine(p_fig)
        _save_figure(p_fig, out / f"panel_{letter}_{experiment_id}")
    configure_nature_style() 

def plot_conformal_tradeoff(val_probs: np.ndarray, val_labels: np.ndarray, opt_thresholds: np.ndarray, experiment_id: str, output_dir: str | Path = "."):
    """Plots conformal prediction coverage vs. set size trade-off curve."""
    configure_nature_style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    val_true = val_labels.astype(bool)
    total_true = max(val_true.sum(), 1)
    alphas = np.linspace(0.01, 0.30, 15)
    coverages, set_sizes = [], []

    for alpha in alphas:
        best_m = 0.001
        for m in np.linspace(2.0, 0.001, 500):
            adjusted = np.clip(opt_thresholds * m, 1e-4, 0.9999)
            if ((val_probs >= adjusted) & val_true).sum() / total_true >= 1.0 - alpha:
                best_m = m
                break
        final_sets = val_probs >= np.clip(opt_thresholds * best_m, 1e-4, 0.9999)
        coverages.append((final_sets & val_true).sum() / total_true * 100)
        set_sizes.append(final_sets.sum(axis=1).mean())

    fig, ax1 = plt.subplots(figsize=(SINGLE_COL_IN * 1.6, SINGLE_COL_IN), layout="constrained")
    
    ax1.plot(alphas, coverages, marker="o", markersize=3, color=Colour.BLUE, linewidth=1.0, label="Empirical coverage")
    ax1.axhline(90, color=Colour.BLUE, linestyle="--", linewidth=0.5, alpha=0.5)
    ax1.text(0.28, 91, "Target coverage", color=Colour.BLUE, fontsize=4, ha="right")
    
    ax1.set_xlabel(r"Target failure rate $\alpha$ (proportion)")
    ax1.set_ylabel("Empirical coverage (%)")
    ax1.set_ylim(60, 105)

    ax2 = ax1.twinx()
    ax2.plot(alphas, set_sizes, marker="s", markersize=3, color=Colour.ORANGE, linewidth=1.0, label="Mean set size")
    ax2.set_ylabel("Mean prediction set size (count)")
    
    ax1.tick_params(axis='both', which='major', left=True, bottom=True, length=2, width=0.5, labelsize=5)
    ax2.tick_params(axis='y', which='major', right=True, length=2, width=0.5, labelsize=5)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=4.5, frameon=False)

    ax1.set_title("Conformal safety-efficiency trade-off", fontsize=7, fontweight="bold")
    
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    _save_figure(fig, out / f"conformal_tradeoff_{experiment_id}")

def plot_semantic_manifold(embeddings: np.ndarray, labels: np.ndarray, class_names: list[str], experiment_id: str, output_dir: str | Path = ".", max_samples: int = 3000):
    """Generates the high-dimensional latent space UMAP projection."""
    configure_nature_style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    n = min(max_samples, len(embeddings))
    emb, lbl = embeddings[:n], labels[:n]
    primary_class, has_disease = np.argmax(lbl, axis=1), lbl.sum(axis=1) > 0

    # Feature Standardization (SOTA Preprocessing Requirement)
    scaled_emb = StandardScaler().fit_transform(emb)

    # Fit SOTA densMAP with Cosine Metric (Reflects UMAP 0.5.x Best Practices)
    if _HAS_CUML:
        reducer = GPU_UMAP(
            n_neighbors=30,
            min_dist=0.1,
            n_components=2,
            metric='cosine',
            random_state=42
        )
    else:
        reducer = umap.UMAP(
            n_neighbors=30,             # recommended larger n_neighbors for densMAP
            min_dist=0.1,
            n_components=2,
            metric='cosine',            # Mathematically rigorous for deep latent features
            densmap=True,               # Preserves semantic density (Crucial for Nature)
            dens_lambda=2.0,            # Standard density weight
            random_state=42,            # For absolute reproducibility
            n_jobs=-1                   # Allow multithreading control where applicable
        )
    
    proj = reducer.fit_transform(scaled_emb)
    
    if not _HAS_CUML:
        disconnected = disconnected_vertices(reducer)
        valid_mask = ~disconnected
        proj = proj[valid_mask]
        has_disease = has_disease[valid_mask]
        primary_class = primary_class[valid_mask]
    else:
        # GPU UMAP handles connectivity natively, preserve indexing
        valid_mask = np.ones(len(proj), dtype=bool)
    
    fig, ax = plt.subplots(figsize=(3.5, 3.5), layout="constrained")

    ax.scatter(proj[~has_disease, 0], proj[~has_disease, 1], 
               c=Colour.GREY, alpha=0.15, s=3, lw=0, label="Healthy", rasterized=True)
    
    top5 = np.argsort(lbl.sum(axis=0))[::-1][:5]
    for colour, cls_idx in zip(COLOUR_CYCLE, top5):
        mask = (primary_class == cls_idx) & has_disease
        ax.scatter(proj[mask, 0], proj[mask, 1], 
                   c=colour, s=6, alpha=0.8, lw=0, rasterized=True)

    handles = [Patch(facecolor=Colour.GREY, label="Healthy")] + \
              [Patch(facecolor=COLOUR_CYCLE[k], label=_shorten_name(class_names[idx])) 
               for k, idx in enumerate(top5)]
    
    ax.legend(handles=handles, title="Primary pathology", loc="upper right", 
              bbox_to_anchor=(1.25, 1.0), markerscale=1, fontsize=4.5, title_fontsize=5)

    ax.set_xlabel("UMAP dimension 1 (a.u.)")
    ax.set_ylabel("UMAP dimension 2 (a.u.)")
    ax.set_title("LISA latent topology (densMAP)")
    
    ax.tick_params(
        axis='both', 
        which='major', 
        left=True,      
        bottom=True,    
        length=2, 
        width=0.5, 
        labelsize=5
    )
    
    sns.despine(ax=ax)
    _save_figure(fig, out / f"manifold_umap_{experiment_id}", fmt="png")

def plot_paq_attention(image_array, attn_weights, pathology_name: str, experiment_id: str, output_dir: str | Path = "."):
    """Plots cross-attention weight heatmaps overlaid on the input image."""
    configure_nature_style()
    plt.update({"font.size": 7})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    img = image_array.cpu().numpy().squeeze() if hasattr(image_array, "cpu") else np.asarray(image_array).squeeze()
    if img.ndim == 3: img = img[0]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    attn = (attn_weights.cpu().numpy() if hasattr(attn_weights, "cpu") else np.asarray(attn_weights)).reshape(8, 8).astype(float)
    attn_up = nd_zoom(attn, (img.shape[0]/8, img.shape[1]/8), order=3)
    attn_up = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, SINGLE_COL_IN), layout="constrained")
    ax.imshow(img, cmap="bone", interpolation="none")
    im = ax.imshow(attn_up, cmap="magma", alpha=0.5, interpolation="none") 
    ax.set_axis_off()

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label("Attention weight (normalized)", rotation=270, labelpad=10, fontsize=6)
    ax.set_title(f"PaQ evidence: {pathology_name}", fontsize=8, fontweight="bold", pad=8)
    
    _save_figure(fig, out / f"paq_attention_{pathology_name.replace(' ', '_')}_{experiment_id}", fmt="png")
    configure_nature_style()

# Clean up visualizer memory
plt.close('all')
gc.collect()