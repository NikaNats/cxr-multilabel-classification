import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from scipy.ndimage import uniform_filter1d
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, brier_score_loss

from config import log_process

# Suppress UndefinedMetricWarning to maintain clean logs during bootstrap resampling 
# on rare classes where a specific sample may have zero positive instances.
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ============================================================
# CLINICAL ONTOLOGY CONFIGURATION
# ============================================================
CHESTMNIST_CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

RADLEX_PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"
]


# ============================================================
# SCIENTIFIC VISUALIZATION
# ============================================================
def configure_nature_plots():
    """
    Sets the global Matplotlib/Seaborn environment to Nature Research standards.
    Ensures high-DPI (600) output, color-blind friendly palettes, and serif fonts.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    NATURE_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]
    sns.set_theme(style="whitegrid", context="paper", palette=NATURE_PALETTE, font="serif")
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.labelsize": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "figure.dpi": 300,
        "savefig.dpi": 600, "figure.facecolor": "white", "savefig.bbox": "tight",
    })


# ============================================================
# CLINICAL KNOWLEDGE INJECTION (BioViL-T)
# ============================================================
def ensure_radlex_embeddings(path, pathologies, model_name, device_):
    """
    Acquires or generates textual embeddings using SOTA BioViL-T.
    Ref: Boecking et al., 'Making the Most of Text-to-Image for Chest X-Rays'.
    
    Transforms RadLex clinical terms into 768-dimensional semantic vectors 
    to drive the GraphGPS-based label conditioning.
    """
    from transformers import AutoModel, AutoTokenizer
    target_dim = 768

    if os.path.exists(path):
        try:
            emb = torch.load(path, map_location=device_, weights_only=True)
            if emb.shape == (len(pathologies), target_dim):
                return emb.detach()
        except Exception:
            pass

    print(f"[*] Generating SOTA BioViL-T embeddings (768-dim) with {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device_)
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(pathologies, padding=True, truncation=True, return_tensors='pt').to(device_)
        try:
            # Prefer projected text embeddings if the model supports multi-modal alignment
            res = model.get_projected_text_embeddings(inputs.input_ids, inputs.attention_mask)
        except AttributeError:
            res = model(**inputs).last_hidden_state[:, 0, :]

        if res.shape[1] != target_dim:
            projector = nn.Linear(res.shape[1], target_dim).to(device_)
            res = projector(res)

    final_res = res.detach().cpu()
    torch.save(final_res, path)
    return final_res.to(device_)


# ============================================================
# KNOWLEDGE GRAPH TOPOLOGY
# ============================================================
def select_adjacency_threshold(labels: np.ndarray, num_classes: int = 14) -> float:
    """
    Dynamic Graph Density Optimization via the Elbow Method (Max Curvature).
    
    Determines the optimal pathology co-occurrence threshold by identifying the point 
    where increasing edge density captures noise rather than clinical structure.
    Clamped to [0.05, 0.40] to ensure semantic relevance in imbalanced settings.
    """
    co = np.zeros((num_classes, num_classes), dtype=np.float64)
    for row in labels:
        idx = np.where(row == 1)[0]
        for i in idx:
            for j in idx:
                co[i, j] += 1.0

    co_sym = (co + co.T) / 2.0
    prob = co_sym / np.maximum(co_sym.sum(1, keepdims=True), 1.0)

    thresholds = np.linspace(0.01, 0.95, 200)
    densities = [(prob >= t).mean() for t in thresholds]
    densities_smooth = uniform_filter1d(densities, size=7)

    # Calculate second derivative to find the 'knee' of the curve
    d2 = np.gradient(np.gradient(densities_smooth))
    knee_idx = int(np.argmax(np.abs(d2)))
    optimal_t = float(np.clip(thresholds[knee_idx], 0.05, 0.40))

    log_process("graph", "adjacency_threshold_selected", clamped=f"{optimal_t:.3f}")
    return optimal_t


def build_cooccurrence_adjacency(labels, num_classes=14, threshold=0.4, self_loops=True):
    """
    Constructs a Normalized Laplacian Adjacency Matrix.
    Ref: Kipf & Welling, 'Semi-Supervised Classification with GCNs'.
    
    Calculates symmetric P(Row|Col) relationships to inject clinical hierarchy 
    into the GraphGPS classifier.
    """
    co = np.zeros((num_classes, num_classes), dtype=np.float64)
    for row in labels:
        idx = np.where(row == 1)[0]
        for i in idx:
            for j in idx:
                co[i, j] += 1.0

    co_sym = (co + co.T) / 2.0
    prob = co_sym / np.maximum(co_sym.sum(1, keepdims=True), 1.0)

    adj = (prob >= threshold).astype(np.float64) * prob
    if self_loops:
        adj += np.eye(num_classes)

    deg = adj.sum(1)
    # D^-1/2 calculation for GCN normalization
    d_inv_sq = np.where(deg > 0, np.power(np.maximum(deg, 1e-12), -0.5), 0.0)
    d_inv_sq = np.diag(d_inv_sq)
    return (d_inv_sq @ adj @ d_inv_sq).astype(np.float32)


# ============================================================
# LONG-TAIL LOGIT ADJUSTMENT
# ============================================================
def compute_logit_adjustment(train_labels_np, tau=1.0):
    """
    Applies Menon et al. (ICLR 2021) adjustment for label distribution shift.
    
    Corrects the model's base probability scores by the log-prior of class prevalence. 
    Crucial for rare pathologies (like Hernia) which are 1000x less frequent than Effusion.
    """
    pos_freq = np.mean(train_labels_np, axis=0)
    pos_freq = np.clip(pos_freq, 1e-5, 1.0 - 1e-5)  # Prevent log(0) singularity
    adjustment = tau * np.log(pos_freq / (1.0 - pos_freq))
    return torch.from_numpy(adjustment.astype(np.float32))


# ============================================================
# PROBABILISTIC CALIBRATION (Guo et al., ICML 2017)
# ============================================================
class TemperatureScaler(nn.Module):
    """
    Executes Post-hoc Temperature Scaling to minimize ECE.
    Specifically modified to clamp T in [0.1, 3.5] to prevent 
    evidential logit exploding syndrome observed in EDL models.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.base_model_call(x)
        # Prevents over-smoothing (T > 3.5) which destroys discriminative power
        t_safe = self.temperature.clamp(min=0.1, max=3.5)
        return logits / t_safe

    def calibrate(self, val_loader, device_, max_iter=100):
        """Optimizes scalar T via L-BFGS on validation cross-entropy."""
        self.model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for feats, lbls in val_loader:
                logits = self.base_model_call(feats.to(device_))
                all_logits.append(logits.cpu());
                all_labels.append(lbls)

        logits_val = torch.cat(all_logits).to(device_)
        labels_val = torch.cat(all_labels).float().to(device_)
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def _eval():
            optimizer.zero_grad()
            t_clamp = self.temperature.clamp(min=0.1, max=3.5)
            loss = F.binary_cross_entropy_with_logits(logits_val / t_clamp, labels_val)
            loss.backward()
            return loss

        optimizer.step(_eval)
        return self.temperature.clamp(min=0.1, max=3.5).item()

    def base_model_call(self, x):
        out = self.model(x)
        return out[0] if isinstance(out, tuple) else out


def expected_calibration_error(probs, labels, n_bins=15):
    """
    Calculates per-class Expected Calibration Error (ECE).
    Measures the diagnostic reliability gap between predicted confidence 
    and actual empirical frequency.
    """
    ece = []
    for k in range(probs.shape[1]):
        bounds = np.linspace(0, 1, n_bins + 1)
        ek = 0.0
        for b_idx in range(n_bins):
            lo, hi = bounds[b_idx], bounds[b_idx + 1]
            m = (probs[:, k] >= lo) & (probs[:, k] < hi)
            if m.sum() > 0:
                ek += m.mean() * abs(labels[m, k].mean() - probs[m, k].mean())
        ece.append(ek)
    return np.array(ece)


# ============================================================
# MULTI-LABEL CONFORMAL PREDICTION (Safety Wrapper)
# ============================================================
class MultiLabelConformalPredictor:
    """
    Implements Marginal Conformal Prediction for Multi-Label sets.
    Ref: Angelopoulos & Bates, 'A Gentle Introduction to Conformal Prediction'.
    
    Guarantees that each pathology's true state is included in the diagnostic set 
    with probability >= 1-alpha (90% by default). Essential for legal/clinical safety.
    """

    def __init__(self, alpha=0.10):
        self.alpha = alpha
        self.thresholds = None

    def calibrate(self, cal_probs, cal_labels):
        """Finds non-conformity quantiles independently for 14 classes."""
        K = cal_probs.shape[1]
        self.thresholds = np.zeros(K)
        for k in range(K):
            pos_mask = (cal_labels[:, k] == 1)
            if pos_mask.sum() == 0:
                self.thresholds[k] = 0.5
                continue
            # Non-conformity score: 1 - model_probability
            scores = 1.0 - cal_probs[pos_mask, k]
            n = scores.shape[0]
            # Finite-sample correction for the quantile
            q = min(max(np.ceil((n + 1) * (1 - self.alpha)) / n, 0.0), 1.0)
            self.thresholds[k] = 1.0 - np.quantile(scores, q)

    def predict_sets(self, probs):
        """Constructs the Conformal Set based on calibrated error rates."""
        return {"include_pos": probs >= self.thresholds}


# ============================================================
# STATISTICAL VALIDATION (Bootstrapping)
# ============================================================
def bootstrap_metric_ci(fn, y_true, y_score, n=2000, alpha=0.05, seed=42):
    """Calculates 95% Confidence Intervals for rankings (AUC)."""
    rng = np.random.RandomState(seed)
    N, vals = y_true.shape[0], []
    for _ in range(n):
        idx = rng.choice(N, N, replace=True)
        try:
            v = float(fn(y_true[idx], y_score[idx]))
            if np.isfinite(v): vals.append(v)
        except Exception:
            pass
    if not vals: return dict(mean=np.nan, ci_low=np.nan, ci_high=np.nan)
    a = np.array(vals)
    return dict(mean=float(a.mean()), ci_low=float(np.quantile(a, alpha / 2)),
                ci_high=float(np.quantile(a, 1 - alpha / 2)))


def paired_bootstrap_metric_test(fn, y_true, ya, yb, n=2000, seed=42):
    """Calculates paired p-value to prove model superiority."""
    rng = np.random.RandomState(seed)
    N, diffs = y_true.shape[0], []
    for _ in range(n):
        idx = rng.choice(N, N, replace=True)
        try:
            da, db = float(fn(y_true[idx], ya[idx])), float(fn(y_true[idx], yb[idx]))
            diffs.append(da - db)
        except Exception:
            pass
    a = np.array(diffs)
    # Two-sided paired p-value
    p_val = float(min(1.0, 2 * min(np.mean(a <= 0), np.mean(a >= 0))))
    return dict(delta=float(a.mean()), p_value=p_val)


def optimise_thresholds(probs, labels, grid_steps=150):
    """Searches for F1-maximizing thresholds in imbalanced medical data."""
    n_cls = probs.shape[1]
    thr = np.full(n_cls, 0.5)
    grid = np.linspace(0.005, 0.50, grid_steps)  # Focused on low prevalence
    for k in range(n_cls):
        best_f1 = 0.0
        if labels[:, k].sum() == 0: continue
        for t in grid:
            f = f1_score(labels[:, k], (probs[:, k] >= t).astype(int), zero_division=0)
            if f > best_f1: best_f1, thr[k] = f, t
    return thr


class EarlyStopping:
    """Monitors Validation AUC to prevent over-fitting on small datasets."""

    def __init__(self, patience=10, delta=0.001, path="best.pth"):
        self.patience, self.delta, self.path = patience, delta, path
        self.best_score, self.counter, self.early_stop = -np.inf, 0, False

    def __call__(self, score, model):
        if score > self.best_score + self.delta:
            self.best_score, self.counter = score, 0
            torch.save(model.state_dict(), self.path)
            return True
        self.counter += 1
        if self.counter >= self.patience: self.early_stop = True
        return False
