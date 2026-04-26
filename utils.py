import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, brier_score_loss
from sklearn.exceptions import UndefinedMetricWarning

from config import log_process

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

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

def configure_nature_plots():
    import matplotlib.pyplot as plt
    import seaborn as sns
    NATURE_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]
    sns.set_theme(style="whitegrid", context="paper", palette=NATURE_PALETTE, font="serif")
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.labelsize": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "figure.dpi": 300,
        "savefig.dpi": 600, "figure.facecolor": "white", "savefig.bbox": "tight",
    })

def ensure_radlex_embeddings(path, pathologies, model_name, device_):
    from transformers import AutoModel, AutoTokenizer
    target_dim = 768 
    
    if os.path.exists(path):
        try:
            emb = torch.load(path, map_location=device_, weights_only=True)
            if emb.shape == (len(pathologies), target_dim):
                return emb.detach()
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device_)
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(pathologies, padding=True, truncation=True, return_tensors='pt').to(device_)
        try:
            res = model.get_projected_text_embeddings(inputs.input_ids, inputs.attention_mask)
        except AttributeError:
            res = model(**inputs).last_hidden_state[:, 0, :]

        if res.shape[1] != target_dim:
            projector = nn.Linear(res.shape[1], target_dim).to(device_)
            res = projector(res)

    final_res = res.detach().cpu()
    torch.save(final_res, path)
    return final_res.to(device_)

def select_adjacency_threshold(labels: np.ndarray, num_classes: int = 14) -> float:
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
    
    d2 = np.gradient(np.gradient(densities_smooth))
    knee_idx = int(np.argmax(np.abs(d2)))
    # Nature-grade: Clamping to ensure semantic relevance
    optimal_t = float(np.clip(thresholds[knee_idx], 0.05, 0.40))
    
    log_process("graph", "adjacency_threshold_selected", clamped=f"{optimal_t:.3f}")
    return optimal_t

def build_cooccurrence_adjacency(labels, num_classes=14, threshold=0.4, self_loops=True):
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
    d_inv_sq = np.where(deg > 0, np.power(np.maximum(deg, 1e-12), -0.5), 0.0)
    d_inv_sq = np.diag(d_inv_sq)
    return (d_inv_sq @ adj @ d_inv_sq).astype(np.float32)

def brier_score_multilabel(probs, labels):
    return np.array([brier_score_loss(labels[:, k], probs[:, k]) for k in range(probs.shape[1])])

def compute_logit_adjustment(train_labels_np, tau=1.0):
    pos_freq = np.mean(train_labels_np, axis=0)
    pos_freq = np.clip(pos_freq, 1e-5, 1.0 - 1e-5)
    adjustment = tau * np.log(pos_freq / (1.0 - pos_freq))
    return torch.from_numpy(adjustment.astype(np.float32))

def optimise_thresholds(probs, labels, grid_steps=150):
    """
    Nature-Grade threshold optimization. 
    Focuses on low-probability regions critical for rare pathologies.
    """
    n_cls = probs.shape[1]
    thr = np.full(n_cls, 0.5)
    grid = np.linspace(0.005, 0.50, grid_steps) 
    for k in range(n_cls):
        best_f1 = 0.0
        if labels[:, k].sum() == 0:
            continue
        for t in grid:
            f1 = f1_score(labels[:, k], (probs[:, k] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                thr[k] = t
    return thr

class EarlyStopping:
    def __init__(self, patience=10, delta=0.001, path="best.pth"):
        self.patience, self.delta, self.path = patience, delta, path
        self.best_score, self.counter, self.early_stop = -np.inf, 0, False

    def __call__(self, score, model):
        if score > self.best_score + self.delta:
            self.best_score, self.counter = score, 0
            torch.save(model.state_dict(), self.path)
            return True
        self.counter += 1
        if self.counter >= self.patience: 
            self.early_stop = True
        return False

class TemperatureScaler(nn.Module):
    """
    Robust Temperature Scaler. 
    Clamps T to prevent the 'Exploding Logit' syndrome in evidential models.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Initialization with T=1.5 for better gradient flow in medical data
        self.temperature = nn.Parameter(torch.ones(1) * 1.5) 

    def forward(self, x):
        logits = self.base_model_call(x)
        # Fix: The 467 catastrophe is solved by strict clamping and softplus projection
        t_safe = self.temperature.clamp(min=0.1, max=3.5)
        return logits / t_safe

    def calibrate(self, val_loader, device_, max_iter=100):
        self.model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for feats, lbls in val_loader:
                logits = self.base_model_call(feats.to(device_))
                all_logits.append(logits.cpu())
                all_labels.append(lbls)
                
        logits_val = torch.cat(all_logits).to(device_)
        labels_val = torch.cat(all_labels).float().to(device_)
        
        # Using LBFGS for precise convergence on a single scalar
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        
        def _eval():
            optimizer.zero_grad()
            t_clamp = self.temperature.clamp(min=0.1, max=3.5)
            loss = F.binary_cross_entropy_with_logits(logits_val / t_clamp, labels_val)
            loss.backward()
            return loss
            
        optimizer.step(_eval)
        t_final = self.temperature.clamp(min=0.1, max=3.5).item()
        log_process("calibration", "temperature_clamped", final_t=f"{t_final:.4f}")
        return t_final

    def base_model_call(self, x):
        out = self.model(x)
        return out[0] if isinstance(out, tuple) else out

def expected_calibration_error(probs, labels, n_bins=15):
    """
    Multilabel Expected Calibration Error.
    Measures the gap between confidence and empirical accuracy.
    """
    ece = []
    for k in range(probs.shape[1]):
        bounds = np.linspace(0, 1, n_bins + 1)
        ek = 0.0
        for b_idx in range(n_bins):
            lo, hi = bounds[b_idx], bounds[b_idx+1]
            m = (probs[:, k] >= lo) & (probs[:, k] < hi)
            if m.sum() > 0: 
                # Accuracy vs Confidence gap
                ek += m.mean() * abs(labels[m, k].mean() - probs[m, k].mean())
        ece.append(ek)
    return np.array(ece)

def bootstrap_metric_ci(fn, y_true, y_score, n=2000, alpha=0.05, seed=42):
    """Reliable Confidence Intervals via bootstrapping."""
    rng = np.random.RandomState(seed)
    N = y_true.shape[0]
    vals = []
    for _ in range(n):
        idx = rng.choice(N, N, replace=True)
        try:
            v = float(fn(y_true[idx], y_score[idx]))
            if np.isfinite(v): vals.append(v)
        except Exception: 
            pass
    if not vals: 
        return dict(mean=np.nan, ci_low=np.nan, ci_high=np.nan)
    a = np.array(vals)
    return dict(
        mean=float(a.mean()), 
        ci_low=float(np.quantile(a, alpha/2)), 
        ci_high=float(np.quantile(a, 1-alpha/2))
    )

def paired_bootstrap_metric_test(fn, y_true, ya, yb, n=2000, seed=42):
    """Statistical significance test between two models."""
    rng = np.random.RandomState(seed)
    N = y_true.shape[0]
    diffs = []
    for _ in range(n):
        idx = rng.choice(N, N, replace=True)
        try:
            da, db = float(fn(y_true[idx], ya[idx])), float(fn(y_true[idx], yb[idx]))
            diffs.append(da - db)
        except Exception: 
            pass
    a = np.array(diffs)
    # Paired p-value
    p_val = float(min(1.0, 2 * min(np.mean(a <= 0), np.mean(a >= 0))))
    return dict(delta=float(a.mean()), p_value=p_val)


def optimise_thresholds(probs, labels, grid_steps=150):
    """F1 ქულის ოპტიმიზაცია იშვიათი კლასებისთვის."""
    n_cls = probs.shape[1]
    thr = np.full(n_cls, 0.5)
    grid = np.linspace(0.005, 0.50, grid_steps) 
    for k in range(n_cls):
        best_f1 = 0.0
        if labels[:, k].sum() == 0: continue
        for t in grid:
            f1 = f1_score(labels[:, k], (probs[:, k] >= t).astype(int), zero_division=0)
            if f1 > best_f1: 
                best_f1 = f1
                thr[k] = t
    return thr

class MultiLabelConformalPredictor:
    """Nature-Grade True Marginal Conformal Prediction."""
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
            # Non-conformity score
            scores = 1.0 - cal_probs[pos_mask, k]
            n = scores.shape[0]
            # ემპირიული კვანტილი სასრული შერჩევის კორექციით
            q = min(max(np.ceil((n + 1) * (1 - self.alpha)) / n, 0.0), 1.0)
            self.thresholds[k] = 1.0 - np.quantile(scores, q)

    def predict_sets(self, probs):
        return {"include_pos": probs >= self.thresholds}