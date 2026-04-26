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

# ── NATURE-GRADE: ვაუქმებთ გაფრთხილებებს იშვიათი კლასების ბუტსტრაპირებისას ──
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
        except Exception as e:
            print(f"  [!] Failed to load existing RadLex embeddings: {e}. Regenerating...")

    print(f"[*] Generating SOTA BioViL-T embeddings (768-dim) with {model_name}...")
    log_process("radlex", "embedding_generation_started", model=model_name, target_dim=target_dim, labels=len(pathologies))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device_)
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(pathologies, padding=True, truncation=True, return_tensors='pt').to(device_)
        try:
            res = model.get_projected_text_embeddings(inputs.input_ids, inputs.attention_mask)
        except AttributeError:
            res = model(**inputs).last_hidden_state[:, 0, :]

        # თუ ემბედინგის ზომა არ ემთხვევა target_dim-ს, ვუკეთებთ პროექციას
        if res.shape[1] != target_dim:
            torch.manual_seed(42)
            projector = nn.Linear(res.shape[1], target_dim).to(device_)
            res = projector(res)

    final_res = res.detach().cpu()
    torch.save(final_res, path)
    return final_res.to(device_)

def select_adjacency_threshold(labels: np.ndarray, num_classes: int = 14, threshold_bounds: tuple = (0.05, 0.40)) -> float:
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
    optimal_t = float(thresholds[knee_idx])
    
    # Nature-grade guard: Clamp to clinically valid range
    optimal_t = float(np.clip(optimal_t, threshold_bounds[0], threshold_bounds[1]))
    
    print(f"  [Graph-Opt] Dynamic threshold (raw elbow): {thresholds[knee_idx]:.3f} → clamped to: {optimal_t:.3f}")
    log_process("graph", "adjacency_threshold_selected", raw=f"{thresholds[knee_idx]:.3f}", clamped=f"{optimal_t:.3f}")
    
    return optimal_t

def build_cooccurrence_adjacency(labels, num_classes=14, threshold=0.4, self_loops=True):
    # NATURE-GRADE FIX 1: Symmetrize raw counts BEFORE normalization
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

class EarlyStopping:
    def __init__(self, patience=7, delta=0.001, path="best.pth"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_score = -np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, score, model):
        if score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            return True
        self.counter += 1
        if self.counter >= self.patience: 
            self.early_stop = True
        return False

class TemperatureScaler(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x):
        logits = self.model(x)
        if isinstance(logits, tuple): 
            logits = logits[0]
        return logits / self.temperature.clamp(min=0.05)

    def calibrate(self, val_loader, device_, max_iter=100):
        self.model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for feats, lbls in val_loader:
                logits = self.model(feats.to(device_))
                if isinstance(logits, tuple): 
                    logits = logits[0]
                all_logits.append(logits.cpu())
                all_labels.append(lbls)
                
            calib_device = self.temperature.device
            logits_val = torch.cat(all_logits).to(calib_device)
            labels_val = torch.cat(all_labels).float().to(calib_device)

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter, line_search_fn="strong_wolfe")
        
        def _eval():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(logits_val / self.temperature.clamp(min=0.05), labels_val)
            loss.backward()
            return loss
            
        optimizer.step(_eval)
        
        T_opt = self.temperature.item()
        print(f"  ✓ Temperature scaling: T = {T_opt:.4f}")
        log_process("calibration", "temperature_scaling_fitted", temperature=f"{T_opt:.4f}")
        return T_opt

def expected_calibration_error(probs, labels, n_bins=15):
    ece = []
    for k in range(probs.shape[1]):
        bounds = np.linspace(0, 1, n_bins + 1)
        ek = 0.0
        for b_idx, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
            if b_idx == n_bins - 1:
                m = (probs[:, k] >= lo) & (probs[:, k] <= hi)
            else:
                m = (probs[:, k] >= lo) & (probs[:, k] < hi)
                
            if m.sum() > 0: 
                ek += m.mean() * abs(labels[m, k].mean() - probs[m, k].mean())
        ece.append(ek)
    return np.array(ece)

def brier_score_multilabel(probs, labels):
    return np.array([brier_score_loss(labels[:, k], probs[:, k]) for k in range(probs.shape[1])])

def compute_logit_adjustment(train_labels_np, tau=1.0):
    pos_freq = np.mean(train_labels_np, axis=0)
    pos_freq = np.clip(pos_freq, 1e-5, 1.0 - 1e-5)
    adjustment = tau * np.log(pos_freq / (1.0 - pos_freq))
    return torch.from_numpy(adjustment.astype(np.float32))

def optimise_thresholds(probs, labels, grid_steps=100):
    n_cls = probs.shape[1]
    thr = np.full(n_cls, 0.5)
    grid = np.linspace(0.05, 0.95, grid_steps)
    for k in range(n_cls):
        best = 0.0
        for t in grid:
            f1 = f1_score(labels[:, k], (probs[:, k] >= t).astype(int), zero_division=0)
            if f1 > best:
                best = f1
                thr[k] = t
    return thr

def bootstrap_metric_ci(fn, y_true, y_score, n=2000, alpha=0.05, seed=42):
    rng, N, vals = np.random.RandomState(seed), y_true.shape[0], []
    for _ in range(n):
        idx = rng.choice(N, N, replace=True)
        try:
            v = float(fn(y_true[idx], y_score[idx]))
            if np.isfinite(v): 
                vals.append(v)
        except ValueError: 
            pass
            
    if not vals: 
        return dict(mean=np.nan, ci_low=np.nan, ci_high=np.nan)
        
    a = np.array(vals)
    return dict(
        mean=float(a.mean()), 
        ci_low=float(np.quantile(a, alpha / 2)), 
        ci_high=float(np.quantile(a, 1 - alpha / 2)), 
        se=float(a.std(ddof=1)), 
        n_valid=int(a.size)
    )

def paired_bootstrap_metric_test(fn, y_true, ya, yb, n=2000, seed=42):
    rng, N, diffs = np.random.RandomState(seed), y_true.shape[0], []
    for _ in range(n):
        idx = rng.choice(N, N, replace=True)
        try:
            da, db = float(fn(y_true[idx], ya[idx])), float(fn(y_true[idx], yb[idx]))
            if np.isfinite(da) and np.isfinite(db): 
                diffs.append(da - db)
        except ValueError: 
            pass
            
    if not diffs: 
        return dict(delta=np.nan, p_value=np.nan)
        
    a = np.array(diffs)
    return dict(
        delta=float(a.mean()), 
        ci_low=float(np.quantile(a, 0.025)), 
        ci_high=float(np.quantile(a, 0.975)), 
        p_value=float(min(1.0, 2 * min(np.mean(a <= 0), np.mean(a >= 0)))), 
        n_valid=int(a.size)
    )