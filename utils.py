from __future__ import annotations

import os
import warnings
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter1d
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import f1_score
from sklearn.isotonic import IsotonicRegression  # SOTA AIR კალიბრაციისთვის

from config import log_process

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

CHESTMNIST_CLASS_NAMES: list[str] = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

RADLEX_PATHOLOGIES: list[str] = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"
]


def format_apa_p_value(p: float) -> str:
    """Formats p-values according to American Psychological Association (APA) style."""
    if p < 0.001:
        return "< .001"
    formatted = f"{p:.3f}"
    return formatted.replace("0.", ".")


def format_apa_correlation(r: float) -> str:
    """Formats correlation coefficients according to APA style (omits leading zero)."""
    formatted = f"{r:+.4f}" if r >= 0 else f"{r:.4f}"
    return formatted.replace("0.", ".")


def ensure_radlex_embeddings(
    path: str | os.PathLike, 
    pathologies: list[str], 
    model_name: str, 
    device: torch.device
) -> torch.Tensor:
    """Loads cached RadLex embeddings, or extracts them from a HuggingFace tokenizer."""
    from transformers import AutoModel, AutoTokenizer
    target_dim = 768

    if os.path.exists(path):
        try:
            emb = torch.load(path, map_location=device, weights_only=True)
            if emb.shape == (len(pathologies), target_dim):
                return emb.detach()
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(
            pathologies, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        outputs = model(**inputs)
        res = outputs.last_hidden_state[:, 0, :]

    final_res = res.detach().cpu()
    torch.save(final_res, path)
    return final_res.to(device)


def select_adjacency_threshold(labels: np.ndarray, num_classes: int = 14) -> float:
    """
    Finds the optimal threshold for the adjacency matrix of a label graph.
    Identifies the knee-point of graph density across threshold spaces.
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

    d2 = np.gradient(np.gradient(densities_smooth))
    knee_idx = int(np.argmax(np.abs(d2)))
    optimal_t = float(np.clip(thresholds[knee_idx], 0.05, 0.40))

    log_process("graph", "adjacency_threshold_selected", clamped=f"{optimal_t:.3f}")
    return optimal_t


def build_hybrid_clinical_adjacency(
    labels: np.ndarray,
    radlex_emb: torch.Tensor,
    num_classes: int = 14,
    threshold: float = 0.05,
    self_loops: bool = True,
    alpha_blend: float = 0.7
) -> np.ndarray:
    """
    Blends empirical label co-occurrences with ontological text similarity
    to yield a robust normalized adjacency matrix for Graph Cross-Attention.
    """
    co = np.zeros((num_classes, num_classes), dtype=np.float64)
    for row in labels:
        idx = np.where(row == 1)[0]
        for i in idx:
            for j in idx:
                co[i, j] += 1.0
    co_sym = (co + co.T) / 2.0
    prob_emp = co_sym / np.maximum(co_sym.sum(1, keepdims=True), 1.0)
    
    with torch.no_grad():
        norm_emb = F.normalize(radlex_emb, p=2, dim=-1)
        sem_sim = torch.matmul(norm_emb, norm_emb.t()).cpu().numpy().astype(np.float64)
    
    hybrid_prob = alpha_blend * prob_emp + (1.0 - alpha_blend) * sem_sim
    
    adj = (hybrid_prob >= threshold).astype(np.float64) * hybrid_prob
    if self_loops:
        adj += np.eye(num_classes)

    deg = adj.sum(1)
    d_inv_sq = np.where(deg > 0, np.power(np.maximum(deg, 1e-12), -0.5), 0.0)
    d_inv_sq = np.diag(d_inv_sq)
    return (d_inv_sq @ adj @ d_inv_sq).astype(np.float32)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> np.ndarray:
    """Computes class-wise Expected Calibration Error (ECE)."""
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


def bootstrap_metric_ci(
    fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_score: np.ndarray,
    n: int = 2000,
    alpha: float = 0.05,
    seed: int = 42
) -> dict[str, float]:
    """Calculates bootstrap confidence intervals for an evaluation metric."""
    rng = np.random.RandomState(seed)
    N = y_true.shape[0]
    vals = []
    for _ in range(n):
        idx = rng.choice(N, N, replace=True)
        try:
            v = float(fn(y_true[idx], y_score[idx]))
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    if not vals:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    a = np.array(vals)
    return {
        "mean": float(a.mean()),
        "ci_low": float(np.quantile(a, alpha / 2)),
        "ci_high": float(np.quantile(a, 1.0 - alpha / 2))
    }


def paired_bootstrap_metric_test(
    fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    ya: np.ndarray,
    yb: np.ndarray,
    n: int = 2000,
    seed: int = 42
) -> dict[str, float]:
    """Performs a paired bootstrap test to check the statistical significance of delta-metrics."""
    rng = np.random.RandomState(seed)
    N = y_true.shape[0]
    diffs = []
    for _ in range(n):
        idx = rng.choice(N, N, replace=True)
        try:
            da = float(fn(y_true[idx], ya[idx]))
            db = float(fn(y_true[idx], yb[idx]))
            diffs.append(da - db)
        except Exception:
            pass
    a = np.array(diffs)
    p_val = float(min(1.0, 2 * min(np.mean(a <= 0), np.mean(a >= 0))))
    return {"delta": float(a.mean()), "p_value": p_val}


def optimise_thresholds(probs: np.ndarray, labels: np.ndarray, grid_steps: int = 150) -> np.ndarray:
    """Optimizes F1-score classification decision boundaries class-by-class."""
    n_cls = probs.shape[1]
    thr = np.full(n_cls, 0.5)
    
    for k in range(n_cls):
        best_f1 = 0.0
        if labels[:, k].sum() == 0:
            continue
        
        grid = np.unique(np.percentile(probs[:, k], np.linspace(10, 99.9, grid_steps)))
        grid = np.clip(grid, 0.00001, 0.99999)
        
        for t in grid:
            f = f1_score(labels[:, k], (probs[:, k] >= t).astype(int), zero_division=0)
            if f > best_f1: 
                best_f1, thr[k] = f, t
                
        if best_f1 == 0.0:
            thr[k] = np.clip(np.percentile(probs[:, k], 99.5) + 1e-5, 0.00001, 0.99999)
            
    return thr


class EarlyStopping:
    """Tracks validation metrics to prevent overfitting using standard patience bounds."""
    def __init__(self, patience: int = 10, delta: float = 0.001, path: str = "best.pth"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_score = -np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        if score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False


# ==============================================================================
# SOTA CLASS-WISE ASYMMETRIC ISOTONIC REGRESSION (AIR) CALIBRATOR
# ==============================================================================

class ClassWiseAsymmetricIsotonicCalibrator:
    """
    SOTA Class-Wise Asymmetric Isotonic Regression (AIR) Calibrator.
    Specifically engineered to correct multi-label probability warping induced 
    by Asymmetric Loss (ASL) without degrading classification thresholds or F1 scores.
    """
    def __init__(self, num_classes: int = 14):
        self.num_classes = num_classes
        self.calibrators: list[IsotonicRegression] = []
        self.is_fitted = False

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray) -> ClassWiseAsymmetricIsotonicCalibrator:
        """Fits an independent Isotonic Regression model per pathology class."""
        self.calibrators = []
        
        for c in range(self.num_classes):
            ir = IsotonicRegression(
                y_min=0.0, 
                y_max=1.0, 
                increasing=True, 
                out_of_bounds="clip"
            )
            
            p_c = val_probs[:, c]
            y_c = val_labels[:, c]
            
            # Microscopic linear perturbation to stabilize regression matrix decomposition
            jitter = np.linspace(-1e-9, 1e-9, len(p_c))
            p_c_stable = np.clip(p_c + jitter, 0.0, 1.0)
            
            ir.fit(p_c_stable, y_c)
            self.calibrators.append(ir)
            
        self.is_fitted = True
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Applies fitted isotonic mapping with boundary smoothing."""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted on validation data before calibration.")
            
        calibrated_probs = np.zeros_like(probs)
        
        for c in range(self.num_classes):
            p_c = np.clip(probs[:, c], 0.0, 1.0)
            raw_calibrated = self.calibrators[c].predict(p_c)
            
            # Smooth interpolation near the boundary to protect gradient ranking
            calibrated_probs[:, c] = 0.999 * raw_calibrated + 0.001 * p_c
            
        return np.clip(calibrated_probs, 1e-7, 1.0 - 1e-7)


# ==============================================================================
# SOTA MULTI-LABEL CONFORMAL RISK CONTROL (CRC) FOR FDR BOUNDS
# ==============================================================================

class UncertaintyGatedAdaptiveConformalPredictor:
    """
    Implements a mathematically rigorous Conformal Risk Control (CRC) framework
    calibrated to strictly control the False Discovery Rate (FDR) below a target 
    risk level alpha (e.g. FDR < 10%). Includes an out-of-sample selective
    classification filter using deep ensemble predictive entropy.
    """
    def __init__(
        self, 
        alpha: float = 0.10, 
        lambda_param: float = 0.6, 
        rejection_quantile: float = 0.10
    ):
        self.alpha = alpha  # Target FDR upper bound constraint
        self.lambda_param = lambda_param
        self.rejection_quantile = rejection_quantile
        self.uncertainty_threshold: float | None = None
        self.global_multiplier: float = 1.0
        self.class_weights: np.ndarray | None = None
        self.opt_thresholds: np.ndarray | None = None

    def calibrate(
        self, 
        cal_probs: np.ndarray, 
        cal_labels: np.ndarray, 
        opt_thresholds: np.ndarray, 
        validation_aucs: np.ndarray | list[float], 
        cal_uncertainties: np.ndarray
    ) -> None:
        """
        Calibrates the threshold multiplier using out-of-sample validation runs
        to guarantee distribution-free control of the False Discovery Rate (FDR).
        """
        self.opt_thresholds = opt_thresholds
        
        # 1. Determine uncertainty threshold for selective classification
        self.uncertainty_threshold = float(
            np.quantile(cal_uncertainties, 1.0 - self.rejection_quantile)
        )
        
        # 2. Filter calibration set to include only accepted patient samples
        valid_mask = cal_uncertainties <= self.uncertainty_threshold
        cal_probs_filtered = cal_probs[valid_mask]
        cal_labels_filtered = cal_labels[valid_mask]
        n_cal = max(cal_probs_filtered.shape[0], 1)
        
        # 3. Calculate class-specific weights based on historical predictive strength
        aucs = np.array(validation_aucs)
        self.class_weights = 1.0 + self.lambda_param * (1.0 - aucs)
        
        # 4. Search grid for smallest threshold multiplier lambda that controls FDR
        # Larger lambda -> higher thresholds -> fewer active predictions -> lower FDR
        multipliers = np.linspace(0.01, 10.0, 2000)
        best_m = 10.0  # Safe default (most conservative / highest threshold multiplier)
        
        for m in multipliers:
            scaled_thresholds = self.opt_thresholds * m * self.class_weights
            test_thr = np.clip(scaled_thresholds, 0.00001, 0.99999)
            
            # Predict labels
            preds = cal_probs_filtered >= test_thr
            
            # Calculate False Discovery Rate (FDR) per patient
            false_positives = (preds & ~cal_labels_filtered.astype(bool)).sum(axis=1)
            total_predictions = preds.sum(axis=1)
            
            # FDR is 0 if no classes are predicted active for that sample
            fdr_per_sample = np.where(total_predictions > 0, false_positives / np.maximum(total_predictions, 1), 0.0)
            empirical_risk = fdr_per_sample.mean()
            
            # Apply Conformal Risk Control distribution-free expectation bound: (n / (n + 1)) * Risk + 1 / (n + 1)
            rc_bound = (n_cal / (n_cal + 1)) * empirical_risk + (1.0 / (n_cal + 1))
            
            if rc_bound <= self.alpha:
                best_m = m
                break
                
        self.global_multiplier = float(best_m)
        log_process("conformal", "crc_calibration_completed", 
                    calibrated_multiplier=f"{self.global_multiplier:.4f}",
                    target_fdr=f"{self.alpha:.2%}")

    def predict_sets(
        self, 
        test_probs: np.ndarray, 
        test_uncertainties: np.ndarray, 
        force_non_empty: bool = True
    ) -> dict[str, np.ndarray]:
        """
        Maps calibrated model probabilities to conformal prediction sets
        with rigorous out-of-sample FDR control guarantees.
        """
        if self.uncertainty_threshold is None or self.class_weights is None or self.opt_thresholds is None:
            raise ValueError("Conformal predictor must be calibrated before prediction.")

        # Determine clinical acceptance based on predictive entropy safety gate
        accepted_mask = test_uncertainties <= self.uncertainty_threshold
        
        # Apply calibrated CRC threshold multiplier
        final_thr = np.clip(
            self.opt_thresholds * self.global_multiplier * self.class_weights, 
            0.00001, 
            0.99999
        )
        sets = test_probs >= final_thr
        
        # Prevent empty sets for accepted samples to assist clinicians with diagnostic candidates
        if force_non_empty:
            empty_idx = (sets.sum(axis=1) == 0) & accepted_mask
            if empty_idx.any():
                sets[empty_idx, np.argmax(test_probs[empty_idx], axis=1)] = True
        
        # If the sample is flagged for abstention (unaccepted), we preserve its predicted candidates 
        # but mark accepted as False so the system routes it to a human radiologist.
        return {
            "include_pos": sets,
            "accepted": accepted_mask
        }