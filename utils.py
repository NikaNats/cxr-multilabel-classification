from __future__ import annotations

import numpy as np
import os
import torch
import torch.nn.functional as F
import warnings
from scipy.ndimage import uniform_filter1d
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.isotonic import IsotonicRegression
from typing import Callable

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
    if p < 0.001:
        return "< .001"
    formatted = f"{p:.3f}"
    return formatted.replace("0.", ".")


def format_apa_correlation(r: float) -> str:
    formatted = f"{r:+.4f}" if r >= 0 else f"{r:.4f}"
    return formatted.replace("0.", ".")


def ensure_radlex_embeddings(
        path: str | os.PathLike,
        pathologies: list[str],
        model_name: str,
        device: torch.device
) -> torch.Tensor:
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
    co = np.zeros((num_classes, num_classes), dtype=np.float64)
    for row in labels:
        idx = np.where(row == 1)[0]
        for i in idx:
            for j in idx:
                co[i, j] += 1.0

    prob = co / np.maximum(co.sum(1, keepdims=True), 1.0)

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
    co = np.zeros((num_classes, num_classes), dtype=np.float64)
    for row in labels:
        idx = np.where(row == 1)[0]
        for i in idx:
            for j in idx:
                co[i, j] += 1.0

    prob_emp = co / np.maximum(co.sum(1, keepdims=True), 1.0)

    with torch.no_grad():
        norm_emb = F.normalize(radlex_emb, p=2, dim=-1)
        sem_sim = torch.matmul(norm_emb, norm_emb.t()).cpu().numpy().astype(np.float64)

    hybrid_prob = alpha_blend * prob_emp + (1.0 - alpha_blend) * sem_sim

    hybrid_prob_sym = (hybrid_prob + hybrid_prob.T) / 2.0

    adj = (hybrid_prob_sym >= threshold).astype(np.float64) * hybrid_prob_sym
    if self_loops:
        adj += np.eye(num_classes)

    deg = adj.sum(1)
    d_inv_sq = np.where(deg > 0, np.power(np.maximum(deg, 1e-12), -0.5), 0.0)
    d_inv_sq = np.diag(d_inv_sq)
    return (d_inv_sq @ adj @ d_inv_sq).astype(np.float32)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> np.ndarray:
    ece = []
    for k in range(probs.shape[1]):
        bounds = np.linspace(0, 1, n_bins + 1)
        ek = 0.0
        for b_idx in range(n_bins):
            lo, hi = bounds[b_idx], bounds[b_idx + 1]
            if b_idx == n_bins - 1:
                m = (probs[:, k] >= lo) & (probs[:, k] <= hi)
            else:
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
    """
    Optimised Threshold Finder utilizing vectorized matrix operations.
    Fully compatible with floating-point label matrices (float32/float64).
    """
    n_samples, n_cls = probs.shape
    thr = np.full(n_cls, 0.5)

    for k in range(n_cls):
        y_true_bool = labels[:, k].astype(bool)
        total_pos = y_true_bool.sum()
        if total_pos == 0:
            continue

        grid = np.unique(np.percentile(probs[:, k], np.linspace(10, 99.9, grid_steps)))
        grid = np.clip(grid, 0.00001, 0.99999)

        preds = probs[:, k, np.newaxis] >= grid[np.newaxis, :]

        tp = (preds & y_true_bool[:, np.newaxis]).sum(axis=0)
        fp = (preds & (~y_true_bool)[:, np.newaxis]).sum(axis=0)
        fn = ((~preds) & y_true_bool[:, np.newaxis]).sum(axis=0)

        denominator = 2 * tp + fp + fn
        f1_scores = np.zeros_like(grid)
        valid_mask = denominator > 0
        f1_scores[valid_mask] = (2 * tp[valid_mask]) / denominator[valid_mask]

        best_idx = np.argmax(f1_scores)
        if f1_scores[best_idx] > 0.0:
            thr[k] = grid[best_idx]
        else:
            thr[k] = np.clip(np.percentile(probs[:, k], 99.5) + 1e-5, 0.00001, 0.99999)

    return thr


class EarlyStopping:
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


class ClassWiseAsymmetricIsotonicCalibrator:
    """
    Class-Wise Asymmetric Isotonic Regression (AIR) Calibrator.
    """

    def __init__(self, num_classes: int = 14):
        self.num_classes = num_classes
        self.calibrators: list[IsotonicRegression] = []
        self.is_fitted = False

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray) -> ClassWiseAsymmetricIsotonicCalibrator:
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

            jitter = np.linspace(-1e-9, 1e-9, len(p_c))
            p_c_stable = np.clip(p_c + jitter, 0.0, 1.0)

            ir.fit(p_c_stable, y_c)
            self.calibrators.append(ir)

        self.is_fitted = True
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted on validation data before calibration.")

        calibrated_probs = np.zeros_like(probs)
        for c in range(self.num_classes):
            p_c = np.clip(probs[:, c], 0.0, 1.0)
            raw_calibrated = self.calibrators[c].predict(p_c)
            calibrated_probs[:, c] = 0.999 * raw_calibrated + 0.001 * p_c

        return np.clip(calibrated_probs, 1e-7, 1.0 - 1e-7)


class UncertaintyGatedAdaptiveConformalPredictor:
    """
    Implements a Conformal Risk Control (CRC) framework.
    """

    def __init__(
            self,
            alpha: float = 0.10,
            lambda_param: float = 0.6,
            rejection_quantile: float = 0.10
    ):
        self.lambda_param = lambda_param
        self.rejection_quantile = rejection_quantile
        self.uncertainty_threshold: float | None = None
        self.global_multipliers: np.ndarray | None = None
        self.class_weights: np.ndarray | None = None
        self.opt_thresholds: np.ndarray | None = None

        self.alphas = np.array([
            0.25,  # Atelectasis
            0.05,  # Cardiomegaly
            0.05,  # Effusion
            0.25,  # Infiltration
            0.25,  # Mass
            0.25,  # Nodule
            0.05,  # Pneumonia
            0.05,  # Pneumothorax
            0.25,  # Consolidation
            0.05,  # Edema
            0.25,  # Emphysema
            0.30,  # Fibrosis
            0.30,  # Pleural_Thickening
            0.30  # Hernia
        ])

    def calibrate(
            self,
            cal_probs: np.ndarray,
            cal_labels: np.ndarray,
            opt_thresholds: np.ndarray,
            validation_aucs: np.ndarray | list[float],
            cal_uncertainties: np.ndarray
    ) -> None:
        self.opt_thresholds = opt_thresholds
        num_classes = cal_probs.shape[1]

        self.uncertainty_threshold = float(
            np.quantile(cal_uncertainties, 1.0 - self.rejection_quantile)
        )

        valid_mask = cal_uncertainties <= self.uncertainty_threshold
        cal_probs_filtered = cal_probs[valid_mask]
        cal_labels_filtered = cal_labels[valid_mask]
        n_cal = max(cal_probs_filtered.shape[0], 1)

        aucs = np.array(validation_aucs)
        self.class_weights = 1.0 + self.lambda_param * (1.0 - aucs)

        self.global_multipliers = np.ones(num_classes)
        multipliers = np.linspace(0.01, 10.0, 2000)

        for c in range(num_classes):
            alpha_c = self.alphas[c]
            p_c = cal_probs_filtered[:, c]
            y_c_bool = ~cal_labels_filtered[:, c].astype(bool)

            test_thresholds = np.clip(
                self.opt_thresholds[c] * multipliers * self.class_weights[c],
                0.00001,
                0.99999
            )

            preds_matrix = p_c[:, np.newaxis] >= test_thresholds[np.newaxis, :]
            fps_vector = (preds_matrix & y_c_bool[:, np.newaxis]).sum(axis=0)

            empirical_risk = fps_vector / n_cal
            rc_bounds = (n_cal / (n_cal + 1)) * empirical_risk + (1.0 / (n_cal + 1))

            satisfying_indices = np.where(rc_bounds <= alpha_c)[0]
            if len(satisfying_indices) > 0:
                best_m = float(multipliers[satisfying_indices[0]])
            else:
                best_m = 10.0

            self.global_multipliers[c] = best_m

        log_process("conformal", "class_wise_crc_calibration_completed",
                    calibrated_multipliers=[float(x) for x in np.round(self.global_multipliers, 4)])

    def predict_sets(
            self,
            test_probs: np.ndarray,
            test_uncertainties: np.ndarray,
            force_non_empty: bool = True
    ) -> dict[str, np.ndarray]:
        if self.uncertainty_threshold is None or self.class_weights is None or self.opt_thresholds is None:
            raise ValueError("Conformal predictor must be calibrated before prediction.")

        accepted_mask = test_uncertainties <= self.uncertainty_threshold

        final_thr = np.clip(
            self.opt_thresholds * self.global_multipliers * self.class_weights,
            0.00001,
            0.99999
        )
        sets = test_probs >= final_thr

        if force_non_empty:
            empty_idx = (sets.sum(axis=1) == 0) & accepted_mask
            if empty_idx.any():
                row_indices = np.where(empty_idx)[0]
                col_indices = np.argmax(test_probs[empty_idx], axis=1)
                sets[row_indices, col_indices] = True

        return {
            "include_pos": sets,
            "accepted": accepted_mask
        }
