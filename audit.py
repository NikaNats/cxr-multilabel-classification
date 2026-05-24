# -*- coding: utf-8 -*-
from __future__ import annotations

import gc
import hashlib
import warnings
from typing import Any, Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

import cupy as cp
import cucim.skimage.filters as cur_filters
from scipy.stats import chi2_contingency, entropy, ks_2samp, mannwhitneyu
from sklearn.metrics import matthews_corrcoef, mutual_info_score

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

COLOUR_CYCLE = [
    Colour.BLUE, Colour.ORANGE, Colour.GREEN,
    Colour.SKY, Colour.PURPLE, Colour.YELLOW,
]

_MM_TO_IN = 1.0 / 25.4
DOUBLE_COL_IN = 183 * _MM_TO_IN
MAX_HEIGHT_IN = 230 * _MM_TO_IN
FIGURE_DPI = 450

_NATURE_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6,
    "axes.labelsize": 6,
    "axes.titlesize": 7,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.5,
    "lines.linewidth": 0.75,
    "patch.linewidth": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.top": False,
    "ytick.right": False,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 5,
    "legend.title_fontsize": 5.5,
    "legend.frameon": False,
    "legend.handlelength": 1.2,
    "legend.handletextpad": 0.4,
    "figure.dpi": FIGURE_DPI,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": FIGURE_DPI,
    "savefig.bbox": "tight",
}

def configure_nature_style() -> None:
    plt.rcParams.update(_NATURE_RCPARAMS)
    sns.set_theme(style="white", palette=COLOUR_CYCLE, rc=_NATURE_RCPARAMS)

try:
    from IPython.display import display
except ImportError:
    def display(obj: Any) -> None:
        if hasattr(obj, "to_string"):
            print(obj.to_string())
        else:
            print(obj)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ForensicDatasetAuditor:
    """Forensic Auditor for Medical Imaging Datasets with Hardware Acceleration."""

    IMPLAUSIBLE_PAIRS = [
        ('Emphysema', 'Consolidation'),
        ('Pneumothorax', 'Effusion'),
    ]

    def __init__(
        self, 
        dataset: Any, 
        split_name: str = "Train", 
        n_bins_mi: int = 50, 
        chunk_size: int = 2000
    ):
        self.dataset = dataset
        self.split_name = split_name
        self.n_bins_mi = n_bins_mi
        self.chunk_size = chunk_size

        self.labels = dataset.labels.astype(np.int32)
        self.class_names = [dataset.info['label'][str(i)] for i in range(len(dataset.info['label']))]
        self.n_samples = len(self.labels)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Initialized Auditor on device: {self.device}")

        if self.n_samples > 0:
            first_image = dataset.imgs[0]
            self.h, self.w = first_image.shape[0], first_image.shape[1]
        else:
            self.h, self.w = 0, 0

        self.n_classes = len(self.class_names)
        self.results: dict[str, Any] = {}

    def _chunked_image_generator(
        self, indices: np.ndarray | list[int] | None = None
    ) -> Generator[torch.Tensor, None, None]:
        target_indices = np.arange(self.n_samples) if indices is None else np.array(indices)
        for i in range(0, len(target_indices), self.chunk_size):
            batch_idx = target_indices[i : i + self.chunk_size]
            chunk = torch.from_numpy(self.dataset.imgs[batch_idx]).to(self.device, dtype=torch.float32) / 255.0
            yield chunk

    def audit_information_theory(self) -> None:
        """Calculates information-theoretic metrics with chunked streaming."""
        print("[*] Auditing Information Theory (Chunked)...")
        img_means_list = []
        img_stds_list = []

        with torch.no_grad():
            for chunk in self._chunked_image_generator():
                mean = torch.mean(chunk, dim=(1, 2))
                std = torch.std(chunk, dim=(1, 2))
                img_means_list.append(mean.cpu().numpy())
                img_stds_list.append(std.cpu().numpy())

        img_means = np.concatenate(img_means_list)
        img_stds = np.concatenate(img_stds_list)

        n_bins = min(self.n_bins_mi, max(10, int(np.sqrt(self.n_samples / 5))))
        bins_mean = np.histogram_bin_edges(img_means, bins=n_bins)
        bins_std = np.histogram_bin_edges(img_stds, bins=n_bins)
        q_means = np.digitize(img_means, bins_mean)
        q_stds = np.digitize(img_stds, bins_std)

        q_joint = q_means * (n_bins + 2) + q_stds

        mi_mean_scores, mi_std_scores, mi_joint_scores = [], [], []
        for c in range(self.n_classes):
            mi_mean_scores.append(mutual_info_score(self.labels[:, c], q_means))
            mi_std_scores.append(mutual_info_score(self.labels[:, c], q_stds))
            mi_joint_scores.append(mutual_info_score(self.labels[:, c], q_joint))

        label_tuples = [tuple(row) for row in self.labels]
        _, counts = np.unique(label_tuples, axis=0, return_counts=True)
        j_entropy = entropy(counts, base=2)

        label_entropy = [entropy(np.bincount(self.labels[:, c], minlength=2), base=2) for c in range(self.n_classes)]
        nmi_scores = [mi_joint_scores[c] / max(label_entropy[c], 1e-10) for c in range(self.n_classes)]

        self.results.update({
            'mi_mean': mi_mean_scores, 'mi_std': mi_std_scores, 'mi_joint': mi_joint_scores,
            'nmi': nmi_scores, 'joint_entropy': j_entropy, 'q_means': q_means, 'q_stds': q_stds,
            'img_means_cached': img_means, 'img_stds_cached': img_stds
        })
        gc.collect()

    def audit_spatial_morphology(self) -> None:
        """Accelerated Spatial Morphology using GPU PyTorch and cuCIM filters."""
        print("[*] Auditing Spatial Morphology (PyTorch & cuCIM Accelerated)...")
        sum_img = torch.zeros((self.h, self.w), dtype=torch.float64, device=self.device)
        sum_sq_img = torch.zeros((self.h, self.w), dtype=torch.float64, device=self.device)

        with torch.no_grad():
            for chunk in self._chunked_image_generator():
                sum_img += torch.sum(chunk, dim=0)
                sum_sq_img += torch.sum(chunk ** 2, dim=0)

        global_mean = (sum_img / self.n_samples).cpu().numpy().astype(np.float32)
        global_var = (sum_sq_img / self.n_samples) - (torch.tensor(global_mean, device=self.device) ** 2)
        std_heatmap = torch.sqrt(torch.clamp(global_var, min=0.0)).cpu().numpy().astype(np.float32)

        diff_maps = {}
        for c, name in enumerate(self.class_names):
            class_indices = np.where(self.labels[:, c] == 1)[0]
            n_pos = len(class_indices)
            if n_pos >= 10:
                class_sum = torch.zeros((self.h, self.w), dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    for chunk in self._chunked_image_generator(indices=class_indices):
                        class_sum += torch.sum(chunk, dim=0)
                class_mean = (class_sum / n_pos).cpu().numpy()
                diff_maps[name] = class_mean - global_mean

        rng = np.random.RandomState(42)
        sample_idx = rng.choice(self.n_samples, min(1000, self.n_samples), replace=False)
        
        # cuCIM-Accelerated Edge Density computation on the GPU
        edge_density_accum = cp.zeros((self.h, self.w), dtype=cp.float32)
        
        for chunk in self._chunked_image_generator(indices=sample_idx):
            # Wrap PyTorch CUDA tensor as a zero-copy CuPy array
            chunk_cupy = cp.asnumpy(chunk.cpu().numpy()) if self.device.type == "cpu" else cp.asarray(chunk)
            for i in range(chunk_cupy.shape[0]):
                edge_density_accum += cur_filters.sobel(chunk_cupy[i])

        edge_density = cp.asnumpy(edge_density_accum / len(sample_idx))

        self.results.update({
            'global_mean': global_mean, 'std_heatmap': std_heatmap, 'diff_maps': diff_maps,
            'edge_density': edge_density
        })
        gc.collect()

    def audit_clinical_logic(self) -> None:
        print("[*] Auditing Clinical Logic...")
        n = self.n_classes
        phi_matrix = np.zeros((n, n))
        cond_prevalence = np.zeros((n, n))
        cramers_v_matrix = np.zeros((n, n))

        labels_t = torch.from_numpy(self.labels).to(self.device, dtype=torch.float32)
        pos_counts = torch.sum(labels_t, dim=0)

        for i in range(n):
            for j in range(n):
                phi_matrix[i, j] = matthews_corrcoef(self.labels[:, i], self.labels[:, j])
                denom = pos_counts[j].item()
                if denom > 0:
                    cond_prevalence[i, j] = torch.sum(labels_t[:, i] * labels_t[:, j]).item() / denom

                if i != j:
                    ct = pd.crosstab(self.labels[:, i], self.labels[:, j])
                    if ct.shape == (2, 2):
                        chi2, _, _, _ = chi2_contingency(ct, correction=True)
                        cramers_v_matrix[i, j] = np.sqrt(chi2 / self.n_samples)

        cardinality = np.sum(self.labels, axis=1)
        self.results.update({
            'phi': phi_matrix, 'cond_prev': cond_prevalence, 'cramers_v': cramers_v_matrix,
            'cardinality': cardinality, 'mean_card': float(cardinality.mean()),
            'std_card': float(cardinality.std()), 'max_card': int(cardinality.max()),
        })

    def audit_safety_integrity(self) -> None:
        print("[*] Auditing Safety & Integrity...")
        image_entropies = np.zeros(self.n_samples)
        
        idx = 0
        with torch.no_grad():
            for chunk in self._chunked_image_generator():
                for i in range(chunk.shape[0]):
                    img_np = chunk[i].cpu().numpy()
                    hist, _ = np.histogram(img_np, bins=30, density=True)
                    image_entropies[idx] = entropy(hist + 1e-10)
                    idx += 1

        img_means = self.results['img_means_cached']
        img_stds = self.results['img_stds_cached']
        snrs = img_means / (img_stds + 1e-8)

        low_snr_mask = snrs < np.percentile(snrs, 1)
        low_snr_count = int(low_snr_mask.sum())

        hashes = set()
        dup_indices = []
        for i in range(self.n_samples):
            h = hashlib.blake2b(self.dataset.imgs[i].tobytes(), digest_size=16).hexdigest()
            if h in hashes:
                dup_indices.append(i)
            hashes.add(h)

        self.results.update({
            'img_entropy': image_entropies, 'snr_dist': snrs, 'low_snr_count': low_snr_count,
            'duplicates': len(dup_indices), 'dup_indices': dup_indices, 'n_unique_hash': len(hashes),
        })
        gc.collect()

    @staticmethod
    def compute_image_hashes(dataset: Any, digest_size: int = 16) -> set[str]:
        imgs = dataset.imgs
        hashes = set()
        for i in range(len(imgs)):
            h = hashlib.blake2b(imgs[i].tobytes(), digest_size=digest_size).hexdigest()
            hashes.add(h)
        return hashes

    def audit_cross_split_leakage(self, other_datasets: dict[str, Any]) -> dict[str, int]:
        print(f"\n[*] Cross-Split Image Leakage Audit ({self.split_name} vs others)...")
        self_hashes = self.compute_image_hashes(self.dataset)
        leakage_report = {}

        for name, ds in other_datasets.items():
            other_hashes = self.compute_image_hashes(ds)
            overlap = self_hashes.intersection(other_hashes)
            leakage_report[name] = len(overlap)
            if len(overlap) > 0:
                print(f"     [!] LEAKAGE DETECTED: {len(overlap)} images shared with {name}!")
            else:
                print(f"     [✓] CLEAN: 0 image overlaps with {name}.")
            del other_hashes
            gc.collect()

        self.results['cross_split_leakage'] = leakage_report
        del self_hashes
        gc.collect()
        return leakage_report

    def audit_localized_structure(self, grid_size: int = 2) -> dict[str, Any]:
        print(f"\n[*] Localized Spatial Audit ({grid_size}x{grid_size} grid)...")
        ph, pw = self.h // grid_size, self.w // grid_size
        quadrant_info = {}

        with torch.no_grad():
            for i in range(grid_size):
                for j in range(grid_size):
                    y0, y1 = i * ph, (i + 1) * ph
                    x0, x1 = j * pw, (j + 1) * pw
                    
                    patch_means_list = []
                    patch_stds_list = []
                    
                    for chunk in self._chunked_image_generator():
                        patches = chunk[:, y0:y1, x0:x1]
                        patch_means_list.append(torch.mean(patches, dim=(1, 2)).cpu().numpy())
                        patch_stds_list.append(torch.std(patches, dim=(1, 2)).cpu().numpy())
                    
                    patch_means = np.concatenate(patch_means_list)
                    patch_stds = np.concatenate(patch_stds_list)
                    
                    patch_feat = np.round(patch_means * 50).astype(int) * 100 + np.round(patch_stds * 50).astype(int)
                    patch_mi = [mutual_info_score(self.labels[:, c], patch_feat) for c in range(self.n_classes)]
                    
                    quadrant_info[f"Q_{i}{j}"] = {
                        'mi': patch_mi, 'mean_intensity': float(np.mean(patch_means)),
                        'std_intensity': float(np.std(patch_means)),
                    }

        self.results['spatial_quadrants'] = quadrant_info
        gc.collect()
        return quadrant_info

    def audit_clinical_consistency(self) -> tuple[int, list[str]]:
        print("[*] Clinical Consistency Audit...")
        violations = 0
        violation_details = []
        label_sums = np.sum(self.labels, axis=1)

        for i in range(self.n_samples):
            if label_sums[i] > 5:
                violations += 1
                if len(violation_details) < 5:
                    active = [self.class_names[c] for c in range(self.n_classes) if self.labels[i, c] == 1]
                    violation_details.append(f"Sample {i}: {label_sums[i]} pathologies ({', '.join(active)})")

        implausible_count = 0
        for name_a, name_b in self.IMPLAUSIBLE_PAIRS:
            if name_a in self.class_names and name_b in self.class_names:
                idx_a, idx_b = self.class_names.index(name_a), self.class_names.index(name_b)
                cooccur = int(np.sum(self.labels[:, idx_a] & self.labels[:, idx_b]))
                if cooccur > 0:
                    implausible_count += cooccur
                    violation_details.append(f"Implausible: {name_a} & {name_b} in {cooccur} samples")

        empty_classes = [self.class_names[c] for c in range(self.n_classes) if self.labels[:, c].sum() == 0]

        self.results.update({'clinical_violations': violations, 'implausible_cooccurrences': implausible_count,
                             'empty_classes': empty_classes, 'violation_details': violation_details})
        return violations, violation_details

    def audit_effective_samples(self, beta: float = 0.9999) -> pd.DataFrame:
        ens_data = []
        for c in range(self.n_classes):
            n_c = int(self.labels[:, c].sum())
            ens = (1.0 - beta ** n_c) / (1.0 - beta) if beta < 1.0 else float(n_c)
            ens_data.append({'Class': self.class_names[c], 'n_pos': n_c, 'n_neg': self.n_samples - n_c,
                             'ENS': f"{ens:.1f}", 'Pos/Neg': f"1:{(self.n_samples - n_c) / max(n_c, 1):.1f}"})
        ens_df = pd.DataFrame(ens_data)
        self.results['ens_table'] = ens_df
        return ens_df

    def audit_entropy_gap(self) -> dict[str, float]:
        if 'img_entropy' not in self.results:
            self.audit_safety_integrity()
        ent = self.results['img_entropy']
        has_disease = self.labels.sum(axis=1) > 0
        ent_pos, ent_neg = ent[has_disease], ent[~has_disease]

        if len(ent_pos) > 0 and len(ent_neg) > 0:
            stat, p_val = mannwhitneyu(ent_pos, ent_neg, alternative='two-sided')
            gap, ks_stat, ks_p = float(np.mean(ent_pos) - np.mean(ent_neg)), *ks_2samp(ent_pos, ent_neg)
        else:
            stat, p_val, gap, ks_stat, ks_p = 0, 1, 0, 0, 1

        self.results['entropy_gap'] = {
            'gap': gap, 'mwu_stat': float(stat), 'mwu_p': float(p_val), 'ks_stat': float(ks_stat), 'ks_p': float(ks_p),
            'mean_pos': float(np.mean(ent_pos)) if len(ent_pos) > 0 else 0,
            'mean_neg': float(np.mean(ent_neg)) if len(ent_neg) > 0 else 0,
        }
        return self.results['entropy_gap']

    def generate_report(self) -> None:
        configure_nature_style()
        print(f"\n{'=' * 65}\n  FORENSIC DATASET AUDIT — {self.split_name} Split\n  n = {self.n_samples:,} | {self.n_classes} classes | {self.h}x{self.w} px\n{'=' * 65}")

        self.audit_information_theory()
        self.audit_spatial_morphology()
        self.audit_clinical_logic()
        self.audit_safety_integrity()
        self.audit_localized_structure(grid_size=2)
        self.audit_clinical_consistency()
        self.audit_effective_samples()
        self.audit_entropy_gap()

        print(f"\n  TABLE S2 — Effective Number of Samples (β=0.9999)\n  {'-' * 55}")
        display(self.results['ens_table'])

        eg = self.results['entropy_gap']
        print(f"\n  Entropy Gap (Disease vs Normal):")
        mwu_p_str = f"{eg['mwu_p']:.3e}" if eg['mwu_p'] < 0.001 else f"{eg['mwu_p']:.3f}".replace("0.", ".")
        ks_p_str = f"{eg['ks_p']:.3e}" if eg['ks_p'] < 0.001 else f"{eg['ks_p']:.3f}".replace("0.", ".")
        print(f"    Δ(H) = {eg['gap']:.4f}  |  MWU p = {mwu_p_str}  |  KS p = {ks_p_str}")

        cv, ip = self.results.get('clinical_violations', 0), self.results.get('implausible_cooccurrences', 0)
        print(f"\n  Clinical Logic: {cv} high-cardinality, {ip} implausible co-occurrence(s)")

        try:
            fig = plt.figure(figsize=(DOUBLE_COL_IN * 1.5, MAX_HEIGHT_IN * 1.1), layout="constrained")
            mosaic = """
            AAABB
            AAACC
            DDDEE
            FFGHH
            IIIJJ
            KKKKK
            LLLLL
            """
            ax = fig.subplot_mosaic(mosaic)
            short_names = [n[:12] for n in self.class_names]

            # A. Conditional Prevalence
            sns.heatmap(self.results['cond_prev'], ax=ax['A'], annot=True, annot_kws={"size": 4}, fmt='.2f', cmap='mako',
                        xticklabels=short_names, yticklabels=short_names, linewidths=0.3, linecolor='white', cbar=False)
            ax['A'].set_title("a. Conditional Prevalence P(Row | Col)")

            # B. Pixel-Wise Variance Map
            im_b = ax['B'].imshow(self.results['std_heatmap'], cmap='magma', aspect='equal')
            ax['B'].set_title("b. Pixel-Wise Variance")
            ax['B'].axis('off')
            plt.colorbar(im_b, ax=ax['B'], fraction=0.046, pad=0.04)

            # C. SNR Distribution
            sns.histplot(self.results['snr_dist'], ax=ax['C'], bins=40, kde=True, color=Colour.BLUE, alpha=0.7, edgecolor="white", linewidth=0.3)
            ax['C'].set_title("c. Signal-to-Noise Ratio (SNR)")
            ax['C'].set_xlabel("SNR (score)")
            ax['C'].set_ylabel("Frequency (count)")

            # D. Phi Coefficient
            mask = np.triu(np.ones_like(self.results['phi'], dtype=bool))
            sns.heatmap(self.results['phi'], ax=ax['D'], annot=True, annot_kws={"size": 4}, fmt='.2f', cmap='RdBu_r', center=0, mask=mask,
                        xticklabels=short_names, yticklabels=short_names, linewidths=0.3, cbar=False)
            ax['D'].set_title("d. Phi Coefficient (Correlation)")

            # E. Mutual Information
            sorted_idx = np.argsort(self.results['mi_joint'])[::-1]
            ax['E'].barh([short_names[i] for i in sorted_idx], [self.results['mi_joint'][i] for i in sorted_idx],
                         color=Colour.ORANGE, alpha=0.85, height=0.6)
            ax['E'].set_title("e. Joint Mutual Information")
            ax['E'].set_xlabel("Mutual Info (score)")
            ax['E'].invert_yaxis()

            # F. Global Mean Image
            ax['F'].imshow(self.results['global_mean'], cmap='bone', aspect='equal')
            ax['F'].set_title("f. Global Mean Image")
            ax['F'].axis('off')

            # G. Edge Density
            ax['G'].imshow(self.results['edge_density'], cmap='viridis', aspect='equal')
            ax['G'].set_title("g. Spatial Edge Density")
            ax['G'].axis('off')

            # H. Class Intensity Profiles
            for i in range(min(5, self.n_classes)):
                mask_c = self.labels[:, i] == 1
                if mask_c.sum() > 0:
                    sampled_intensities = []
                    for chunk in self._chunked_image_generator(indices=np.where(mask_c)[0]):
                        sampled_intensities.append(chunk.cpu().numpy().flatten())
                    flattened_pixels = np.concatenate(sampled_intensities)
                    if len(flattened_pixels) > 10000:
                        flattened_pixels = np.random.choice(flattened_pixels, 10000, replace=False)
                    sns.kdeplot(flattened_pixels, ax=ax['H'], label=short_names[i], linewidth=0.6)
            ax['H'].set_title("h. Intensity Profiles")
            ax['H'].set_xlabel("Intensity (normalized)")
            ax['H'].set_ylabel("Density (proportion)")
            ax['H'].legend(fontsize=4, ncol=2, loc="upper right")

            # I. Cardinality
            sns.histplot(self.results['cardinality'], ax=ax['I'], discrete=True, color=Colour.GREEN, edgecolor="white", linewidth=0.3)
            ax['I'].set_title("i. Label Cardinality Distribution")
            ax['I'].set_xlabel("Pathologies (count)")
            ax['I'].set_ylabel("Frequency (count)")

            # J. Localized Spatial MI map
            mi_map = np.zeros((2, 2), dtype=np.float32)
            spatial_data = self.results.get('spatial_quadrants', {})
            if spatial_data:
                for (r, c) in [(0, 0), (0, 1), (1, 0), (1, 1)]: 
                    mi_map[r, c] = np.mean(spatial_data[f'Q_{r}{c}']['mi'])
            sns.heatmap(mi_map, ax=ax['J'], annot=True, fmt='.4f', cmap='YlOrRd', cbar=False, annot_kws={"size": 5})
            ax['J'].set_title("j. Spatial MI Map (2x2)")

            # K. Cramer's V
            sns.heatmap(self.results['cramers_v'], ax=ax['K'], annot=True, annot_kws={"size": 4}, fmt='.3f', cmap='rocket_r',
                        xticklabels=short_names, yticklabels=short_names, linewidths=0.3, cbar=False)
            ax['K'].set_title("k. Cramér's V (Association)")

            # L. Diagnostic Text Summary Panel
            ax['L'].axis('off')
            summary_lines = (
                f"FORENSIC SUMMARY — n = {self.n_samples:,}\n"
                f"Duplicates: {self.results['duplicates']} | Max Joint-MI: {max(self.results['mi_joint']):.4f}"
            )
            ax['L'].text(0.5, 0.5, summary_lines, ha='center', va='center', fontsize=6, family='monospace',
                         transform=ax['L'].transAxes)

            out_pdf = f"forensic_audit_{self.split_name.lower()}.pdf"
            out_png = f"forensic_audit_{self.split_name.lower()}.png"
            fig.savefig(out_pdf, format="pdf", dpi=FIGURE_DPI, bbox_inches="tight")
            fig.savefig(out_png, format="png", dpi=FIGURE_DPI, bbox_inches="tight")
            print(f"[✓] Vector PDF Saved  → {out_pdf}")
            print(f"[✓] Raster PNG Saved  → {out_png}")
            plt.close(fig)
        except Exception as e:
            print(f"\n[!] Failed to generate publication-quality figures: {e}")
            import traceback
            traceback.print_exc()
        finally:
            plt.close('all')
            gc.collect()


# ============================================================
# AUDIT EXECUTION WRAPPER
# ============================================================
def execute_forensic_audit(train_dataset: Any, val_dataset: Any, test_dataset: Any, chunk_size: int = 2000) -> None:
    print("\n" + "=" * 65)
    print("  EXECUTING FORENSIC DATASET AUDIT")
    print("=" * 65)

    auditor_for_leakage = ForensicDatasetAuditor(train_dataset, split_name="Train", chunk_size=chunk_size)
    leakage = auditor_for_leakage.audit_cross_split_leakage({'Val': val_dataset, 'Test': test_dataset})

    auditor_val_leakage = ForensicDatasetAuditor(val_dataset, split_name="Val", chunk_size=chunk_size)
    leakage_vt = auditor_val_leakage.audit_cross_split_leakage({'Test': test_dataset})

    total_leakage = sum(leakage.values()) + sum(leakage_vt.values())
    print(f"\n  {'=' * 50}")
    print(f"  LEAKAGE VERDICT: {'[✓] CLEAN (Strictly Separated)' if total_leakage == 0 else f'[!] CONTAMINATED ({total_leakage} overlap nodes detected)'}")
    print(f"  {'=' * 50}")

    del auditor_for_leakage, auditor_val_leakage
    gc.collect()

    for ds, name in zip([train_dataset, val_dataset, test_dataset], ["Train", "Val", "Test"]):
        print(f"\n--- Starting detailed audit for {name} split ---")
        auditor = ForensicDatasetAuditor(ds, split_name=name, chunk_size=chunk_size)
        auditor.generate_report()
        del auditor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"--- Finished detailed audit for {name} split ---\n")