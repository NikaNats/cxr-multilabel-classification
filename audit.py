import gc  # Add garbage collection
import hashlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import warnings
from scipy.stats import entropy, chi2_contingency, spearmanr, mannwhitneyu, ks_2samp
from skimage import filters
from sklearn.metrics import matthews_corrcoef, mutual_info_score
from typing import Dict, List, Tuple, Optional, Set

from config import GLOBAL_SEED

# IPython Display Fallback (for terminal runs)
try:
    from IPython.display import display
except ImportError:
    def display(obj):
        if hasattr(obj, "to_string"):
            print(obj.to_string())
        else:
            print(obj)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ForensicDatasetAuditor:
    """Nature-Level Forensic Auditor for Medical Imaging Datasets."""

    IMPLAUSIBLE_PAIRS = [
        ('Emphysema', 'Consolidation'),
        ('Pneumothorax', 'Effusion'),
    ]

    def __init__(self, dataset, split_name: str = "Train", n_bins_mi: int = 50):
        self.dataset = dataset  # Keep original dataset reference
        self.split_name = split_name
        self.n_bins_mi = n_bins_mi

        self.labels = dataset.labels.astype(np.int32)
        self.class_names = [dataset.info['label'][str(i)] for i in range(len(dataset.info['label']))]
        self.n_samples = len(self.labels)  # Use labels length for consistency

        # Get image dimensions from the first image without loading all to float32
        if self.n_samples > 0:
            first_image = dataset.imgs[0]
            if first_image.ndim == 3:  # Handle (H, W, C) or (H, W, 1)
                self.h, self.w = first_image.shape[0], first_image.shape[1]
            else:  # (H, W)
                self.h, self.w = first_image.shape[0], first_image.shape[1]
        else:
            self.h, self.w = 0, 0  # Default for empty dataset

        self.n_classes = len(self.class_names)
        self.results = {}

    def _get_images_as_float32(self, indices=None):
        """Lazily load and convert images to float32, or a subset if indices are provided."""
        if indices is not None:
            return self.dataset.imgs[indices].astype(np.float32) / 255.0
        return self.dataset.imgs.astype(np.float32) / 255.0

    def audit_information_theory(self):
        # Temporarily load all images to perform calculations
        images_float32 = self._get_images_as_float32()

        img_means = np.mean(images_float32, axis=(1, 2))
        img_stds = np.std(images_float32, axis=(1, 2))

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
        })
        del images_float32  # Free memory
        gc.collect()

    def audit_spatial_morphology(self):
        # Temporarily load all images for global stats
        images_float32 = self._get_images_as_float32()

        global_mean = np.mean(images_float32, axis=0)
        std_heatmap = np.std(images_float32, axis=0)

        diff_maps = {}
        for i, name in enumerate(self.class_names):
            mask = self.labels[:, i] == 1
            if mask.sum() >= 10:
                class_mean = np.mean(images_float32[mask], axis=0)
                diff_maps[name] = class_mean - global_mean

        rng = np.random.RandomState(GLOBAL_SEED)
        sample_idx = rng.choice(self.n_samples, min(1000, self.n_samples), replace=False)
        # Only load a subset for edge density calculation
        sampled_images = self._get_images_as_float32(sample_idx)
        edge_density = np.mean([filters.sobel(img) for img in sampled_images], axis=0)

        self.results.update({'global_mean': global_mean, 'std_heatmap': std_heatmap, 'diff_maps': diff_maps,
                             'edge_density': edge_density})
        del images_float32, sampled_images  # Free memory
        gc.collect()

    def audit_clinical_logic(self):
        n = self.n_classes
        phi_matrix = np.zeros((n, n))
        cond_prevalence = np.zeros((n, n))
        cramers_v_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                phi_matrix[i, j] = matthews_corrcoef(self.labels[:, i], self.labels[:, j])
                denom = np.sum(self.labels[:, j])
                if denom > 0:
                    cond_prevalence[i, j] = np.sum(self.labels[:, i] & self.labels[:, j]) / denom

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

    def audit_safety_integrity(self):
        # Temporarily load all images for entropy, SNR, and hashing
        images_float32 = self._get_images_as_float32()

        image_entropies = np.array(
            [entropy(np.histogram(images_float32[i], bins=30, density=True)[0] + 1e-10) for i in range(self.n_samples)])
        img_means = np.mean(images_float32, axis=(1, 2))
        img_stds = np.std(images_float32, axis=(1, 2))
        snrs = img_means / (img_stds + 1e-8)

        low_snr_mask = snrs < np.percentile(snrs, 1)
        low_snr_count = int(low_snr_mask.sum())

        hashes = set()
        dup_indices = []
        for i in range(self.n_samples):
            # Hash directly from original uint8 data to save memory/speed
            h = hashlib.blake2b(self.dataset.imgs[i].tobytes(), digest_size=16).hexdigest()
            if h in hashes:
                dup_indices.append(i)
            hashes.add(h)

        self.results.update({
            'img_entropy': image_entropies, 'snr_dist': snrs, 'low_snr_count': low_snr_count,
            'duplicates': len(dup_indices), 'dup_indices': dup_indices, 'n_unique_hash': len(hashes),
        })
        del images_float32  # Free memory
        gc.collect()

    @staticmethod
    def compute_image_hashes(dataset, digest_size: int = 16) -> Set[str]:
        # Use uint8 for hashing to avoid float32 conversion memory overhead
        imgs = dataset.imgs
        hashes = set()
        for i in range(len(imgs)):
            h = hashlib.blake2b(imgs[i].tobytes(), digest_size=digest_size).hexdigest()
            hashes.add(h)
        return hashes

    def audit_cross_split_leakage(self, other_datasets: Dict[str, 'any']) -> Dict[str, int]:
        print(f"\n[*] Cross-Split Image Leakage Audit ({self.split_name} vs others)...")
        self_hashes = self.compute_image_hashes(self.dataset)
        leakage_report = {}

        for name, ds in other_datasets.items():
            other_hashes = self.compute_image_hashes(ds)
            overlap = self_hashes.intersection(other_hashes)
            leakage_report[name] = len(overlap)
            if len(overlap) > 0:
                print(f"     [!] LEAKAGE: {len(overlap)} images shared with {name}!")
            else:
                print(f"     [✓] CLEAN: 0 images shared with {name}.")
            del other_hashes  # Free memory
            gc.collect()

        self.results['cross_split_leakage'] = leakage_report
        del self_hashes  # Free memory
        gc.collect()
        return leakage_report

    def audit_localized_structure(self, grid_size: int = 2):
        print(f"\n[*] Localized Spatial Audit ({grid_size}x{grid_size} grid)...")
        # Temporarily load all images
        images_float32 = self._get_images_as_float32()

        ph, pw = self.h // grid_size, self.w // grid_size
        quadrant_info = {}

        for i in range(grid_size):
            for j in range(grid_size):
                y0, y1 = i * ph, (i + 1) * ph
                x0, x1 = j * pw, (j + 1) * pw
                patches = images_float32[:, y0:y1, x0:x1]
                patch_means = np.mean(patches, axis=(1, 2))
                patch_std = np.std(patches, axis=(1, 2))
                patch_feat = np.round(patch_means * 50).astype(int) * 100 + np.round(patch_std * 50).astype(int)
                patch_mi = [mutual_info_score(self.labels[:, c], patch_feat) for c in range(self.n_classes)]
                quadrant_info[f"Q_{i}{j}"] = {
                    'mi': patch_mi, 'mean_intensity': float(np.mean(patch_means)),
                    'std_intensity': float(np.std(patch_means)),
                }

        self.results['spatial_quadrants'] = quadrant_info
        del images_float32  # Free memory
        gc.collect()
        return quadrant_info

    def audit_clinical_consistency(self):
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

    def audit_effective_samples(self, beta: float = 0.9999):
        ens_data = []
        for c in range(self.n_classes):
            n_c = int(self.labels[:, c].sum())
            ens = (1.0 - beta ** n_c) / (1.0 - beta) if beta < 1.0 else float(n_c)
            ens_data.append({'Class': self.class_names[c], 'n_pos': n_c, 'n_neg': self.n_samples - n_c,
                             'ENS': f"{ens:.1f}", 'Pos/Neg': f"1:{(self.n_samples - n_c) / max(n_c, 1):.1f}"})
        ens_df = pd.DataFrame(ens_data)
        self.results['ens_table'] = ens_df
        return ens_df

    def audit_entropy_gap(self):
        if 'img_entropy' not in self.results:
            self.audit_safety_integrity()  # Call this to ensure img_entropy is computed
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
        print(
            f"\n{'=' * 65}\n  FORENSIC DATASET AUDIT — {self.split_name} Split\n  n = {self.n_samples:,} | {self.n_classes} classes | {self.h}x{self.w} px\n{'=' * 65}")

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
        print(f"    Δ(H) = {eg['gap']:.4f}  |  MWU p = {eg['mwu_p']:.2e}  |  KS p = {eg['ks_p']:.2e}")

        cv, ip = self.results.get('clinical_violations', 0), self.results.get('implausible_cooccurrences', 0)
        print(f"\n  Clinical Logic: {cv} high-cardinality, {ip} implausible co-occurrence(s)")

        # Generate Mosaic Figure
        try:
            fig = plt.figure(figsize=(24, 30))
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

            sns.heatmap(self.results['cond_prev'], ax=ax['A'], annot=True, fmt='.2f', cmap='mako',
                        xticklabels=short_names, yticklabels=short_names, linewidths=0.5, linecolor='white')
            ax['A'].set_title("A. Conditional Prevalence P(Row | Col)", fontweight='bold')

            # Use a sample of images for pixel-wise variance if dataset is very large
            if self.n_samples > 5000:  # Heuristic to avoid OOM for very large datasets
                sample_indices = np.random.choice(self.n_samples, 5000, replace=False)
                sampled_images_float32 = self._get_images_as_float32(sample_indices)
                im_b = ax['B'].imshow(np.std(sampled_images_float32, axis=0), cmap='magma', aspect='equal')
                del sampled_images_float32
            else:
                images_float32 = self._get_images_as_float32()
                im_b = ax['B'].imshow(np.std(images_float32, axis=0), cmap='magma', aspect='equal')
                del images_float32
            ax['B'].set_title("B. Pixel-Wise Variance", fontweight='bold');
            ax['B'].axis('off')
            plt.colorbar(im_b, ax=ax['B'], fraction=0.046)

            sns.histplot(self.results['snr_dist'], ax=ax['C'], bins=50, kde=True, color='#0072B2', alpha=0.7)
            ax['C'].set_title("C. Signal-to-Noise Ratio", fontweight='bold')

            mask = np.triu(np.ones_like(self.results['phi'], dtype=bool))
            sns.heatmap(self.results['phi'], ax=ax['D'], annot=True, fmt='.2f', cmap='RdBu_r', center=0, mask=mask,
                        xticklabels=short_names, yticklabels=short_names, linewidths=0.3)
            ax['D'].set_title("D. Phi Coefficient", fontweight='bold')

            sorted_idx = np.argsort(self.results['mi_joint'])[::-1]
            ax['E'].barh([short_names[i] for i in sorted_idx], [self.results['mi_joint'][i] for i in sorted_idx],
                         color='#D55E00', alpha=0.85)
            ax['E'].set_title("E. Mutual Info", fontweight='bold');
            ax['E'].invert_yaxis()

            # Global mean image requires all images
            images_float32_for_global = self._get_images_as_float32()
            ax['F'].imshow(np.mean(images_float32_for_global, axis=0), cmap='bone', aspect='equal')
            ax['F'].set_title("F. Global Mean Image", fontweight='bold');
            ax['F'].axis('off')

            # Edge density is already sampled in audit_spatial_morphology
            ax['G'].imshow(self.results['edge_density'], cmap='viridis', aspect='equal')
            ax['G'].set_title("G. Edge Density", fontweight='bold');
            ax['G'].axis('off')

            # Class Intensity Profiles - sample if needed
            images_float32_for_h = self._get_images_as_float32()
            for i in range(min(5, self.n_classes)):
                mask_c = self.labels[:, i] == 1
                if mask_c.sum() > 0:
                    # Sample up to 10000 pixels if class is very large
                    flattened_pixels = images_float32_for_h[mask_c].flatten()
                    if len(flattened_pixels) > 10000:
                        flattened_pixels = np.random.choice(flattened_pixels, 10000, replace=False)
                    sns.kdeplot(flattened_pixels, ax=ax['H'], label=short_names[i], linewidth=0.8)
            ax['H'].set_title("H. Class Intensity Profiles", fontweight='bold');
            ax['H'].legend(fontsize=6, ncol=2)
            del images_float32_for_global, images_float32_for_h  # Free memory
            gc.collect()

            sns.histplot(self.results['cardinality'], ax=ax['I'], discrete=True, color='#009E73')
            ax['I'].set_title(f"I. Label Cardinality", fontweight='bold')

            mi_map = np.zeros((2, 2), dtype=np.float32)
            spatial_data = self.results.get('spatial_quadrants', {})
            if spatial_data:
                for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)]: mi_map[i, j] = np.mean(spatial_data[f'Q_{i}{j}']['mi'])
            sns.heatmap(mi_map, ax=ax['J'], annot=True, fmt='.4f', cmap='YlOrRd')
            ax['J'].set_title("J. Spatial MI Map", fontweight='bold')

            sns.heatmap(self.results['cramers_v'], ax=ax['K'], annot=True, fmt='.3f', cmap='rocket_r',
                        xticklabels=short_names, yticklabels=short_names, linewidths=0.3)
            ax['K'].set_title("K. Cramér's V", fontweight='bold')

            ax['L'].axis('off')
            summary_lines = (
                f"FORENSIC SUMMARY — n = {self.n_samples:,}\nDuplicates: {self.results['duplicates']} | Max MI: {max(self.results['mi_joint']):.4f}")
            ax['L'].text(0.5, 0.5, summary_lines, ha='center', va='center', fontsize=14, family='monospace',
                         transform=ax['L'].transAxes)

            plt.savefig(f'forensic_audit_{self.split_name.lower()}.png', dpi=300, bbox_inches='tight')
            print(f"\n[✓] Audit figure saved: forensic_audit_{self.split_name.lower()}.png")
            plt.close(fig)
        except Exception as e:
            print(f"\n[!] Could not generate plots: {e}")
        finally:
            # Ensure figures are closed and memory is freed even if plotting fails
            plt.close('all')
            gc.collect()


# ============================================================
# EXECUTION WRAPPER
# ============================================================
def execute_forensic_audit(train_dataset, val_dataset, test_dataset):
    print("\n" + "=" * 65)
    print("  EXECUTING FORENSIC DATASET AUDIT (Memory-Optimized)")
    print("=" * 65)

    # Perform leakage audit first, as it needs all datasets
    auditor_for_leakage = ForensicDatasetAuditor(train_dataset, split_name="Train")
    leakage = auditor_for_leakage.audit_cross_split_leakage({'Val': val_dataset, 'Test': test_dataset})

    # Audit leakage between Val and Test too
    auditor_val_leakage = ForensicDatasetAuditor(val_dataset, split_name="Val")
    leakage_vt = auditor_val_leakage.audit_cross_split_leakage({'Test': test_dataset})

    total_leakage = sum(leakage.values()) + sum(leakage_vt.values())
    print(f"\n  {'=' * 50}")
    print(f"  LEAKAGE VERDICT: {'[✓] CLEAN' if total_leakage == 0 else f'[!] CONTAMINATED ({total_leakage} overlap)'}")
    print(f"  {'=' * 50}")

    del auditor_for_leakage, auditor_val_leakage  # Free memory after leakage check
    gc.collect()

    # Now, run full audits sequentially, one dataset at a time
    print("\n[*] Generating full audit reports per split (sequentially to manage memory)...")
    for ds, name in zip([train_dataset, val_dataset, test_dataset], ["Train", "Val", "Test"]):
        print(f"\n--- Starting detailed audit for {name} split ---")
        auditor = ForensicDatasetAuditor(ds, split_name=name)
        auditor.generate_report()
        del auditor  # Delete the auditor object
        gc.collect()  # Force garbage collection
        print(f"--- Finished detailed audit for {name} split ---\n")
