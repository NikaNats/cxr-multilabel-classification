from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# § 1  ANATOMICAL 2D POSITIONAL ENCODING
# ==============================================================================

def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, temperature: float = 10000.0
) -> torch.Tensor:
    """Generates a 2D sine-cosine spatial positional embedding grid."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    
    pos_embed = np.zeros((grid_size * grid_size, embed_dim), dtype=np.float32)
    dim_t = np.arange(embed_dim // 2, dtype=np.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / (embed_dim // 2))
    
    pos_h = grid[1].reshape(-1, 1) / dim_t
    pos_w = grid[0].reshape(-1, 1) / dim_t
    
    pos_embed[:, 0 : embed_dim // 2 : 2] = np.sin(pos_h[:, 0::2])
    pos_embed[:, 1 : embed_dim // 2 : 2] = np.cos(pos_h[:, 1::2])
    pos_embed[:, embed_dim // 2 : : 2] = np.sin(pos_w[:, 0::2])
    pos_embed[:, embed_dim // 2 + 1 : : 2] = np.cos(pos_w[:, 1::2])
    
    return torch.from_numpy(pos_embed).unsqueeze(0)


# ==============================================================================
# § 2  PRIOR-AWARE EVIDENTIAL METRICS
# ==============================================================================

def get_evidential_metrics(
    z_v: torch.Tensor, priors: torch.Tensor | np.ndarray | None = None, W: float = 2.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes prior-aware probabilities, epistemic uncertainty, and aleatoric 
    uncertainty from unshifted logits using Binomial Subjective Logic.
    """
    z_safe = torch.clamp(z_v, min=-10.0, max=10.0)
    alpha = torch.exp(z_safe) + 1.0
    beta = torch.exp(-z_safe) + 1.0
    s_param = alpha + beta
    
    r = alpha - 1.0
    
    if priors is not None:
        if isinstance(priors, torch.Tensor):
            priors_t = priors.to(device=z_v.device, dtype=z_v.dtype)
        else:
            priors_t = torch.as_tensor(priors, device=z_v.device, dtype=z_v.dtype)
        prob = (r + W * priors_t) / s_param
    else:
        prob = torch.sigmoid(z_v)
        
    epistemic = W / s_param
    aleatoric = prob * (1.0 - prob)
    return prob, epistemic, aleatoric


# ==============================================================================
# § 3  PATHOLOGY-AS-QUERY (PaQ) SPATIAL CROSS-ATTENTION
# ==============================================================================

class PathologyCrossAttention(nn.Module):
    """
    Routes localized chest embeddings using pathology-specific RadLex queries 
    and structured clinical graph adjacency maps.
    """
    def __init__(
        self,
        num_classes: int = 14,
        radlex_dim: int = 768,
        feat_dim: int = 384,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(radlex_dim, feat_dim),
            nn.LayerNorm(feat_dim)
        )
        self.graph_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_self = nn.LayerNorm(feat_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(feat_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 4, feat_dim)
        )
        self.norm_ffn = nn.LayerNorm(feat_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / 0.07))

    def forward(
        self, patches: torch.Tensor, radlex_emb: torch.Tensor, adjacency_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = patches.shape[0]
        base_queries = self.text_proj(radlex_emb).unsqueeze(0).expand(batch_size, -1, -1)
        
        adj_batch = adjacency_mask.unsqueeze(0).expand(batch_size, -1, -1)
        graph_queries = self.graph_proj(torch.bmm(adj_batch, base_queries))
        
        fused_queries = base_queries + graph_queries
        self_out, _ = self.self_attn(
            query=fused_queries, key=fused_queries, value=fused_queries
        )
        queries = self.norm_self(fused_queries + self_out)
        
        attn_out, _ = self.cross_attn(query=queries, key=patches, value=patches)
        hidden_cross = self.norm_cross(attn_out)
        
        disease_features = self.norm_ffn(hidden_cross + self.ffn(hidden_cross))
        
        norm_disease = F.normalize(disease_features, p=2, dim=-1)
        norm_queries = F.normalize(queries, p=2, dim=-1)
        
        scale_safe = self.logit_scale.clamp(max=math.log(100.0))
        logits = torch.sum(norm_disease * norm_queries, dim=-1) * torch.exp(scale_safe)
        
        return logits, disease_features


# ==============================================================================
# § 4  MAIN FOUNDATION ARCHITECTURE
# ==============================================================================

class CXR_Synapse_Foundation(nn.Module):
    """Main chest X-ray foundation architecture with Graph-Guided cross-attention."""
    def __init__(
        self,
        num_classes: int = 14,
        cxr_dim: int = 1376,
        feat_dim: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim_reduction = nn.Sequential(
            nn.Linear(cxr_dim, feat_dim * 2),
            nn.LayerNorm(feat_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )
        
        pos_emb = get_2d_sincos_pos_embed(feat_dim, grid_size=8)
        self.register_buffer("pos_embed", pos_emb)
        self.pathology_router = PathologyCrossAttention(num_classes, 768, feat_dim)
        
        self.register_buffer("radlex_emb", torch.zeros(num_classes, 768))
        self.register_buffer("priors", torch.zeros(num_classes) + 0.05)
        self.register_buffer("adjacency_mask", torch.zeros(num_classes, num_classes))

    def set_adjacency_mask(self, adj_norm_np: np.ndarray) -> None:
        """Loads a normalized adjacency matrix into the local buffer."""
        self.adjacency_mask.copy_(torch.from_numpy(adj_norm_np).float())

    def set_radlex_embeddings(self, emb: torch.Tensor) -> None:
        """Loads precomputed clinical RadLex text embeddings into the local buffer."""
        self.radlex_emb.copy_(emb)

    def set_priors(self, priors_np: np.ndarray) -> None:
        """Loads class prevalence priors into the local buffer."""
        self.priors.copy_(torch.from_numpy(priors_np).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, h_spatial, w_spatial, channels = x.shape
        patches = x.view(batch_size, h_spatial * w_spatial, channels)
        proj = self.dim_reduction(patches)
        proj = proj + self.pos_embed
        
        z_v, _ = self.pathology_router(proj, self.radlex_emb, self.adjacency_mask)
        return z_v


# ==============================================================================
# § 5  PRIOR-AWARE ASYMMETRIC FOCAL EVIDENTIAL LOSS
# ==============================================================================

class PriorAwareAsymmetricFocalLoss(nn.Module):
    """
    Computes prior-aware evidential classification losses using Dirichlet 
    parameter regularization and asymmetric focal scaling.
    """
    def __init__(
        self,
        priors: torch.Tensor | np.ndarray,
        annealing_epochs: int = 20,
        gamma_neg: float = 2.0,
        gamma_pos: float = 1.0,
        W: float = 2.0,
    ):
        super().__init__()
        self.annealing_epochs = max(int(annealing_epochs), 1)
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.W = W
        self.register_buffer("priors", torch.as_tensor(priors, dtype=torch.float32))

    def beta_kl_divergence(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Calculates the symmetric Kullback-Leibler divergence between two Beta distributions."""
        gamma_ab = torch.lgamma(alpha + beta)
        gamma_a = torch.lgamma(alpha)
        gamma_b = torch.lgamma(beta)
        digamma_ab = torch.digamma(alpha + beta)
        return (
            (gamma_ab - gamma_a - gamma_b)
            + (alpha - 1.0) * (torch.digamma(alpha) - digamma_ab)
            + (beta - 1.0) * (torch.digamma(beta) - digamma_ab)
        )

    def forward(self, z_v: torch.Tensor, targets: torch.Tensor, current_epoch: int) -> torch.Tensor:
        # 1. Compute prior-aware calibrated probabilities
        z_safe = torch.clamp(z_v, min=-10.0, max=10.0)
        alpha = torch.exp(z_safe) + 1.0
        beta = torch.exp(-z_safe) + 1.0
        s_param = alpha + beta
        r = alpha - 1.0
        
        probs = (r + self.W * self.priors) / s_param
        probs_clamped = torch.clamp(probs, 1e-7, 1.0 - 1e-7)

        # 2. Asymmetric Focal Objective on prior-aware probabilities
        focal_weight = (
            targets * ((1.0 - probs_clamped) ** self.gamma_pos)
            + (1.0 - targets) * (probs_clamped ** self.gamma_neg)
        )

        bce_loss = -(targets * torch.log(probs_clamped) + (1.0 - targets) * torch.log(1.0 - probs_clamped))
        clf_loss = torch.mean(focal_weight * bce_loss)

        # 3. KL Evidential annealing on unshifted Dirichlet parameters
        # Numerical Safety Fix: Add a tiny epsilon (1e-8) to ensure smooth gradients under FP16
        alpha_tilde = targets + (1.0 - targets) * alpha + 1e-8
        beta_tilde = (1.0 - targets) + targets * beta + 1e-8
        kl_raw = self.beta_kl_divergence(alpha_tilde, beta_tilde)

        class_anneal_factor = torch.clamp(1.0 / (self.priors + 1e-5), min=1.0, max=10.0)
        anneal_coef = torch.clamp(
            float(current_epoch) / (float(self.annealing_epochs) * class_anneal_factor), 
            max=1.0
        )
        kl_loss = torch.mean(anneal_coef * kl_raw)

        return clf_loss + (0.05 * kl_loss)