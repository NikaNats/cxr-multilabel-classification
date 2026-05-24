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
# § 2  CALIBRATED MULTI-LABEL PROBABILITIES & SHANNON ENTROPY (REPLACED EDL)
# ==============================================================================

def get_evidential_metrics(
    z_v: torch.Tensor, priors: torch.Tensor | np.ndarray | None = None, W: float = 2.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replaces Binomial Subjective Logic with standard sigmoid-based probabilities
    and normalized predictive Shannon entropy. Preserves the function signature 
    to prevent breaking existing validation downstream dependencies.
    
    Returns:
        prob: Sigmoid probabilities.
        epistemic: Shannon entropy per class scaled to [0, 1] as a proxy for uncertainty.
        aleatoric: Statistical binomial variance p * (1 - p).
    """
    prob = torch.sigmoid(z_v)
    eps = 1e-8
    
    # Aleatoric Uncertainty: Binomial variance of predictions
    aleatoric = prob * (1.0 - prob)
    
    # Epistemic Uncertainty: Shannon entropy per class rescaled to [0, 1] range using base-2 logarithm
    entropy = - (prob * torch.log2(prob + eps) + (1.0 - prob) * torch.log2(1.0 - prob + eps))
    epistemic = entropy.clamp(min=0.0, max=1.0)
    
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
# § 5  ASYMMETRIC LOSS (ASL) FOR RIGOROUS IMBALANCE CONTROL
# ==============================================================================

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for Multi-Label Classification.
    Decouples positive and negative focusing parameter dynamics to preserve
    stable clinical gradient flow on highly skewed medical distributions.
    """
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        """
        Calculates loss over target-decoupled sigmoid outputs.
        Accepts the 'epoch' argument optionally to preserve API compatibility.
        """
        # 1. Compute baseline sigmoid probabilities
        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos

        # 2. Asymmetric shifting/clipping margin for easy negative classes
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # 3. Standard Bilateral Cross Entropy components
        loss_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        # 4. Target-decoupled asymmetric focal focusing scale
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            if self.gamma_pos > 0:
                factors_pos = (1.0 - xs_pos) ** self.gamma_pos
                loss_pos *= factors_pos
            if self.gamma_neg > 0:
                factors_neg = (1.0 - xs_neg) ** self.gamma_neg
                loss_neg *= factors_neg
            loss = loss_pos + loss_neg

        return -loss.mean()