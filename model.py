from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# § 1  2D ROTARY POSITION EMBEDDING (2D RoPE) — SOTA 2025/2026
# ==============================================================================

def apply_2d_rope(x: torch.Tensor, grid_size: int = 8) -> torch.Tensor:
    """
    Applies 2D Rotary Position Embeddings (RoPE) to a spatial feature tensor.
    Preserves semantic energy of frozen backbone representations.
    
    Args:
        x: Tensor of shape (batch_size, grid_size * grid_size, feat_dim)
        grid_size: Spatial resolution of the patch grid (default: 8x8)
    Returns:
        Rotated tensor of the same shape.
    """
    batch_size, num_patches, feat_dim = x.shape
    device = x.device
    
    # კუთხური სიხშირეების გენერირება
    dim_half = feat_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim_half, 2, dtype=torch.float32, device=device) / dim_half))
    
    # 2D ბადის კოორდინატების აგება
    t = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(t, t, indexing="ij")
    grid_y, grid_x = grid_y.reshape(-1), grid_x.reshape(-1) # Shape: (64,)
    
    # ფაზების გამოთვლა x და y კოორდინატებისთვის
    freqs_x = torch.outer(grid_x, inv_freq) # (64, dim_half/2)
    freqs_y = torch.outer(grid_y, inv_freq) # (64, dim_half/2)
    
    # სინუსურ-კოსინუსური მატრიცების გაერთიანება
    freqs = torch.cat([freqs_x, freqs_y], dim=-1) # (64, dim_half)
    emb = torch.cat([freqs, freqs], dim=-1) # (64, feat_dim)
    
    cos = emb.cos().unsqueeze(0) # (1, 64, feat_dim)
    sin = emb.sin().unsqueeze(0) # (1, 64, feat_dim)
    
    # როტაციული მატრიცის ტრიუკი (Rotary embedding transformation)
    def rotate_half(tensor):
        t1 = tensor[..., :tensor.shape[-1] // 2]
        t2 = tensor[..., tensor.shape[-1] // 2:]
        return torch.cat((-t2, t1), dim=-1)

    # როტაციული ბრუნვის გამოყენება
    x_rotated = (x * cos) + (rotate_half(x) * sin)
    return x_rotated


# ==============================================================================
# § 2  CALIBRATED MULTI-LABEL PROBABILITIES & SHANNON ENTROPY
# ==============================================================================

def get_evidential_metrics(
    z_v: torch.Tensor, priors: torch.Tensor | np.ndarray | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates multi-label evidential metrics.
    
    Returns:
        prob: Sigmoid probabilities.
        epistemic: Shannon entropy per class rescaled to [0, 1].
        aleatoric: Statistical binomial variance p * (1 - p).
    """
    prob = torch.sigmoid(z_v)
    eps = 1e-8
    
    aleatoric = prob * (1.0 - prob)
    
    entropy = - (prob * torch.log2(prob + eps) + (1.0 - prob) * torch.log2(1.0 - prob + eps))
    epistemic = entropy.clamp(min=0.0, max=1.0)
    
    return prob, epistemic, aleatoric


# ==============================================================================
# § 3  PATHOLOGY-AS-QUERY (PaQ) SPATIAL CROSS-ATTENTION WITH 2D RoPE
# ==============================================================================

class PathologyCrossAttention(nn.Module):
    """
    Routes localized chest embeddings using pathology-specific RadLex queries 
    and structured clinical graph adjacency maps with 2D RoPE.
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
        
        # 1. 2D RoPE-ის გამოყენება სივრცით პატჩებზე (კროს-ყურადღებამდე)
        # ეს მათემატიკურად სუფთად ახდენს სივრცითი კოორდინატების პროექციას
        rotated_patches = apply_2d_rope(patches, grid_size=8)
        
        attn_out, _ = self.cross_attn(query=queries, key=rotated_patches, value=rotated_patches)
        
        # რეზიდუალური კავშირი კროს-ყურადღების შემდგომ
        hidden_cross = self.norm_cross(queries + attn_out)
        
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
    """Main chest X-ray foundation architecture with Graph-Guided cross-attention and 2D RoPE."""
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
        
        # ძველი ადიტიური pos_embed მთლიანად ამოღებულია არქიტექტურიდან
        self.pathology_router = PathologyCrossAttention(num_classes, 768, feat_dim)
        
        self.register_buffer("radlex_emb", torch.zeros(num_classes, 768))
        self.register_buffer("priors", torch.zeros(num_classes) + 0.05)
        self.register_buffer("adjacency_mask", torch.zeros(num_classes, num_classes))

    def set_adjacency_mask(self, adj_norm_np: np.ndarray) -> None:
        self.adjacency_mask.copy_(torch.from_numpy(adj_norm_np).float())

    def set_radlex_embeddings(self, mt_emb: torch.Tensor) -> None:
        self.radlex_emb.copy_(mt_emb)

    def set_priors(self, priors_np: np.ndarray) -> None:
        self.priors.copy_(torch.from_numpy(priors_np).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, h_spatial, w_spatial, channels = x.shape
        
        patches = x.reshape(batch_size, h_spatial * w_spatial, channels)
        proj = self.dim_reduction(patches)
        
        # 2D RoPE ავტომატურად შესრულდება PathologyCrossAttention მოდულის შიგნით
        z_v, _ = self.pathology_router(proj, self.radlex_emb, self.adjacency_mask)
        return z_v


# ==============================================================================
# § 5  ASYMMETRIC LOSS (ASL) WITH NUMERICAL STABILITY
# ==============================================================================

class AsymmetricLoss(nn.Module):
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
        log_xs_pos = F.logsigmoid(x)
        log_xs_neg = F.logsigmoid(-x)

        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg_clipped = (xs_neg + self.clip).clamp(max=1.0)
            log_xs_neg = torch.log(xs_neg_clipped)
        else:
            xs_neg_clipped = xs_neg

        loss_pos = y * log_xs_pos
        loss_neg = (1.0 - y) * log_xs_neg

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            if self.gamma_pos > 0:
                factors_pos = (1.0 - xs_pos) ** self.gamma_pos
                loss_pos *= factors_pos
            if self.gamma_neg > 0:
                factors_neg = (1.0 - xs_neg_clipped) ** self.gamma_neg
                loss_neg *= factors_neg

        loss = loss_pos + loss_neg
        return -loss.mean()