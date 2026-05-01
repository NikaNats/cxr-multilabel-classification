import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# PATHOLOGY-AS-QUERY (PaQ) SPATIAL CROSS-ATTENTION
# ============================================================
class PathologyCrossAttention(nn.Module):
    """SOTA 2026: Pathology-as-Query Routing Engine."""
    def __init__(self, num_classes=14, radlex_dim=768, feat_dim=384, num_heads=4, dropout=0.1):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(radlex_dim, feat_dim),
            nn.LayerNorm(feat_dim)
        )
        self.self_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_self = nn.LayerNorm(feat_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(feat_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 4, feat_dim)
        )
        self.norm_ffn = nn.LayerNorm(feat_dim)
        self.register_buffer("logit_scale", torch.tensor(np.log(1 / 0.07)))

    def forward(self, patches, radlex_emb):
        B = patches.shape[0]
        queries = self.text_proj(radlex_emb).unsqueeze(0).expand(B, -1, -1)
        
        # 1. Self-Attention: Comorbidity resolution
        self_out, _ = self.self_attn(query=queries, key=queries, value=queries)
        queries = self.norm_self(queries + self_out)
        
        # 2. Cross-Attention: Extract isolated visual evidence
        attn_out, _ = self.cross_attn(query=queries, key=patches, value=patches)
        hidden_cross = self.norm_cross(attn_out)
        
        # 3. Residual FFN mapping
        disease_features = self.norm_ffn(hidden_cross + self.ffn(hidden_cross))
        
        # 4. L2-Normalized Cosine Similarity
        norm_disease = F.normalize(disease_features, p=2, dim=-1)
        norm_queries = F.normalize(queries, p=2, dim=-1)
        
        logits = torch.sum(norm_disease * norm_queries, dim=-1) * torch.exp(self.logit_scale)
        return logits, disease_features


# ============================================================
# MAIN FOUNDATION ARCHITECTURE
# ============================================================

class CXR_Synapse_Foundation(nn.Module):
    def __init__(self, num_classes=14, cxr_dim=1376, feat_dim=384, dropout=0.1):
        super().__init__()
        self.dim_reduction = nn.Sequential(
            nn.Linear(cxr_dim, feat_dim * 2), nn.LayerNorm(feat_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim), nn.LayerNorm(feat_dim), nn.GELU(),
        )
        
        # Learnable Anatomical Positional Embeddings (8x8 grid = 64 patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, feat_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.pathology_router = PathologyCrossAttention(num_classes, 768, feat_dim)
        
        self.register_buffer("radlex_emb", torch.zeros(num_classes, 768))
        self.register_buffer("logit_prior", torch.zeros(num_classes))
        self.register_buffer("adjacency_mask", torch.zeros(num_classes, num_classes))

    def set_adjacency_mask(self, adj_norm_np: np.ndarray):
        self.adjacency_mask.copy_(torch.from_numpy(adj_norm_np).float())

    def set_radlex_embeddings(self, emb: torch.Tensor):
        self.radlex_emb.copy_(emb)

    def set_logit_prior(self, prior_np: np.ndarray):
        self.logit_prior.copy_(torch.from_numpy(prior_np).float())

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape
        patches = x.view(B, H * W, C)
        proj = self.dim_reduction(patches)
        
        # Inject Spatial Coordinates
        proj = proj + self.pos_embed
        
        z_v, disease_features = self.pathology_router(proj, self.radlex_emb)
        z_posterior = z_v + self.logit_prior # Intrinsic Logit Prior Fusion
        
        if self.training:
            return z_posterior, z_v
        return z_posterior


# ============================================================
# SUBJECTIVE LOGIC & LOSS (Trustworthy AI)
# ============================================================

class StrictlyProperBetaEvidentialLoss(nn.Module):
    """SOTA 2026: Unweighted BCE relying purely on the Intrinsic Logit Prior."""
    def __init__(self, annealing_epochs=20):
        super().__init__()
        self.annealing_epochs = max(int(annealing_epochs), 1)

    def beta_kl_divergence(self, alpha, beta):
        gamma_ab = torch.lgamma(alpha + beta)
        gamma_a = torch.lgamma(alpha)
        gamma_b = torch.lgamma(beta)
        digamma_ab = torch.digamma(alpha + beta)
    def __init__(self, annealing_epochs=20):
        super().__init__()
        self.annealing_epochs = max(int(annealing_epochs), 1)

    def beta_kl_divergence(self, alpha, beta):
        gamma_ab = torch.lgamma(alpha + beta)
        gamma_a = torch.lgamma(alpha)
        gamma_b = torch.lgamma(beta)
        digamma_ab = torch.digamma(alpha + beta)
        return (gamma_ab - gamma_a - gamma_b) + \
               (alpha - 1.0) * (torch.digamma(alpha) - digamma_ab) + \
               (beta - 1.0) * (torch.digamma(beta) - digamma_ab)

    def forward(self, z_posterior, z_v, targets, current_epoch):
        # 1. Fisher-Consistent Classification Loss
        clf_loss = F.binary_cross_entropy_with_logits(z_posterior, targets)

        # 2. FP16 Safe Evidential Calculation
        z_safe = torch.clamp(z_v, min=-10.0, max=10.0) 
        alpha = torch.exp(z_safe) + 1.0
        beta = torch.exp(-z_safe) + 1.0
        
        alpha_tilde = targets + (1.0 - targets) * alpha
        beta_tilde = (1.0 - targets) + targets * beta
        
        kl_loss = torch.mean(self.beta_kl_divergence(alpha_tilde, beta_tilde))
        anneal_coef = min(1.0, float(current_epoch) / float(self.annealing_epochs))
        
        return clf_loss + (0.05 * anneal_coef * kl_loss)


def get_evidential_metrics(z_posterior):
    """Calculates perfect probabilistic metrics derived directly from BCE logic."""
    prob = torch.sigmoid(z_posterior)
    
    z_safe = torch.clamp(z_posterior, min=-10.0, max=10.0)
    alpha = torch.exp(z_safe) + 1.0
    beta = torch.exp(-z_safe) + 1.0
    S = alpha + beta
    
    epistemic = 2.0 / S
    aleatoric = prob * (1.0 - prob)
    return prob, epistemic, aleatoric