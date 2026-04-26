import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import log_process, LOGGER
from utils import CHESTMNIST_CLASS_NAMES
import logging

# ============================================================
# CLUSTERING & LATENT REGULARIZATION (Nature-Grade Utilities)
# ============================================================

def target_distribution(q):
    """
    Computes the sharpened target distribution P for the KL divergence loss.
    Ref: Xie et al., DEC, ICML 2016.
    """
    p = q**2 / q.sum(0, keepdim=True).clamp(min=1e-12)
    return p / p.sum(1, keepdim=True).clamp(min=1e-12)

def batch_hard_triplet_loss(emb, pseudo_lbl, margin=0.3):
    """
    Vectorized Batch-Hard Triplet Loss. 
    Aids in latent space separation between pathologies.
    """
    D = torch.cdist(emb, emb, p=2.0)
    pos = pseudo_lbl.unsqueeze(0) == pseudo_lbl.unsqueeze(1)
    neg = ~pos
    
    if neg.sum() == 0 or pos.sum() == len(pseudo_lbl):
        return torch.tensor(0.0, device=emb.device, requires_grad=True)
    
    h_pos = (D * pos.float()).max(1)[0]
    h_neg = (D + pos.float() * (D.max().item() + 1e-5)).min(1)[0]
    return F.relu(h_pos - h_neg + margin).mean()

class StudentTClustering(nn.Module):
    """Soft cluster assignments via heavy-tailed Student-t kernel (Xie et al., ICML 2016)."""
    def __init__(self, num_clusters=20, feat_dim=384, v=1.0):
        super().__init__()
        self.v = v # Degrees of freedom
        self.centers = nn.Parameter(torch.empty(num_clusters, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, x):
        dist = (x.pow(2).sum(1, keepdim=True) + self.centers.pow(2).sum(1) - 2 * torch.matmul(x, self.centers.t())).clamp(min=0)
        q = (1 + dist / self.v).pow(-(self.v + 1) / 2)
        return q / q.sum(1, keepdim=True)

# ============================================================
# GRAPH ARCHITECTURE COMPONENTS
# ============================================================

class GraphGPSLayer(nn.Module):
    """Parallel residual: Local GCN branch + Global Transformer branch."""
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1_l = nn.LayerNorm(dim)
        self.local_p = nn.Linear(dim, dim, bias=False)
        self.norm1_g = nn.LayerNorm(dim)
        self.glob_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, h, adj):
        h_loc = self.local_p(torch.matmul(adj, self.norm1_l(h)))
        h_n = self.norm1_g(h)
        h_g, _ = self.glob_attn(h_n.unsqueeze(0), h_n.unsqueeze(0), h_n.unsqueeze(0))
        h = h + self.drop(h_loc) + self.drop(h_g.squeeze(0))
        return h + self.ffn(self.norm2(h))

class LabelGraphGPSClassifier(nn.Module):
    """RadLex-conditioned GraphGPS generating class weight vectors."""
    def __init__(self, num_classes=14, embed_dim=128, hidden_dim=128, output_dim=384, radlex_dim=768, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.label_embed = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.trunc_normal_(self.label_embed, std=0.02)
        self.input_proj = nn.Sequential(
            nn.Linear(radlex_dim + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.gps_layers = nn.ModuleList([GraphGPSLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, radlex_emb, adj):
        h = self.input_proj(torch.cat([radlex_emb, self.label_embed], -1))
        for layer in self.gps_layers:
            h = layer(h, adj)
        return self.out_proj(self.norm_out(h))

# ============================================================
# POOLING & MAIN MODEL
# ============================================================

class GLoRI_Lite_Pooler(nn.Module):
    def __init__(self, feat_dim=384, gem_p_init=3.0, dropout=0.1):
        super().__init__()
        self.gem_p = nn.Parameter(torch.tensor(gem_p_init))
        self.gate_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim), nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, cls_token, patches):
        p = self.gem_p.clamp(min=1.0, max=10.0) 
        gem = patches.clamp(min=1e-6).pow(p).mean(1).pow(1.0 / p)
        g = self.gate_mlp(torch.cat([cls_token, gem], -1))
        fused = g * cls_token + (1 - g) * gem
        return self.norm(fused)

class CXR_Synapse_Foundation(nn.Module):
    def __init__(self, num_classes=14, cxr_dim=1376, feat_dim=384, gat_embed=128, gat_hidden=128, gat_heads=4, dropout=0.1):
        super().__init__()
        self.dim_reduction = nn.Sequential(
            nn.Linear(cxr_dim, feat_dim * 2), nn.LayerNorm(feat_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim), nn.LayerNorm(feat_dim), nn.GELU(),
        )
        self.pooler = GLoRI_Lite_Pooler(feat_dim, 3.0, dropout)
        self.student_t_clustering = StudentTClustering(20, feat_dim, v=1.0)
        self.label_graph = LabelGraphGPSClassifier(num_classes, gat_embed, gat_hidden, feat_dim, 768, gat_heads, 2, dropout)
        self.feat_dropout = nn.Dropout(dropout)
        
        self.register_buffer("adj_mask", torch.eye(num_classes))
        self.register_buffer("radlex_emb", torch.zeros(num_classes, 768))

    def set_adjacency_mask(self, adj_norm_np: np.ndarray):
        self.adj_mask.copy_(torch.from_numpy(adj_norm_np).float())

    def set_radlex_embeddings(self, emb: torch.Tensor):
        self.radlex_emb.copy_(emb)

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape
        patches = x.view(B, H * W, C)
        proj = self.dim_reduction(patches)
        cls = proj.mean(dim=1)
        fused = self.pooler(cls, proj)
        q = self.student_t_clustering(fused)
        W_class = self.label_graph(self.radlex_emb, self.adj_mask) 
        logits = torch.matmul(self.feat_dropout(fused), W_class.t())
        
        if self.training:
            return logits, fused, q
        return logits

# ============================================================
# LOSS FUNCTIONS
# ============================================================

class LogitAdjustedAsymmetricEvidentialLoss(nn.Module):
    def __init__(self, logit_adj, gamma_pos=0.0, gamma_neg=2.0, clip=0.05, annealing_epochs=20):
        super().__init__()
        self.register_buffer("logit_adj", logit_adj)
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.annealing_epochs = max(int(annealing_epochs), 1)
        self.pos_weight = 7.0 # Balances rare pathology recall (F1 Fix)

    def kl_divergence(self, alpha, num_classes):
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        first_term = (torch.lgamma(sum_alpha) - torch.lgamma(alpha.new_tensor(float(num_classes))) - 
                      torch.sum(torch.lgamma(alpha), dim=1, keepdim=True))
        second_term = torch.sum((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(sum_alpha)), dim=1, keepdim=True)
        return first_term + second_term

    def forward(self, logits, targets, current_epoch):
        adjusted_logits = logits + self.logit_adj
        prob = torch.sigmoid(adjusted_logits)
        
        # Classification Component
        loss_pos = targets * (-torch.log(prob.clamp(min=1e-8))) * torch.pow(1.0 - prob, self.gamma_pos) * self.pos_weight
        prob_neg = (1.0 - prob + self.clip).clamp(max=1.0)
        loss_neg = (1.0 - targets) * (-torch.log(prob_neg.clamp(min=1e-8))) * torch.pow(1.0 - prob_neg, self.gamma_neg)
        clf_loss = (loss_pos + loss_neg).mean()

        # Evidential Sparsity Component (Coverage Fix)
        evidence = F.softplus(adjusted_logits)
        alpha = evidence + 1.0
        rem_alpha = targets + (1.0 - targets) * alpha
        anneal_coef = min(1.0, float(current_epoch) / float(self.annealing_epochs))
        kl_loss = torch.mean(self.kl_divergence(rem_alpha, alpha.shape[1]))

        return clf_loss + (0.1 * anneal_coef * kl_loss)

def get_evidential_metrics(logits):
    ep = F.softplus(logits); en = F.softplus(-logits)
    alpha = ep + 1.0; beta = en + 1.0; S = alpha + beta
    prob = alpha / S; epistemic = 2.0 / S; aleatoric = prob * (1.0 - prob)
    return prob, epistemic, aleatoric