import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import log_process, LOGGER
from utils import CHESTMNIST_CLASS_NAMES
import logging

# ============================================================
# GRAPH ARCHITECTURE COMPONENTS
# ============================================================

class GraphGPSLayer(nn.Module):
    """
    Parallel residual: Local GCN branch + Global Transformer branch.
    Ref: Rampasek et al., GraphGPS, NeurIPS 2022.
    """
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1_l = nn.LayerNorm(dim)
        self.local_p = nn.Linear(dim, dim, bias=False)
        self.norm1_g = nn.LayerNorm(dim)
        self.glob_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, h, adj):
        h_loc = self.local_p(torch.matmul(adj, self.norm1_l(h)))
        h_n = self.norm1_g(h)
        h_g, _ = self.glob_attn(h_n.unsqueeze(0), h_n.unsqueeze(0), h_n.unsqueeze(0))
        h = h + self.drop(h_loc) + self.drop(h_g.squeeze(0))
        return h + self.ffn(self.norm2(h))

class LabelGraphGPSClassifier(nn.Module):
    """RadLex-conditioned GraphGPS generating class weight vectors W_class [K, D]."""
    def __init__(self, num_classes=14, embed_dim=128, hidden_dim=128, output_dim=384, radlex_dim=768, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.label_embed = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.trunc_normal_(self.label_embed, std=0.02)
        self.input_proj = nn.Sequential(
            nn.Linear(radlex_dim + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
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
# CLUSTERING & REPRESENTATION LEARNING
# ============================================================

class StudentTClustering(nn.Module):
    """Soft cluster assignments via heavy-tailed Student-t kernel (Xie et al., ICML 2016)."""
    def __init__(self, num_clusters=20, feat_dim=384, v=4.0):
        super().__init__()
        self.v = v
        self.centers = nn.Parameter(torch.empty(num_clusters, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, x):
        dist = (x.pow(2).sum(1, keepdim=True) + self.centers.pow(2).sum(1) - 2 * torch.matmul(x, self.centers.t())).clamp(min=0)
        q = (1 + dist / self.v).pow(-(self.v + 1) / 2)
        return q / q.sum(1, keepdim=True)

def batch_hard_triplet_loss(emb, pseudo_lbl, margin=0.3):
    """Vectorised Batch-Hard Triplet Loss (Hermans et al., 2017)."""
    D = torch.cdist(emb, emb, p=2.0)
    pos = pseudo_lbl.unsqueeze(0) == pseudo_lbl.unsqueeze(1)
    neg = ~pos
    if neg.sum() == 0 or pos.sum() == len(pseudo_lbl):
        return torch.tensor(0.0, device=emb.device, requires_grad=True)
    h_pos = (D * pos.float()).max(1)[0]
    h_neg = (D + pos.float() * (D.max().item() + 1e-5)).min(1)[0]
    return F.relu(h_pos - h_neg + margin).mean()

def target_distribution(q):
    p = q**2 / q.sum(0, keepdim=True).clamp(1e-12)
    return p / p.sum(1, keepdim=True).clamp(1e-12)

# ============================================================
# POOLING & MAIN MODEL
# ============================================================

class GLoRI_Lite_Pooler(nn.Module):
    """
    f = gate ⊙ CLS_proxy + (1−gate) ⊙ GeM(patches)
    """
    def __init__(self, feat_dim=384, gem_p_init=3.0, dropout=0.1):
        super().__init__()
        self.gem_p = nn.Parameter(torch.tensor(gem_p_init))
        self.gate_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, cls_token, patches):
        # NATURE-GRADE FIX 2: 
        # p-ს შეზღუდვა [1, 10] დიაპაზონში და clamp(min=1e-6) abs()-ის ნაცვლად
        # ეს უზრუნველყოფს გლუვ გრადიენტებს და თავიდან აცილებს რიცხვით გადავსებას (overflow).
        p = self.gem_p.clamp(min=1.0, max=10.0)
        gem = patches.clamp(min=1e-6).pow(p).mean(1).pow(1.0 / p)
        g = self.gate_mlp(torch.cat([cls_token, gem], -1))
        return self.norm(g * cls_token + (1 - g) * gem)

class CXR_Synapse_Foundation(nn.Module):
    """
    CXR-Synapse with Google ELIXR-C v2 as frozen external backbone.
    """
    def __init__(self, num_classes=14, cxr_dim=1376, feat_dim=384, gat_embed=128, gat_hidden=128, gat_heads=4, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.dim_reduction = nn.Sequential(
            nn.Linear(cxr_dim, feat_dim * 2),
            nn.LayerNorm(feat_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )
        self.pooler = GLoRI_Lite_Pooler(feat_dim, 3.0, dropout)
        self.student_t_clustering = StudentTClustering(20, feat_dim, v=4.0)
        self.label_graph = LabelGraphGPSClassifier(num_classes, gat_embed, gat_hidden, feat_dim, 768, gat_heads, 2, dropout)
        self.feat_dropout = nn.Dropout(dropout)
        
        # Buffers for non-trainable graph parameters
        self.register_buffer("adj_mask", torch.eye(num_classes))
        self.register_buffer("radlex_emb", torch.zeros(num_classes, 768))

    def set_adjacency_mask(self, adj_norm_np: np.ndarray):
        self.adj_mask.copy_(torch.from_numpy(adj_norm_np).float())
        print(f"  ✓ Adjacency set. Edge density: {(self.adj_mask > 0).float().mean():.2%}")
        log_process("model", "adjacency_mask_updated", edge_density=f"{(self.adj_mask > 0).float().mean().item():.4f}", shape=tuple(self.adj_mask.shape))

    def set_radlex_embeddings(self, emb: torch.Tensor):
        self.radlex_emb.copy_(emb)
        print(f"  ✓ RadLex embeddings set: {tuple(emb.shape)}")
        log_process("model", "radlex_embeddings_set", shape=tuple(emb.shape))

    def forward(self, x: torch.Tensor):
        assert self.radlex_emb.abs().sum() > 0, "RadLex embeddings are zero — call set_radlex_embeddings() first."
        B, H, W, C = x.shape
        patches = x.view(B, H * W, C)            # [B, 64, 1376]
        proj = self.dim_reduction(patches)        # [B, 64, 384]
        cls = proj.mean(dim=1)                    # CLS proxy
        fused = self.pooler(cls, proj)            # [B, 384]
        q = self.student_t_clustering(fused)      # Clustering assignment
        W_class = self.label_graph(self.radlex_emb, self.adj_mask) # [14, 384]
        
        logits = torch.matmul(self.feat_dropout(fused), W_class.t()) # [B, 14]
        
        if self.training:
            return logits, fused, q
        return logits

# ============================================================
# LOSS FUNCTIONS & EVIDENTIAL UNCERTAINTY
# ============================================================

class LogitAdjustedAsymmetricEvidentialLoss(nn.Module):
    """
    Unified loss: Logit Adjustment + Asymmetric Loss + Evidential penalty.
    """
    def __init__(self, logit_adj, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, annealing_epochs=15):
        super().__init__()
        self.register_buffer("logit_adj", logit_adj)
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.annealing_epochs = max(int(annealing_epochs), 1)

    def forward(self, logits, targets, current_epoch):
        # 1. Logit Adjustment (Prior shift)
        adjusted_logits = logits + self.logit_adj
        
        # 2. Asymmetric Focal Loss
        prob = torch.sigmoid(adjusted_logits)
        prob_pos = prob
        prob_neg = 1.0 - prob
        if self.clip > 0:
            prob_neg = (prob_neg + self.clip).clamp(max=1.0)
            
        loss_pos = targets * (-torch.log(prob_pos.clamp(min=1e-8))) * torch.pow(1.0 - prob_pos, self.gamma_pos)
        loss_neg = (1.0 - targets) * (-torch.log(prob_neg.clamp(min=1e-8))) * torch.pow(1.0 - prob_neg, self.gamma_neg)
        clf_loss = (loss_pos + loss_neg).mean()
        
        # 3. Evidential Regularization
        evidence_pos = F.softplus(adjusted_logits)
        evidence_neg = F.softplus(-adjusted_logits)
        penalty = targets * evidence_neg + (1.0 - targets) * evidence_pos
        anneal_coef = min(1.0, float(current_epoch) / float(self.annealing_epochs))
        edl_loss = torch.mean(penalty) * anneal_coef
        
        return clf_loss + 0.1 * edl_loss

def get_evidential_metrics(logits):
    """Decomposition of uncertainty into Epistemic and Aleatoric components."""
    ep = F.softplus(logits)
    en = F.softplus(-logits)
    alpha = ep + 1.0
    beta = en + 1.0
    S = alpha + beta
    prob = alpha / S
    epistemic = 2.0 / S
    aleatoric = prob * (1.0 - prob)
    return prob, epistemic, aleatoric