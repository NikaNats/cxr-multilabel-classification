import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CLUSTERING & LATENT REGULARIZATION (Manifold Alignment)
# ============================================================

def target_distribution(q):
    """
    Sharpening function for Deep Embedded Clustering (DEC).
    Ref: Xie et al., ICML 2016.
    
    Transforms soft assignments (q) into a high-confidence target distribution (P).
    This forces the encoder to learn features that cluster tightly around pathology prototypes.
    """
    p = q ** 2 / q.sum(0, keepdim=True).clamp(min=1e-12)
    return p / p.sum(1, keepdim=True).clamp(min=1e-12)


def batch_hard_triplet_loss(emb, pseudo_lbl, margin=0.3):
    """
    Vectorized Batch-Hard Triplet Loss. 
    Ref: Hermans et al., 2017.
    
    Ensures that the latent space is 'separable'. It pulls embeddings of the same 
    pathology together (positives) while pushing the hardest 'confusing' samples apart.
    Essential for multi-label chest X-ray diagnosis where features overlap.
    """
    # Compute Euclidean pairwise distance matrix
    D = torch.cdist(emb, emb, p=2.0)

    # Binary mask for samples sharing the same dominant label
    pos = pseudo_lbl.unsqueeze(0) == pseudo_lbl.unsqueeze(1)
    neg = ~pos

    if neg.sum() == 0 or pos.sum() == len(pseudo_lbl):
        return torch.tensor(0.0, device=emb.device, requires_grad=True)

    # Select hardest positive (max distance) and hardest negative (min distance)
    h_pos = (D * pos.float()).max(1)[0]
    h_neg = (D + pos.float() * (D.max().item() + 1e-5)).min(1)[0]

    return F.relu(h_pos - h_neg + margin).mean()


class StudentTClustering(nn.Module):
    """
    Subjective Latent Clustering via heavy-tailed Student-t kernel.
    The heavy tail (v=1) prevents the 'crowding problem' in latent space visualization.
    """

    def __init__(self, num_clusters=20, feat_dim=384, v=1.0):
        super().__init__()
        self.v = v  # Degrees of freedom (v=1 equivalent to Cauchy distribution)
        self.centers = nn.Parameter(torch.empty(num_clusters, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, x):
        # Calculate soft-assignment q_ij (probability of sample i belonging to cluster j)
        dist = (x.pow(2).sum(1, keepdim=True) + self.centers.pow(2).sum(1) - 2 * torch.matmul(x,
                                                                                              self.centers.t())).clamp(
            min=0)
        q = (1 + dist / self.v).pow(-(self.v + 1) / 2)
        return q / q.sum(1, keepdim=True)


# ============================================================
# GRAPH ARCHITECTURE COMPONENTS (Clinical Knowledge GPS)
# ============================================================

class GraphGPSLayer(nn.Module):
    """
    Graph General Processing Space (GPS) Layer.
    Ref: Rampasek et al., NeurIPS 2022.
    
    Hybrid logic:
    - Message Passing (GCN): Captures local clinical correlations (e.g., Effusion -> Atelectasis).
    - Global Transformer: Captures long-range relationships between semantically distinct pathologies.
    """

    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1_l = nn.LayerNorm(dim)
        self.local_p = nn.Linear(dim, dim, bias=False)  # GCN branch
        self.norm1_g = nn.LayerNorm(dim)
        self.glob_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)  # Transformer branch
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, h, adj):
        # Local message passing path
        h_loc = self.local_p(torch.matmul(adj, self.norm1_l(h)))
        # Global attention path
        h_n = self.norm1_g(h)
        h_g, _ = self.glob_attn(h_n.unsqueeze(0), h_n.unsqueeze(0), h_n.unsqueeze(0))
        # Residual fusion
        h = h + self.drop(h_loc) + self.drop(h_g.squeeze(0))
        return h + self.ffn(self.norm2(h))


class LabelGraphGPSClassifier(nn.Module):
    """
    Generates dynamic class weights (W_class) by processing RadLex clinical embeddings.
    Allows the model to 'reason' about pathologies rather than treating them as static indices.
    """

    def __init__(self, num_classes=14, embed_dim=128, hidden_dim=128, output_dim=384, radlex_dim=768, num_heads=4,
                 num_layers=2, dropout=0.1):
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
        # Fuse textual RadLex embeddings with learnable label priors
        h = self.input_proj(torch.cat([radlex_emb, self.label_embed], -1))
        for layer in self.gps_layers:
            h = layer(h, adj)
        return self.out_proj(self.norm_out(h))


# ============================================================
# POOLING & FEATURE FUSION
# ============================================================

class GLoRI_Lite_Pooler(nn.Module):
    """
    Generalized Representative Integration (Lite).
    Combines Global Average Pooling with Focal GeM Pooling.
    Ref: Radenovic et al., 2018 (GeM).
    """

    def __init__(self, feat_dim=384, gem_p_init=3.0, dropout=0.1):
        super().__init__()
        self.gem_p = nn.Parameter(torch.tensor(gem_p_init))
        self.gate_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim), nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, cls_token, patches):
        # GeM: Power-mean pooling focuses on high-activation regions (diagnostic focal points)
        p = self.gem_p.clamp(min=1.0, max=10.0)
        gem = patches.clamp(min=1e-6).pow(p).mean(1).pow(1.0 / p)
        # Gating: Dynamically balance global context (cls) and local evidence (gem)
        g = self.gate_mlp(torch.cat([cls_token, gem], -1))
        fused = g * cls_token + (1 - g) * gem
        return self.norm(fused)


# ============================================================
# MAIN FOUNDATION ARCHITECTURE
# ============================================================

class CXR_Synapse_Foundation(nn.Module):
    """
    Unified Multi-Label Clinical Diagnostic System.
    Frozen: Google ELIXR-C (CXR-Foundation v2).
    Trainable: Pro-head with Evidential Logic & GraphGPS.
    """

    def __init__(self, num_classes=14, cxr_dim=1376, feat_dim=384, gat_embed=128, gat_hidden=128, gat_heads=4,
                 dropout=0.1):
        super().__init__()
        # Dimension reduction for frozen 1376-dim feature maps
        self.dim_reduction = nn.Sequential(
            nn.Linear(cxr_dim, feat_dim * 2), nn.LayerNorm(feat_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim), nn.LayerNorm(feat_dim), nn.GELU(),
        )
        self.pooler = GLoRI_Lite_Pooler(feat_dim, 3.0, dropout)
        self.student_t_clustering = StudentTClustering(20, feat_dim, v=1.0)
        self.label_graph = LabelGraphGPSClassifier(num_classes, gat_embed, gat_hidden, feat_dim, 768, gat_heads, 2,
                                                   dropout)
        self.feat_dropout = nn.Dropout(dropout)

        # Knowledge Graph buffers
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

        # Generate class-specific weights via RadLex Knowledge Graph
        W_class = self.label_graph(self.radlex_emb, self.adj_mask)

        # Multi-label dot-product attention
        logits = torch.matmul(self.feat_dropout(fused), W_class.t())

        if self.training:
            return logits, fused, q
        return logits


# ============================================================
# SUBJECTIVE LOGIC & LOSS (Trustworthy AI)
# ============================================================

class LogitAdjustedAsymmetricEvidentialLoss(nn.Module):
    """
    Unified Loss Engine for Safety-Critical Chest X-Ray AI.
    
    1. Logit Adjustment: Corrects the extreme long-tail disease prevalence.
    2. Asymmetric Component: Penalizes False Negatives (FN) more than False Positives (FP).
    3. Evidential Sparsity (Dirichlet KL): The CORE fix for Conformal Coverage. 
       Forces 'No Evidence' (Sparsity) on non-present pathologies.
    """

    def __init__(self, logit_adj, gamma_pos=0.0, gamma_neg=2.0, clip=0.05, annealing_epochs=20):
        super().__init__()
        self.register_buffer("logit_adj", logit_adj)
        self.gamma_pos, self.gamma_neg, self.clip = gamma_pos, gamma_neg, clip
        self.annealing_epochs = max(int(annealing_epochs), 1)
        self.pos_weight = 7.0  # Clinical recall prior

    def kl_divergence(self, alpha, num_classes):
        """Calculates Dirichlet KL penalty to suppress hallucinated diagnostic evidence."""
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        first_term = (torch.lgamma(sum_alpha) - torch.lgamma(alpha.new_tensor(float(num_classes))) -
                      torch.sum(torch.lgamma(alpha), dim=1, keepdim=True))
        second_term = torch.sum((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(sum_alpha)), dim=1, keepdim=True)
        return first_term + second_term

    def forward(self, logits, targets, current_epoch):
        # Apply Menon et al. adjustment to handle ChestMNIST imbalances
        adjusted_logits = logits + self.logit_adj
        prob = torch.sigmoid(adjusted_logits)

        # Balanced Asymmetric Learning
        loss_pos = targets * (-torch.log(prob.clamp(min=1e-8))) * torch.pow(1.0 - prob,
                                                                            self.gamma_pos) * self.pos_weight
        prob_neg = (1.0 - prob + self.clip).clamp(max=1.0)
        loss_neg = (1.0 - targets) * (-torch.log(prob_neg.clamp(min=1e-8))) * torch.pow(1.0 - prob_neg, self.gamma_neg)
        clf_loss = (loss_pos + loss_neg).mean()

        # Evidential Regularization: Decompose logits into Dirichlet alpha parameters
        evidence = F.softplus(adjusted_logits)
        alpha = evidence + 1.0

        # Target sparsity: We only penalize evidence that doesn't belong to the ground truth
        rem_alpha = targets + (1.0 - targets) * alpha
        anneal_coef = min(1.0, float(current_epoch) / float(self.annealing_epochs))
        kl_loss = torch.mean(self.kl_divergence(rem_alpha, alpha.shape[1]))

        return clf_loss + (0.1 * anneal_coef * kl_loss)


def get_evidential_metrics(logits):
    """
    Probabilistic Decomposition of Diagnostic Predictions.
    
    Returns:
    - Prob: Belief-based diagnostic probability.
    - Epistemic: Ignorance (Model uncertainty due to lack of training data).
    - Aleatoric: Stochastic noise (Uncertainty due to poor image quality).
    """
    ep = F.softplus(logits);
    en = F.softplus(-logits)
    alpha = ep + 1.0;
    beta = en + 1.0;
    S = alpha + beta
    prob = alpha / S;
    epistemic = 2.0 / S;
    aleatoric = prob * (1.0 - prob)
    return prob, epistemic, aleatoric
