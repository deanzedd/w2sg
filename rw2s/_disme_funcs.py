"""
CPP-STPR: Contrastive Prototype Projection with Semantic Topology Preserving Regularization
Helper classes and functions used by train_logreg_hard_disme and train_head_DG_hard_disme.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class CPPProjector(nn.Module):
    """Non-linear projector h_phi: z -> v. Linear -> BN -> ReLU -> Linear"""
    def __init__(self, input_dim, hidden_dim=512, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)


def supervised_contrastive_loss(features, labels, temperature=0.07):
    """SupCon loss (Khosla 2020). features: (N,D) L2-normed, labels: (N,)"""
    device = features.device
    N = features.shape[0]
    if N <= 1:
        return torch.tensor(0.0, device=device, requires_grad=True)
    sim = torch.matmul(features, features.T) / temperature
    labels = labels.view(-1)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    pos_mask.fill_diagonal_(0)
    logits_max, _ = sim.max(dim=1, keepdim=True)
    logits = sim - logits_max.detach()
    self_mask = torch.eye(N, device=device)
    exp_logits = torch.exp(logits) * (1.0 - self_mask)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    n_pos = pos_mask.sum(1)
    valid = n_pos > 0
    if not valid.any():
        return torch.tensor(0.0, device=device, requires_grad=True)
    mean_log = (pos_mask * log_prob).sum(1) / (n_pos + 1e-8)
    return -mean_log[valid].mean()


def topology_regularization_loss(z, v, labels, n_classes, tau=1.0):
    """L_STPR: KL div between prototype similarity distributions in z-space vs v-space."""
    device = v.device
    unique = torch.unique(labels)
    if len(unique) < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)
    pz, pv = [], []
    for c in unique:
        m = (labels == c)
        pz.append(z[m].mean(0))
        pv.append(v[m].mean(0))
    pz = F.normalize(torch.stack(pz), dim=-1)
    pv = F.normalize(torch.stack(pv), dim=-1)
    Sz = torch.softmax(pz @ pz.T / tau, dim=-1).detach()
    Sv = torch.softmax(pv @ pv.T / tau, dim=-1)
    return F.kl_div(torch.log(Sv + 1e-8), Sz, reduction='batchmean')


class NearestCentroidClassifier:
    """Prototype-based classifier using cosine similarity."""
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.prototypes = None
        self.active_classes = None

    def fit(self, features, labels):
        device = features.device
        labels_flat = labels.view(-1)
        pl, cl = [], []
        for c in torch.unique(labels_flat).cpu().numpy():
            m = (labels_flat == c)
            if m.sum() > 0:
                pl.append(features[m].mean(0))
                cl.append(int(c))
        self.prototypes = F.normalize(torch.stack(pl), dim=-1)
        self.active_classes = torch.tensor(cl, device=device)

    def predict(self, features):
        fn = F.normalize(features, dim=-1)
        return self.active_classes[torch.matmul(fn, self.prototypes.T).argmax(-1)]

    def predict_with_logits(self, features):
        fn = F.normalize(features, dim=-1)
        sim = torch.matmul(fn, self.prototypes.T)
        full = torch.full((features.shape[0], self.n_classes), -1e9, device=features.device)
        for i, c in enumerate(self.active_classes):
            full[:, c] = sim[:, i]
        return full
