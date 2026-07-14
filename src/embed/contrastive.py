"""
Contrastive Learning Embedder — SimCLR 风格:
  对每条光谱做两种随机增强 → 编码 → NT-Xent loss → 用编码器作为特征提取器
接口兼容 sklearn: fit(X) / transform(X) / fit_transform(X)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ContrastiveEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class ProjectionHead(nn.Module):
    def __init__(self, embed_dim=30, proj_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


def nt_xent_loss(z1, z2, temperature=0.5):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss."""
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.t()) / temperature

    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    pos_mask = torch.zeros_like(sim, dtype=torch.bool)
    for i in range(batch_size):
        pos_mask[i, i + batch_size] = True
        pos_mask[i + batch_size, i] = True

    neg_mask = ~mask
    exp_sim = torch.exp(sim)
    denominator = (exp_sim * neg_mask.float()).sum(dim=1) + (exp_sim * pos_mask.float()).sum(dim=1)
    numerator = (exp_sim * pos_mask.float()).sum(dim=1)
    loss = -torch.log(numerator / denominator).mean()
    return loss


def augment_spectrum(x, noise_std=0.02, scale_range=(0.8, 1.2), mask_ratio=0.1):
    """光谱数据增强: 随机缩放 + 高斯噪声 + 随机掩码"""
    aug = x.clone()
    scale = torch.empty(x.shape[0], 1, device=x.device).uniform_(*scale_range)
    aug = aug * scale
    noise = torch.randn_like(aug) * noise_std
    aug = aug + noise
    mask = torch.rand_like(aug) > mask_ratio
    aug = aug * mask.float()
    return aug


class ContrastiveEmbedder:
    """SimCLR-style contrastive learning embedder."""

    def __init__(self, input_dim, embed_dim=30, device=None):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = ContrastiveEncoder(input_dim, embed_dim)
        self.proj_head = ProjectionHead(embed_dim)
        self.fitted = False

    def fit(self, X, epochs=150, lr=1e-3, batch_size=64, verbose=True):
        X_t = torch.from_numpy(X).float()
        dataset = torch.utils.data.TensorDataset(X_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.encoder.to(self.device)
        self.proj_head.to(self.device)
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.proj_head.parameters()), lr=lr)

        self.encoder.train()
        self.proj_head.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                aug1 = augment_spectrum(batch_x)
                aug2 = augment_spectrum(batch_x)
                h1 = self.encoder(aug1)
                h2 = self.encoder(aug2)
                z1 = self.proj_head(h1)
                z2 = self.proj_head(h2)
                loss = nt_xent_loss(z1, z2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if verbose and (epoch + 1) % 50 == 0:
                print(f"  [Contrastive] Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

        self.fitted = True
        if verbose:
            print(f"  [Contrastive] 训练完成, 最终 Loss: {total_loss/len(loader):.6f}")
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Embedder not fitted yet")
        self.encoder.eval()
        X_t = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            emb = self.encoder(X_t).cpu().numpy()
        return emb

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
