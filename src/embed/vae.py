"""
VAE Embedder — 变分自编码器: 使用 KL 散度 + 重构损失训练，编码器均值作为嵌入。
接口兼容 sklearn: fit(X) / transform(X) / fit_transform(X)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VAEEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=30):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.mean = nn.Linear(256, embed_dim)
        self.log_var = nn.Linear(256, embed_dim)

    def forward(self, x):
        h = self.shared(x)
        return self.mean(h), self.log_var(h)


class VAEDecoder(nn.Module):
    def __init__(self, embed_dim=30, output_dim=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


def vae_loss(recon, x, mu, log_var):
    recon_loss = F.mse_loss(recon, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss


class VAEEmbedder:
    """Variational Autoencoder embedder. Uses encoder mean as embedding."""

    def __init__(self, input_dim, embed_dim=30, device=None):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = VAEEncoder(input_dim, embed_dim)
        self.decoder = VAEDecoder(embed_dim, input_dim)
        self.fitted = False

    def fit(self, X, epochs=150, lr=1e-3, batch_size=64, verbose=True):
        X_t = torch.from_numpy(X).float()
        dataset = torch.utils.data.TensorDataset(X_t, X_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr)

        self.encoder.train()
        self.decoder.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                mu, log_var = self.encoder(batch_x)
                z = reparameterize(mu, log_var)
                recon = self.decoder(z)
                loss = vae_loss(recon, batch_y, mu, log_var)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if verbose and (epoch + 1) % 50 == 0:
                print(f"  [VAE] Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.2f}")

        self.fitted = True
        if verbose:
            print(f"  [VAE] 训练完成, 最终 Loss: {total_loss/len(loader):.2f}")
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Embedder not fitted yet")
        self.encoder.eval()
        X_t = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            mu, _ = self.encoder(X_t)
            emb = mu.cpu().numpy()
        return emb

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
