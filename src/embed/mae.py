"""
MAE Embedder — Masked Autoencoder: 随机掩码 50% 通道，重构完整光谱。
接口兼容 sklearn: fit(X) / transform(X) / fit_transform(X)
"""

import torch
import torch.nn as nn
import numpy as np


class MAEEncoder(nn.Module):
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


class MAEDecoder(nn.Module):
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


class MaskedAutoencoderEmbedder:
    """
    Masked Autoencoder embedder.
    训练时: 随机 mask 50% 的输入通道 → encode → decode → 只计算 mask 位置的 MSE loss
    推理时: 直接 encode 原始光谱 → embedding
    """

    def __init__(self, input_dim, embed_dim=30, mask_ratio=0.5, device=None):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = MAEEncoder(input_dim, embed_dim)
        self.decoder = MAEDecoder(embed_dim, input_dim)
        self.fitted = False

    def fit(self, X, epochs=150, lr=1e-3, batch_size=64, verbose=True):
        X_t = torch.from_numpy(X).float()
        dataset = torch.utils.data.TensorDataset(X_t, X_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr)
        criterion = nn.MSELoss()

        self.encoder.train()
        self.decoder.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                mask = torch.rand_like(batch_x) > self.mask_ratio
                masked_x = batch_x * mask.float()
                latent = self.encoder(masked_x)
                recon = self.decoder(latent)
                loss_mask = (~mask).float()
                loss = criterion(recon * loss_mask, batch_y * loss_mask)
                loss = loss / (loss_mask.mean() + 1e-8)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if verbose and (epoch + 1) % 50 == 0:
                print(f"  [MAE] Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

        self.fitted = True
        if verbose:
            print(f"  [MAE] 训练完成, 最终 Loss: {total_loss/len(loader):.6f}")
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
