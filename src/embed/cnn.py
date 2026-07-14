"""
CNN Autoencoder Embedder — 使用 1D-CNN 编码器提取光谱局部特征。

架构:
  Encoder: Conv1d(7→5→3) → AdaptiveAvgPool → Linear → embed_dim
  Decoder: MLP FC (与 AE 相同，线性层重构)

接口兼容 sklearn: fit(X) / transform(X) / fit_transform(X)
"""

import torch
import torch.nn as nn
import numpy as np


class CNNEncoder(nn.Module):
    """
    1D-CNN 编码器: 通过多层卷积提取局部光谱模式，压缩到嵌入向量。

    卷积设计:
      k=7,s=2  →  半波长窗口捕获谱线形状
      k=5,s=2  →  中等感受野
      k=3,s=2  →  精细结构
      AdaptiveAvgPool → 全局平均池化
    """
    def __init__(self, input_dim=7305, embed_dim=30):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        # x: (batch, input_dim) → unsqueeze → (batch, 1, input_dim)
        x = x.unsqueeze(1)
        x = self.conv(x)     # (batch, 128, 1)
        x = x.squeeze(-1)    # (batch, 128)
        x = self.fc(x)       # (batch, embed_dim)
        return x


class CNNDecoder(nn.Module):
    """MLP 解码器: 从嵌入向量重构全光谱。"""
    def __init__(self, embed_dim=30, output_dim=7305):
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


class CNNEmbedder:
    """
    1D-CNN Autoencoder embedder.
    编码器为 1D-CNN 提取局部光谱模式，解码器为 MLP 重构。
    """

    def __init__(self, input_dim, embed_dim=30, device=None):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = CNNEncoder(input_dim, embed_dim)
        self.decoder = CNNDecoder(embed_dim, input_dim)
        self.fitted = False

    def fit(self, X, epochs=150, lr=1e-3, batch_size=64, verbose=True):
        """Train CNN autoencoder on normalized spectra X."""
        X_t = torch.from_numpy(X).float()
        dataset = torch.utils.data.TensorDataset(X_t, X_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        criterion = nn.MSELoss()

        self.encoder.train()
        self.decoder.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                latent = self.encoder(batch_x)
                recon = self.decoder(latent)
                loss = criterion(recon, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if verbose and (epoch + 1) % 50 == 0:
                print(f"  [CNN] Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

        self.fitted = True
        if verbose:
            print(f"  [CNN] 训练完成, 最终 Loss: {total_loss/len(loader):.6f}")
        return self

    def transform(self, X):
        """Encode spectra to embedding vectors."""
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
