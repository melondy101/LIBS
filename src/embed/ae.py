"""
Autoencoder Embedder — 用自编码器重构训练光谱，编码器输出作为低维嵌入。
接口兼容 sklearn: fit(X) / transform(X) / fit_transform(X)
"""

import torch
import torch.nn as nn
import numpy as np


class AEEncoder(nn.Module):
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


class AEDecoder(nn.Module):
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


class AutoencoderEmbedder:
    """Autoencoder-based spectral embedder. Replaces PCA."""

    def __init__(self, input_dim, embed_dim=30, device=None):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = AEEncoder(input_dim, embed_dim)
        self.decoder = AEDecoder(embed_dim, input_dim)
        self.fitted = False

    def fit(self, X, epochs=150, lr=1e-3, batch_size=64, verbose=True):
        X_t = torch.from_numpy(X).float()
        dataset = torch.utils.data.TensorDataset(X_t, X_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        model = nn.Sequential(self.encoder, self.decoder)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                recon = model(batch_x)
                loss = criterion(recon, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if verbose and (epoch + 1) % 50 == 0:
                print(f"  [AE] Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

        self.fitted = True
        if verbose:
            print(f"  [AE] 训练完成, 最终 Loss: {total_loss/len(loader):.6f}")
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
