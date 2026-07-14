"""
特征工程模块

LIBS 光谱特征分三层:
  1. 统计特征   — 均值/方差/偏度/峰度/熵/分位数/导数
  2. 谱线特征   — 关键元素谱线积分（绝对值 & 相对归一化）
  3. 物理比值   — 可燃元素/灰分比，H/C 比等

三层拼接后经过嵌入降维 + 标准化，作为 Ridge 模型的输入。
"""

import numpy as np
from scipy.stats import skew, kurtosis, entropy as scipy_entropy
from sklearn.preprocessing import StandardScaler

from config import KEY_LINES, RANDOM_STATE


# ── 谱线积分 ──────────────────────────────────────────────────────────────────

def line_integral(wl, inten, center_nm, halfwin_nm):
    mask = (wl >= center_nm - halfwin_nm) & (wl <= center_nm + halfwin_nm)
    return float(inten[mask].sum()) if mask.sum() > 0 else 0.0


# ── 单条光谱特征 ──────────────────────────────────────────────────────────────

def spectrum_features(wl, inten):
    total      = inten.sum() + 1e-8
    inten_norm = inten / total
    deriv      = np.diff(inten_norm)
    deriv2     = np.diff(inten_norm, 2)

    stats = np.array([
        inten.mean(), inten.std(), inten.max(), np.log1p(total),
        inten_norm.mean(), inten_norm.std(),
        float(skew(inten_norm)),
        float(kurtosis(inten_norm)),
        float(scipy_entropy(inten_norm + 1e-12)),
        np.percentile(inten, 10), np.percentile(inten, 25),
        np.percentile(inten, 75), np.percentile(inten, 90),
        deriv.std(),  float(np.abs(deriv).mean()),
        deriv2.std(), float(np.abs(deriv2).mean()),
    ], dtype=np.float32)

    line_names = list(KEY_LINES.keys())
    labs = np.array(
        [line_integral(wl, inten, c, h) for _, (c, h) in KEY_LINES.items()],
        dtype=np.float32
    )
    lrel = labs / total

    comb_sum = labs[[0, 1, 2, 3]].sum()
    ash_sum  = labs[[4, 5, 6, 7, 8, 9]].sum() + 1e-8
    rats = np.array([
        comb_sum / ash_sum,
        labs[1] / ash_sum,
        labs[0] / ash_sum,
        labs[1] / (labs[0] + 1e-8),
    ], dtype=np.float32)

    return inten_norm, stats, labs, lrel, rats


# ── 批量特征计算 ──────────────────────────────────────────────────────────────

def compute_features(data):
    raw_spectra = data['spectra']
    inorms, stats_list, labs_list, lrel_list, rats_list = [], [], [], [], []

    for wl, inten in raw_spectra:
        inorm, stats, labs, lrel, rats = spectrum_features(wl, inten)
        inorms.append(inorm)
        stats_list.append(stats)
        labs_list.append(labs)
        lrel_list.append(lrel)
        rats_list.append(rats)

    min_len = min(len(s) for s in inorms)
    inorm_mat = np.array([s[:min_len] for s in inorms], dtype=np.float32)

    data['stats'] = np.array(stats_list)
    data['labs']  = np.array(labs_list)
    data['lrel']  = np.array(lrel_list)
    data['rats']  = np.array(rats_list)

    return inorm_mat


# ── 嵌入降维 + 拼接 ──────────────────────────────────────────────────────────

def build_feature_matrix(data, n_batches,
                         scaler_spec=None, embedder=None, scaler_hand=None, fit=True):
    """
    构建最终特征矩阵: [嵌入降维(归一化光谱)] + [手工特征]。

    embedder: 对象必须实现 .fit(X) / .transform(X) / .fit_transform(X)

    fit=True  : 拟合 scaler；若 embedder 未拟合则拟合，否则只 transform
    fit=False : 用已拟合的 scaler/embedder 变换
    """
    inorm_mat = compute_features(data)

    if fit:
        scaler_spec = StandardScaler()
        spec_scaled = scaler_spec.fit_transform(inorm_mat)
        if embedder.fitted:
            spec_embed = embedder.transform(spec_scaled)
        else:
            spec_embed = embedder.fit_transform(spec_scaled)
    else:
        spec_embed = embedder.transform(scaler_spec.transform(inorm_mat))

    hand_feats = np.hstack([data['stats'], data['labs'], data['lrel'], data['rats']])
    X = np.hstack([spec_embed, hand_feats])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if fit:
        scaler_hand = StandardScaler()
        X = scaler_hand.fit_transform(X)
        return X, scaler_spec, embedder, scaler_hand
    else:
        return scaler_hand.transform(X)
