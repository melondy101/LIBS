"""
批次内光谱异常检测模块

对每个批次内的光谱，用多准则判断哪些是污染/异常光谱并剔除。
仅在训练集上执行，不污染测试集。

准则:
  1. 谱形相关性 — 每条与批次中值光谱的 Pearson 相关系数
  2. 原始总强度 — 归一化前的 intensity.sum() z-score
  3. C/H 谱线比 — lrel[C] / lrel[H] 的 z-score
"""

import numpy as np


def _pearson(x, y):
    """快速 Pearson 相关系数。"""
    xm, ym = x - x.mean(), y - y.mean()
    num = (xm * ym).sum()
    den = np.sqrt((xm ** 2).sum() * (ym ** 2).sum()) + 1e-12
    return num / den


def _batch_median_spectrum(spectra):
    """计算批次内所有光谱的中值谱（按波长对齐）。"""
    min_len = min(s.shape[0] for _, s in spectra)
    stack = np.array([s[:min_len] for _, s in spectra], dtype=np.float32)
    return np.median(stack, axis=0)


def filter_anomalies(data, corr_mad=3.0, intensity_z=3.0, ratio_z=3.0, verbose=True):
    """
    对 data dict 做批次内异常检测，原地剔除异常光谱。
    """
    groups      = data['groups']
    raw_spectra = data['spectra']
    has_targets = data['targets'] is not None
    has_aux     = data['aux'] is not None

    stats_is_placeholder = (isinstance(data['stats'], list) and
                            len(data['stats']) > 0 and data['stats'][0] is None)

    keep_mask = np.ones(len(raw_spectra), dtype=bool)
    unique_groups = np.unique(groups)

    for g in unique_groups:
        idx = np.where(groups == g)[0]
        if len(idx) < 3:
            continue

        # --- 准则①: 谱形相关性 ---
        batch_spectra = [raw_spectra[i] for i in idx]
        median_spec = _batch_median_spectrum(batch_spectra)
        corrs = np.array([_pearson(raw_spectra[i][1], median_spec) for i in idx])
        med_corr = np.median(corrs)
        mad_corr = np.median(np.abs(corrs - med_corr)) * 1.4826
        corr_thresh = med_corr - corr_mad * mad_corr
        for j, i in enumerate(idx):
            if corrs[j] < corr_thresh:
                keep_mask[i] = False

        # --- 准则②: 原始总强度 ---
        total_intens = np.array([raw_spectra[i][1].sum() for i in idx])
        if total_intens.std() > 1e-6:
            z_intens = (total_intens - total_intens.mean()) / total_intens.std()
            for j, i in enumerate(idx):
                if abs(z_intens[j]) > intensity_z:
                    keep_mask[i] = False

        # --- 准则③: C/H 线强比 ---
        if not stats_is_placeholder:
            line_names = ['C', 'H', 'O', 'N', 'Ca', 'Ca2', 'Mg', 'Al', 'Si', 'Fe', 'Na']
            batch_lrel = np.array([data['lrel'][i] for i in idx])
            c_idx = line_names.index('C')
            h_idx = line_names.index('H')
            c_over_h = batch_lrel[:, c_idx] / (batch_lrel[:, h_idx] + 1e-12)
            if c_over_h.std() > 1e-6:
                z_ratio = (c_over_h - c_over_h.mean()) / c_over_h.std()
                for j, i in enumerate(idx):
                    if abs(z_ratio[j]) > ratio_z:
                        keep_mask[i] = False

    n_removed = (~keep_mask).sum()
    if verbose:
        print(f"  异常检测: {len(keep_mask)} 条光谱中移除 {n_removed} 条 (谱形<{corr_mad}MAD | 强度>{intensity_z}sigma | 比>{ratio_z}sigma)")

    if n_removed == 0:
        return

    # --- 原地裁剪 ---
    data['spectra'] = [s for s, keep in zip(data['spectra'], keep_mask) if keep]
    data['names']   = [n for n, keep in zip(data['names'], keep_mask) if keep]
    data['groups']  = data['groups'][keep_mask]

    batch_names = data['names']
    unique_batches = sorted(set(batch_names), key=lambda x: batch_names.index(x))
    batch_map = {b: i for i, b in enumerate(unique_batches)}
    data['groups'] = np.array([batch_map[n] for n in batch_names])
    data['n_batches'] = len(unique_batches)

    if has_targets:
        data['targets'] = data['targets'][keep_mask]
    if has_aux:
        data['aux'] = data['aux'][keep_mask]

    if not stats_is_placeholder:
        for key in ['stats', 'labs', 'lrel', 'rats']:
            if isinstance(data[key], np.ndarray):
                data[key] = data[key][keep_mask]
    else:
        for key in ['stats', 'labs', 'lrel', 'rats']:
            if isinstance(data[key], list):
                data[key] = [v for v, keep in zip(data[key], keep_mask) if keep]
