"""
等离子体温度 / 谱线比值特征模块

双策略:
  1. Boltzmann 图法 — 用多条 Fe I 谱线拟合等离子体温度
  2. 线强比值法 — 绕原子参数不确定性，直接提取温度敏感特征

Boltzmann 原理:
  ln(I / (g·A)) = -E / (k_B·T) + C
  斜率 = -1/(k_B·T) ⇒ T = -1 / (slope × k_B)

线强比值特征:
  - Ca II(393.4) / Ca I(422.7) — 离子/中性比, 与温度正相关
  - Hα / C I — 激发条件指示
  - Si / Mg — 基体效应代理
"""

import numpy as np
from scipy.stats import linregress


K_B = 8.617333262e-5   # Boltzmann 常数 (eV/K)

# Boltzmann Fe I 谱线: (元素, 中心nm, 窗口nm, E_i eV, log_gf)
# 选自 NIST ASD, 370-440nm 区间较强且孤立的 Fe I 线
BOLTZMANN_FE_LINES = [
    ("Fe", 371.99, 0.8, 3.33, -0.43),
    ("Fe", 374.95, 0.8, 4.22,  0.39),
    ("Fe", 382.04, 0.8, 4.10,  0.41),
    ("Fe", 404.58, 0.8, 4.55,  0.28),
    ("Fe", 438.35, 1.0, 4.35,  0.22),
]

# 线强比值配置: (名称, 中心1, 中心2, 窗口nm)
RATIO_CONFIG = [
    ("CaII_CaI",  393.4,  422.7, 1.0),   # 离子/中性温度指示
    ("Halpha_C",  656.3,  247.86, 1.5),   # 激发条件
    ("Si_Mg",     288.2,  285.2, 1.0),    # 基体效应
]


def _peak_area(wl, inten, center_nm, halfwin_nm):
    """
    提取指定波长附近的总强度（峰面积）。
    不做背景扣除（减少对邻近谱线的敏感性）。
    """
    mask = (wl >= center_nm - halfwin_nm) & (wl <= center_nm + halfwin_nm)
    return float(inten[mask].sum()) if mask.sum() > 0 else 0.0


def estimate_boltzmann_temperature(wl, inten):
    """
    Boltzmann 图法估算等离子体温度。

    返回:
        temp_K:    温度 (K), 不可靠时 None
        r_squared: 拟合 R²
        n_lines:   参与拟合的谱线数
    """
    x_vals, y_vals = [], []

    for elem, center, halfwin, E_i, log_gf in BOLTZMANN_FE_LINES:
        intensity = _peak_area(wl, inten, center, halfwin)
        if intensity <= 1e-6:
            continue
        ln_gf = log_gf * np.log(10)
        y_vals.append(np.log(intensity) - ln_gf)
        x_vals.append(E_i)

    n_lines = len(x_vals)
    if n_lines < 3:
        return None, 0.0, n_lines

    x_arr, y_arr = np.array(x_vals, dtype=np.float64), np.array(y_vals, dtype=np.float64)
    result = linregress(x_arr, y_arr)

    if result.slope >= 0:
        return None, result.rvalue ** 2, n_lines
    if result.rvalue ** 2 < 0.3:
        return None, result.rvalue ** 2, n_lines

    temp_K = -1.0 / (result.slope * K_B)
    if temp_K < 2000 or temp_K > 50000:
        return None, result.rvalue ** 2, n_lines

    return temp_K, result.rvalue ** 2, n_lines


def compute_spectral_ratios(wl, inten):
    """
    计算预定义的线强比值特征。

    返回 dict {名称: 比值}
    """
    results = {}
    for name, c1, c2, hw in RATIO_CONFIG:
        a1 = _peak_area(wl, inten, c1, hw)
        a2 = _peak_area(wl, inten, c2, hw)
        results[name] = a1 / (a2 + 1e-12)
    return results


def compute_boltzmann_feature(wl, inten):
    """
    组合计算: Boltzmann 温度 + 线强比值特征。

    返回:
        (temp_K, r_sq, quality, ca_ratio, hc_ratio, si_mg_ratio)
      各字段说明见 compute_boltzmann_feature_v2 返回值。
    """
    temp_K, r_sq, n_lines = estimate_boltzmann_temperature(wl, inten)
    ratios = compute_spectral_ratios(wl, inten)

    if temp_K is None:
        temp_out = np.nan
        quality = 0.0
    else:
        quality = 1.0 if (r_sq >= 0.4 and n_lines >= 4) else 0.0
        temp_out = float(temp_K)

    return (
        temp_out,                          # 0: 温度(K), NaN 不可靠
        float(r_sq),                       # 1: 拟合 R²
        quality,                           # 2: 质量标志
        float(ratios["CaII_CaI"]),         # 3: Ca II/Ca I
        float(ratios["Halpha_C"]),         # 4: Hα/C
        float(ratios["Si_Mg"]),            # 5: Si/Mg
    )

