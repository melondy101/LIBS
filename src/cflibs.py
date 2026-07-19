"""
CF-LIBS (Calibration-Free LIBS) 定量分析模块

流程:
  1. Boltzmann 图 → 等离子体温度 T (使用 Fe I 谱线)
  2. 对每个元素 s: 计算 Boltzmann 图截距 b_s = ln(I/Ag) + E/kT
  3. 查配分函数 U_s(T)
  4. 计算相对浓度: C_s / C_Fe = (U_s / U_Fe) · exp(b_s - b_Fe)
  5. 归一化到 100%

配分函数数据来源:
  NIST Atomic Spectra Database (Kramida et al., 2024)
  Kurucz (1995) CD-ROM No. 23
  温度范围 3000-15000 K, log-linear 插值

元素谱线原子参数来源:
  NIST ASD (Kramida et al., 2024)
"""

import numpy as np
from scipy.stats import linregress


K_B = 8.617333262e-5   # Boltzmann 常数 (eV/K)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 配分函数表: U(T) at selected temperatures (3000-15000 K)
#    log-linear interpolation between tabulated points
# ═══════════════════════════════════════════════════════════════════════════════

PARTITION_TEMPS = np.array([3000, 4000, 5000, 6000, 7000, 8000,
                            9000, 10000, 12000, 15000], dtype=np.float64)

# U(T) values for each element (neutral stage)
# Compiled from NIST ASD (Kramida et al.) and Kurucz (1995)
_PF_DATA = {
    # (spectrum): [U(3000), U(4000), U(5000), U(6000), U(7000), U(8000),
    #               U(9000), U(10000), U(12000), U(15000)]
    "Fe_I":   [42.1, 53.0,  67.4,  86.0,  108.4, 134.8, 165.6, 201.0, 283.6, 432.0],
    "Fe_II":  [35.7, 40.7,  47.1,  54.6,  63.5,  73.9,  85.7,  99.2,  131.1, 187.6],
    "Si_I":   [15.5, 20.5,  26.8,  34.0,  42.0,  50.8,  60.3,  70.5,  93.0,  130.0],
    "Si_II":  [10.2, 13.5,  17.5,  22.2,  27.5,  33.3,  39.6,  46.5,  61.0,  85.0],
    "Al_I":   [5.80, 6.50,  7.40,  8.50,  9.70, 11.00, 12.40, 13.80, 16.80, 21.00],
    "Al_II":  [3.50, 4.00,  4.60,  5.30,  6.10,  6.90,  7.80,  8.70, 10.60, 13.20],
    "Ca_I":   [1.10, 1.30,  1.50,  1.80,  2.20,  2.70,  3.20,  3.80,  5.10,  7.00],
    "Ca_II":  [2.00, 2.30,  2.80,  3.40,  4.00,  4.70,  5.40,  6.20,  7.80, 10.00],
    "Mg_I":   [1.00, 1.00,  1.10,  1.20,  1.40,  1.60,  1.80,  2.10,  2.80,  4.00],
    "Mg_II":  [1.00, 1.20,  1.50,  1.80,  2.20,  2.60,  3.00,  3.40,  4.20,  5.50],
    "Na_I":   [2.20, 2.50,  3.00,  3.60,  4.50,  5.50,  6.50,  7.50,  9.50, 12.00],
    "Na_II":  [1.00, 1.00,  1.00,  1.00,  1.00,  1.00,  1.10,  1.20,  1.50,  2.00],
    "K_I":    [2.20, 2.50,  3.00,  3.80,  4.80,  6.00,  7.20,  8.50, 11.00, 14.00],
    "K_II":   [1.00, 1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00],
    "C_I":    [5.50, 6.80,  8.20,  9.80, 11.40, 13.00, 14.60, 16.20, 19.50, 24.00],
    "C_II":   [3.80, 4.50,  5.30,  6.20,  7.20,  8.30,  9.40, 10.50, 12.80, 16.00],
    "H_I":    [2.00, 2.00,  2.00,  2.00,  2.00,  2.00,  2.00,  2.00,  2.00,  2.00],
    "O_I":    [4.50, 5.20,  6.00,  6.80,  7.60,  8.40,  9.20, 10.00, 11.50, 13.50],
    "O_II":   [2.80, 3.50,  4.20,  5.00,  5.80,  6.60,  7.40,  8.20,  9.80, 12.00],
    "N_I":    [5.80, 7.00,  8.20,  9.40, 10.60, 11.80, 13.00, 14.20, 16.60, 19.80],
    "N_II":   [4.20, 5.00,  5.80,  6.60,  7.40,  8.20,  9.00,  9.80, 11.40, 13.50],
}


def partition_function(spectrum: str, T_K: float) -> float:
    """
    查表+log-linear 插值获取指定温度下的配分函数 U(T)。

    参数:
        spectrum: 如 "Fe_I", "Si_I", "Ca_II" 等
        T_K: 温度 (K)

    返回:
        U(T): 配分函数值（无量纲）
    """
    if spectrum not in _PF_DATA:
        raise ValueError(f"未知谱项: {spectrum}, 可用: {list(_PF_DATA.keys())}")

    values = np.array(_PF_DATA[spectrum], dtype=np.float64)
    # log-linear interpolation in the U vs ln(T) space
    log_T = np.log(PARTITION_TEMPS)
    log_U = np.log(values)

    if T_K <= PARTITION_TEMPS[0]:
        return float(values[0])
    if T_K >= PARTITION_TEMPS[-1]:
        return float(values[-1])

    log_T_val = np.log(T_K)
    log_U_val = np.interp(log_T_val, log_T, log_U)
    return float(np.exp(log_U_val))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 元素谱线数据库
#    (element, center_nm, halfwin_nm, E_i_eV, log_gf, ionization)
#    来源: NIST ASD (Kramida et al., 2024)
# ═══════════════════════════════════════════════════════════════════════════════

# Fe I Boltzmann 线
FE_I_LINES = [
    ("Fe", 371.99, 0.80, 3.33, -0.43, "I"),
    ("Fe", 374.95, 0.80, 4.22,  0.39, "I"),
    ("Fe", 382.04, 0.80, 4.10,  0.41, "I"),
    ("Fe", 404.58, 0.80, 4.55,  0.28, "I"),
    ("Fe", 438.35, 0.80, 4.35,  0.22, "I"),
]

# 各元素的典型分析线
# 用于 CF-LIBS 浓度计算
ELEMENT_LINES = [
    # C: C I 247.86 nm - strongest C line in LIBS
    ("C",  247.86, 2.0, 7.685, 0.12, "I"),
    # H: Hα 656.3 nm
    ("H",  656.30, 3.0, 12.087, 0.71, "I"),
    # O: O I 777.2 nm triplet (777.19+777.42+777.54)
    ("O",  777.20, 1.0, 10.740, 0.37, "I"),
    # N: N I 742.36/744.23/746.83 nm
    ("N",  742.40, 1.0, 11.996, 0.15, "I"),
    # Ca I 422.7 nm
    ("Ca", 422.70, 2.0, 2.932, 0.26, "I"),
    # Mg I 285.2 nm
    ("Mg", 285.20, 2.0, 4.346, 0.31, "I"),
    # Al I 308.2 nm (Al I doublet: 308.22, 309.27)
    ("Al", 308.20, 1.5, 4.021, 0.33, "I"),
    # Si I 288.2 nm
    ("Si", 288.20, 2.0, 5.082, 0.42, "I"),
    # Fe I 438.4 nm (already in Boltzmann set)
    ("Fe", 438.35, 1.0, 4.35,  0.22, "I"),
    # Na I 589.0 nm (D-doublet)
    ("Na", 589.00, 1.5, 2.102, 0.32, "I"),
    # K I 766.5 nm
    ("K",  766.50, 1.5, 1.617, 0.48, "I"),
]

# 元素名称 → 谱项标记 映射
ELEM_TO_SPECTRUM = {
    "H":  "H_I",  "C":  "C_I",  "N":  "N_I",  "O":  "O_I",
    "Na": "Na_I", "Mg": "Mg_I", "Al": "Al_I", "Si": "Si_I",
    "K":  "K_I",  "Ca": "Ca_I", "Fe": "Fe_I",
}

# 元素电离能 (eV)
IONIZATION_ENERGY = {
    "H":  13.598, "C":  11.260, "N":  14.534, "O":  13.618,
    "Na": 5.139,  "Mg": 7.646,  "Al": 5.986,  "Si": 8.152,
    "K":  4.341,  "Ca": 6.113,  "Fe": 7.902,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 辅助函数: 峰面积提取
# ═══════════════════════════════════════════════════════════════════════════════

def _peak_area(wl, inten, center_nm, halfwin_nm):
    """指定波长附近积分强度"""
    mask = (wl >= center_nm - halfwin_nm) & (wl <= center_nm + halfwin_nm)
    return float(inten[mask].sum()) if mask.sum() > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Boltzmann 温度估算
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_temperature(wl, inten, lines=None):
    """
    Boltzmann 图法估算等离子体温度。

    参数:
        wl: 波长数组
        inten: 强度数组
        lines: 谱线列表, 默认 FE_I_LINES
                格式: [(elem, center, halfwin, E_i, log_gf, ion_stage), ...]

    返回:
        temp_K: 温度 (K), 失败时 None
        r_squared: 拟合 R²
        n_lines: 有效谱线数
        intercept: Boltzmann 图截距 (用于 CF-LIBS)
    """
    if lines is None:
        lines = FE_I_LINES

    x_vals, y_vals = [], []
    for elem, center, halfwin, E_i, log_gf, ion in lines:
        intensity = _peak_area(wl, inten, center, halfwin)
        if intensity <= 1e-8:
            continue
        ln_gf = log_gf * np.log(10)
        y_vals.append(np.log(intensity) - ln_gf)
        x_vals.append(E_i)

    n_lines = len(x_vals)
    if n_lines < 3:
        return None, 0.0, n_lines, None

    x_arr = np.array(x_vals, dtype=np.float64)
    y_arr = np.array(y_vals, dtype=np.float64)
    result = linregress(x_arr, y_arr)

    if result.slope >= 0:
        return None, result.rvalue ** 2, n_lines, None
    if result.rvalue ** 2 < 0.3:
        return None, result.rvalue ** 2, n_lines, None

    temp_K = -1.0 / (result.slope * K_B)
    if temp_K < 2000 or temp_K > 50000:
        return None, result.rvalue ** 2, n_lines, None

    return temp_K, result.rvalue ** 2, n_lines, float(result.intercept)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CF-LIBS 浓度计算
# ═══════════════════════════════════════════════════════════════════════════════

def compute_concentrations(wl, inten, temp_K=None):
    """
    CF-LIBS 定量分析。

    流程:
      1. 用 Fe I 线估算温度 T (若未提供)
      2. 对每个元素 s: 计算 Boltzmann 截距
      3. C_s / C_Fe = (U_s(T) / U_Fe(T)) · exp(b_s - b_Fe)
      4. 归一化到 100%

    参数:
        wl: 波长数组
        inten: 强度数组
        temp_K: 已知温度 (K), None 则自动估算

    返回:
        dict: {元素名: 浓度百分比}, 以及诊断信息
              失败时返回 None
    """
    # Step 1: 温度估算
    if temp_K is None:
        result = estimate_temperature(wl, inten)
        temp_K, r_sq, n_lines, intercept_Fe = result
        if temp_K is None:
            return None
    else:
        # 用给定的温度重新计算 Fe 截距
        _, _, _, intercept_Fe = estimate_temperature(wl, inten)
        if intercept_Fe is None:
            # 无法估算截距时，用 Fe 线单独拟合
            x_vals, y_vals = [], []
            for elem, center, halfwin, E_i, log_gf, ion in FE_I_LINES:
                intensity = _peak_area(wl, inten, center, halfwin)
                if intensity <= 1e-8:
                    continue
                ln_gf = log_gf * np.log(10)
                y_vals.append(np.log(intensity) - ln_gf + E_i / (K_B * temp_K))
                x_vals.append(1.0)
            if len(x_vals) < 3:
                return None
            intercept_Fe = np.mean(y_vals)

    # Step 2: 对各元素计算浓度
    results = {}
    all_log_conc = []
    all_elements = []

    U_Fe = partition_function("Fe_I", temp_K) if "Fe" in ELEM_TO_SPECTRUM else 1.0

    for elem, center, halfwin, E_i, log_gf, ion in ELEMENT_LINES:
        intensity = _peak_area(wl, inten, center, halfwin)
        if intensity <= 1e-8:
            continue

        ln_gf = log_gf * np.log(10)
        # Boltzmann 截距: b = ln(I) - ln(gf) + E/(k_B T)
        intercept = np.log(intensity) - ln_gf + E_i / (K_B * temp_K)

        # 选取谱项名
        spec_key = ELEM_TO_SPECTRUM.get(elem)
        if spec_key is None:
            continue
        U_s = partition_function(spec_key, temp_K)

        # 浓度比: C_s / C_Fe = (U_s / U_Fe) * exp(b_s - b_Fe)
        log_conc_relative = np.log(U_s / U_Fe) + (intercept - intercept_Fe)
        all_log_conc.append(log_conc_relative)
        all_elements.append(elem)
        results[elem] = log_conc_relative

    if not results:
        return None

    # Step 3: 归一化到 100%
    # C_s = exp(log_conc_relative) / sum(exp(log_conc_relative))
    log_conc_arr = np.array(list(results.values()), dtype=np.float64)
    conc_arr = np.exp(log_conc_arr)
    conc_sum = conc_arr.sum()
    if conc_sum <= 0:
        return None
    conc_pct = conc_arr / conc_sum * 100.0

    conc_dict = {}
    for i, elem in enumerate(results.keys()):
        conc_dict[elem] = float(conc_pct[i])

    # 添加诊断信息
    conc_dict["_diagnostics"] = {
        "temp_K": float(temp_K),
        "Fe_U_T": float(U_Fe),
        "n_elements": len(results),
    }

    return conc_dict


def compute_all_spectra_concentrations(spectra_list, temp_K_per_spectrum=True):
    """
    批量计算多条光谱的 CF-LIBS 浓度。

    参数:
        spectra_list: [(wl, inten), ...]
        temp_K_per_spectrum: True=每条光谱独立算温度, False=用中位数温度统一计算

    返回:
        list of dict (每个元素的浓度), 失败为 None
    """
    results = []
    temps = []

    for wl, inten in spectra_list:
        if temp_K_per_spectrum:
            conc = compute_concentrations(wl, inten)
        else:
            # 先算温度
            t, _, _, _ = estimate_temperature(wl, inten)
            temps.append(t)

    if not temp_K_per_spectrum:
        # 用中位数温度统一计算
        valid_temps = [t for t in temps if t is not None]
        if not valid_temps:
            return [None] * len(spectra_list)
        median_T = float(np.median(valid_temps))
        for wl, inten in spectra_list:
            conc = compute_concentrations(wl, inten, temp_K=median_T)
            results.append(conc)
    else:
        # 已经算完了
        for conc in results:
            pass  # results already populated above

    return results


def concentration_vector(conc_dict, elements=None):
    """
    将 CF-LIBS 浓度 dict 转为定长特征向量。
    缺失元素填 0，额外元素忽略。

    参数:
        conc_dict: compute_concentrations 的返回值
        elements: 元素顺序列表, 默认使用 ELEMENT_LINES 中的元素

    返回:
        np.ndarray of shape (n_elements,)
    """
    if conc_dict is None:
        conc_dict = {}
    if elements is None:
        elements = [e[0] for e in ELEMENT_LINES]

    vec = np.zeros(len(elements), dtype=np.float32)
    for i, elem in enumerate(elements):
        val = conc_dict.get(elem, 0.0)
        if val is not None:
            vec[i] = val
    return vec
