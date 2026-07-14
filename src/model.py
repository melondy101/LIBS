"""
两阶段回归模型

Stage 1 — 光谱特征 → 辅助指标 (全水分/灰分/氢/硫)
  - 辅助指标与发热量相关性强，但比发热量本身更稳定
  - 用 OOF(Out-of-Fold) 预测避免数据泄露进 Stage2

Stage 2 — 光谱特征 + Stage1预测辅助指标 → 发热量
  - 信息融合: 数值模型 + 物理知识

均值收缩 (Mean Shrinkage):
  - 批次极少(≤10)的煤种，模型容易过拟合到训练批次的特殊性
  - 将预测值向煤种均值收缩: pred = w * model_pred + (1-w) * coal_mean
  - w 通过 OOF 验证集上网格搜索确定
"""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold

from config import ALPHAS, AUX_COLS, SMALL_BATCH_THRESHOLD
from src.features import build_feature_matrix


# ── CV 分割策略 ───────────────────────────────────────────────────────────────

def get_cv_splits(groups, n_batches):
    """
    批次数 ≤ 10 时用 LOOCV（留一批次），否则用 GroupKFold(5折)。
    关键: 以批次(group)为单位分割，而非以单条光谱为单位——
    同一批次的光谱不能同时出现在训练集和验证集中。
    """
    dummy = np.zeros(len(groups))
    if n_batches <= SMALL_BATCH_THRESHOLD:
        return list(LeaveOneGroupOut().split(dummy, dummy, groups))
    else:
        k = min(5, n_batches)
        return list(GroupKFold(n_splits=k).split(dummy, dummy, groups))


# ── 批次聚合 ──────────────────────────────────────────────────────────────────

def aggregate_to_batch(preds, names):
    """
    同批次多条光谱的预测值 → 取中位数作为批次预测。
    中位数比均值对异常光谱更鲁棒。
    """
    grouped = {}
    for p, n in zip(preds, names):
        grouped.setdefault(n, []).append(p)
    return {k: float(np.median(v)) for k, v in grouped.items()}


# ── 均值收缩搜索 ──────────────────────────────────────────────────────────────

def _search_shrink_w(oof_preds, targets, groups, coal_mean, w_candidates):
    """
    在 OOF 验证集上搜索最优收缩权重 w。
    将模型预测向煤种均值收缩: pred = w * oof_pred + (1-w) * coal_mean
    """
    batch_oof = aggregate_to_batch(oof_preds, groups)
    batch_tgt = aggregate_to_batch(targets, groups)
    best_w, best_rmse = 1.0, float("inf")
    for w in w_candidates:
        shrunk = {k: w * batch_oof[k] + (1 - w) * coal_mean for k in batch_oof}
        rmse = np.sqrt(np.mean([(shrunk[k] - batch_tgt[k]) ** 2 for k in batch_tgt]))
        if rmse < best_rmse:
            best_rmse, best_w = rmse, w
    return best_w, best_rmse


# ── 两阶段训练 ────────────────────────────────────────────────────────────────

def train_coal_model(coal_type, data, embedder=None):
    """
    对单个煤种训练两阶段模型。

    参数:
        coal_type: 煤种名
        data: 由 load_coal_spectra 返回的数据字典
        embedder: 预训练的嵌入模型 (AE/MAE/VAE/Contrastive)
                  不传时使用 PCA (默认)

    返回:
        dict: 包含所有模型组件和 CV-RMSE
    """
    n_batches = data['n_batches']
    groups    = data['groups']
    targets   = data['targets']
    aux_tgts  = data['aux']

    print(f"\n  [{coal_type}] 构建特征矩阵...")
    X_spec, scaler_spec, embedder, scaler_hand = build_feature_matrix(
        data, n_batches,
        scaler_spec=None, embedder=embedder, scaler_hand=None, fit=True)

    print(f"  [{coal_type}] 特征矩阵形状: {X_spec.shape}")

    # ── Stage 1: 辅助指标回归 ─────────────────────────────────────────
    print(f"  [{coal_type}] Stage 1: 辅助指标回归...")
    aux_models = {}
    for col_idx, col_name in enumerate(AUX_COLS):
        y_aux = aux_tgts[:, col_idx]
        valid = ~np.isnan(y_aux)
        if valid.sum() < 5:
            print(f"    ...跳过 {col_name} (有效样本={valid.sum()})")
            continue
        model = RidgeCV(alphas=ALPHAS, scoring="neg_mean_squared_error")
        model.fit(X_spec[valid], y_aux[valid])
        aux_models[col_name] = model

    # ── Stage 2: 发热量回归（OOF 预测辅助指标） ──────────────────────
    print(f"  [{coal_type}] Stage 2: 发热量回归 + 均值收缩...")

    cv_splits = get_cv_splits(groups, n_batches)
    oof_preds = np.zeros(len(targets), dtype=np.float32)
    s2_models = []

    for train_idx, val_idx in cv_splits:
        # Stage 1 在训练折上重训辅助模型
        aux_inner = {}
        for col_idx, col_name in enumerate(AUX_COLS):
            y_aux = aux_tgts[train_idx, col_idx]
            valid = ~np.isnan(y_aux)
            if valid.sum() < 5:
                continue
            m = RidgeCV(alphas=ALPHAS, scoring="neg_mean_squared_error")
            m.fit(X_spec[train_idx][valid], y_aux[valid])
            aux_inner[col_name] = m

        # Stage 1 预测验证折的辅助指标
        pred_aux_val = np.zeros((len(val_idx), len(AUX_COLS)), dtype=np.float32)
        for col_idx, col_name in enumerate(AUX_COLS):
            m = aux_inner.get(col_name)
            if m is not None:
                pred_aux_val[:, col_idx] = m.predict(X_spec[val_idx])

        # Stage 2 输入: 光谱特征 + 验证折的辅助指标预测
        X_s2_val = np.hstack([X_spec[val_idx], pred_aux_val])
        y_s2_val = targets[val_idx]

        # Stage 2 在训练折上训练
        pred_aux_train = np.zeros((len(train_idx), len(AUX_COLS)), dtype=np.float32)
        for col_idx, col_name in enumerate(AUX_COLS):
            m = aux_inner.get(col_name)
            if m is not None:
                pred_aux_train[:, col_idx] = m.predict(X_spec[train_idx])
        X_s2_train = np.hstack([X_spec[train_idx], pred_aux_train])
        y_s2_train = targets[train_idx]

        scaler_s2 = StandardScaler()
        X_s2_train_scaled = scaler_s2.fit_transform(np.nan_to_num(X_s2_train))
        X_s2_val_scaled   = scaler_s2.transform(np.nan_to_num(X_s2_val))

        model_s2 = RidgeCV(alphas=ALPHAS, scoring="neg_mean_squared_error")
        model_s2.fit(X_s2_train_scaled, y_s2_train)
        oof_preds[val_idx] = model_s2.predict(X_s2_val_scaled)
        s2_models.append((scaler_s2, model_s2))

    # ── 均值收缩 ─────────────────────────────────────────────────────────
    coal_mean = float(targets.mean())
    w_candidates = np.linspace(0.1, 1.0, 10)
    best_w, shrink_rmse = _search_shrink_w(oof_preds, targets, groups, coal_mean, w_candidates)

    cv_oof = aggregate_to_batch(oof_preds, groups)
    cv_tgt = aggregate_to_batch(targets, groups)
    cv_rmse = float(np.sqrt(np.mean([(cv_oof[k] - cv_tgt[k]) ** 2 for k in cv_tgt])))
    print(f"  [{coal_type}] CV-RMSE={cv_rmse:.2f}, 收缩w={best_w:.2f}")

    # ── 用全量数据训练最终模型 ────────────────────────────────────────
    scaler_s2 = StandardScaler()
    pred_aux_all = np.zeros((len(targets), len(AUX_COLS)), dtype=np.float32)
    for col_idx, col_name in enumerate(AUX_COLS):
        m = aux_models.get(col_name)
        if m is not None:
            pred_aux_all[:, col_idx] = m.predict(X_spec)
    X_s2_all   = np.hstack([X_spec, pred_aux_all])
    X_s2_all   = scaler_s2.fit_transform(np.nan_to_num(X_s2_all))
    final_model = RidgeCV(alphas=ALPHAS, scoring="neg_mean_squared_error")
    final_model.fit(X_s2_all, targets)

    return {
        'spec_scalers': (scaler_spec, embedder, scaler_hand),
        'aux_models':   aux_models,
        'scaler_s2':    scaler_s2,
        'final_model':  final_model,
        'coal_mean':    coal_mean,
        'shrink_w':     best_w,
        'cv_rmse':      cv_rmse,
    }


# ── 单煤种推理 ────────────────────────────────────────────────────────────────

def predict_coal(coal_type, test_data, model_dict):
    """
    对某煤种的测试批次做推理，返回 {批次名: 预测发热量} 字典。
    推理流程与训练流程一一对应:
      光谱 → [Stage1] → 预测辅助指标 → [Stage2] → 发热量 → 批次聚合
    """
    scaler_spec, embedder, scaler_hand = model_dict['spec_scalers']
    aux_models  = model_dict['aux_models']
    scaler_s2   = model_dict['scaler_s2']
    final_model = model_dict['final_model']
    coal_mean   = model_dict['coal_mean']
    shrink_w    = model_dict['shrink_w']

    X_spec = build_feature_matrix(
        test_data, n_batches=None,
        scaler_spec=scaler_spec, embedder=embedder, scaler_hand=scaler_hand,
        fit=False)

    pred_aux = np.zeros((len(test_data['spectra']), len(AUX_COLS)), dtype=np.float32)
    for col_idx, col_name in enumerate(AUX_COLS):
        m = aux_models.get(col_name)
        if m is not None:
            pred_aux[:, col_idx] = m.predict(X_spec)

    X_s2       = scaler_s2.transform(np.nan_to_num(np.hstack([X_spec, pred_aux])))
    preds      = final_model.predict(X_s2)
    batch_pred = aggregate_to_batch(preds, test_data['names'])

    # 应用收缩（仅小批次煤种，且 w < 1.0 时）
    if shrink_w < 1.0:
        batch_pred = {
            k: shrink_w * v + (1 - shrink_w) * coal_mean
            for k, v in batch_pred.items()
        }

    return batch_pred
