"""
主入口 — 运行方式: python train.py

流程:
  1. 加载标签
  1.5 预训练嵌入模型 (MAE)
  2. 按煤种分别训练两阶段模型
  3. 记录实验日志（timestamp + CV-RMSE）
  4. 测试集推理
  5. 打包 submit.zip
"""

import sys
import os
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import COAL_TYPES, TRAIN_DIR, TEST_DIR
from src.data   import load_labels, load_coal_spectra
from src.model  import train_coal_model, predict_coal
from src.submit import pack_submission
from src.experiment_tracker import log_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--treatment", type=str, default="",
                        help="实验处理描述，写入 experiment_log.csv")
    args = parser.parse_args()
    # ── Step 1: 加载标签 ──────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: 加载标签")
    label_map, aux_map = load_labels()
    print(f"  标签总数: {len(label_map)}")

    # ── Step 1.5: 预训练 MAE Embedder ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 1.5: 预训练 Masked Autoencoder Embedder")

    from src.features import compute_features
    from src.embed import MaskedAutoencoderEmbedder

    all_inorms = []
    for coal_type in COAL_TYPES:
        train_data = load_coal_spectra(TRAIN_DIR, coal_type, label_map, aux_map)
        if train_data is None or train_data["n_batches"] == 0:
            continue
        inorm_mat = compute_features(train_data)
        all_inorms.append(inorm_mat)

    if all_inorms:
        X_pretrain = np.vstack(all_inorms)
        from sklearn.preprocessing import StandardScaler
        scaler_pretrain = StandardScaler()
        X_pretrain_scaled = scaler_pretrain.fit_transform(X_pretrain)

        input_dim = X_pretrain_scaled.shape[1]
        print(f"  预训练数据形状: {X_pretrain_scaled.shape}, 输入维度: {input_dim}")
        global_embedder = MaskedAutoencoderEmbedder(input_dim=input_dim, embed_dim=30)
        global_embedder.fit(X_pretrain_scaled, epochs=150)
    else:
        global_embedder = None

    # ── Step 2: 按煤种训练 ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: 两阶段训练（分煤种）")

    models      = {}
    cv_results  = {}
    n_batches_l = []

    for coal_type in COAL_TYPES:
        train_data = load_coal_spectra(TRAIN_DIR, coal_type, label_map, aux_map)
        if train_data is None or train_data["n_batches"] == 0:
            print(f"\n  [{coal_type}] 未找到训练数据，跳过")
            continue

        model_dict = train_coal_model(coal_type, train_data, embedder=global_embedder)
        models[coal_type]      = model_dict
        cv_results[coal_type]  = model_dict["cv_rmse"]
        n_batches_l.append(train_data["n_batches"])

    weights = np.array(n_batches_l, dtype=np.float32)
    values  = np.array(list(cv_results.values()), dtype=np.float32)
    global_cv_rmse = float(np.average(values, weights=weights))
    print(f"\n{'=' * 60}")
    print(f"全局 CV-RMSE: {global_cv_rmse:.2f}")

    print("\n" + "=" * 60)
    print("Step 3: 记录实验日志")
    log_experiment(global_cv_rmse, treatment=args.treatment)

    print("\n" + "=" * 60)
    print("Step 4: 测试集推理")

    all_preds = {}
    for coal_type in COAL_TYPES:
        if coal_type not in models:
            continue
        test_data = load_coal_spectra(TEST_DIR, coal_type, label_map=None, aux_map=None)
        if test_data is None or len(test_data["spectra"]) == 0:
            print(f"  [{coal_type}] 无测试数据")
            continue
        bp = predict_coal(coal_type, test_data, models[coal_type])
        all_preds.update(bp)

    print("\n  预测结果:")
    for name, pred in sorted(all_preds.items()):
        print(f"    {name}: {pred:.2f}")

    print("\n" + "=" * 60)
    print("Step 5: 生成提交文件")
    pack_submission(all_preds, cv_results, global_cv_rmse)
    print("\n完成！提交文件在 output/submit.zip")


if __name__ == "__main__":
    main()
