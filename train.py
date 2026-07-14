"""AE 维度扫描 — python train.py --embed-dim N"""

import sys, os, argparse, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import COAL_TYPES, TRAIN_DIR, TEST_DIR
from src.data   import load_labels, load_coal_spectra
from src.model  import train_coal_model, predict_coal
from src.submit import pack_submission
from src.experiment_tracker import log_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--treatment", type=str, default="")
    parser.add_argument("--embed-dim", type=int, default=30)
    args = parser.parse_args()
    embed_dim = args.embed_dim

    print("=" * 60)
    print("Step 1: 加载标签")
    label_map, aux_map = load_labels()
    print(f"  标签总数: {len(label_map)}")

    print("\n" + "=" * 60)
    print(f"Step 1.5: 预训练 AE (embed_dim={embed_dim})")
    from src.features import compute_features
    from src.embed import AutoencoderEmbedder

    all_inorms = []
    for ct in COAL_TYPES:
        td = load_coal_spectra(TRAIN_DIR, ct, label_map, aux_map)
        if td is None or td["n_batches"] == 0: continue
        all_inorms.append(compute_features(td))

    if all_inorms:
        Xp = np.vstack(all_inorms)
        from sklearn.preprocessing import StandardScaler
        Xp = StandardScaler().fit_transform(Xp)
        print(f"  预训练数据: {Xp.shape}, embed_dim={embed_dim}")
        global_embedder = AutoencoderEmbedder(input_dim=Xp.shape[1], embed_dim=embed_dim)
        global_embedder.fit(Xp, epochs=100)
    else:
        global_embedder = None

    print("\n" + "=" * 60)
    print("Step 2: 两阶段训练")
    models, cv_results, nbl = {}, {}, []
    for ct in COAL_TYPES:
        td = load_coal_spectra(TRAIN_DIR, ct, label_map, aux_map)
        if td is None or td["n_batches"] == 0: continue
        md = train_coal_model(ct, td, embedder=global_embedder)
        models[ct] = md; cv_results[ct] = md["cv_rmse"]; nbl.append(td["n_batches"])

    w = np.array(nbl, dtype=np.float32)
    v = np.array(list(cv_results.values()), dtype=np.float32)
    gcv = float(np.average(v, weights=w))
    treatment = args.treatment or f"AE嵌入 dim={embed_dim}"
    print(f"\n全局 CV-RMSE: {gcv:.2f}")

    print("\n" + "=" * 60)
    print("Step 3: 记录日志")
    log_experiment(gcv, treatment=treatment)

    print("\n" + "=" * 60)
    print("Step 4: 测试集推理")
    all_preds = {}
    for ct in COAL_TYPES:
        if ct not in models: continue
        td = load_coal_spectra(TEST_DIR, ct, label_map=None, aux_map=None)
        if td is None or len(td["spectra"]) == 0: continue
        all_preds.update(predict_coal(ct, td, models[ct]))
    for n, p in sorted(all_preds.items()): print(f"    {n}: {p:.2f}")

    print("\n" + "=" * 60)
    print("Step 5: 打包")
    pack_submission(all_preds, cv_results, gcv)
    print(f"\n完成! (dim={embed_dim})")

if __name__ == "__main__":
    main()
