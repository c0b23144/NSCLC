import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, binary_erosion
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =========================================================
# 設定 (学習スクリプトと同じ)
# =========================================================
IMAGES_DIR = '/workspace/NSCLC_NIfTI2/imagesTr'
LABELS_DIR = '/workspace/NSCLC_NIfTI2/labelsTr/Neoplasm_Primary'
SAVE_ROOT = 'output/hu_density_analysis'

TEST_SIZE = 0.2
RANDOM_STATE = 42

# 学習スクリプトの群分け (クラスタとの対応を見るため)  下限 <= 面積 < 上限
GROUPS = [
    ('35-919px',     0,    919),
    ('919-1804px',   919,  1804),
    ('1804-10652px', 1804, float('inf')),
]

# Testの使い方
#   'all': Testの腫瘍スライスを全部使う (Trainと同じ条件で比較したいとき)
#   'max': 患者ごとの最大面積スライス1枚だけ (学習スクリプトのtestと同じ)
TEST_MODE = 'max'

# 密度を計算するHU範囲とビン幅 (学習時のクリップ範囲 -1000〜400 に合わせている)
HU_MIN, HU_MAX, BIN = -1000, 400, 10
SMOOTH_SIGMA = 1.0          # ヒストグラムの平滑化 (ビン単位)
MIN_PIX = 50                # これ未満の画素数のスライスは除外
ERODE_ITER = 0              # マスク縁をN画素削る(部分容積効果の除去)。0で無効

# クラスタリング (Trainだけで作る。Testは作ったクラスタに割り当てるだけ)
K_MAX = 8
K_FIXED = None              # 数を固定したいときは整数 (Noneならsilhouetteで自動)
N_EXAMPLES = 8              # クラスタごとの代表スライス枚数
OOD_PERCENTILE = 95         # Trainの重心距離のこのパーセンタイルを超えたTestスライスは「未知の型」候補

EDGES = np.arange(HU_MIN, HU_MAX + BIN, BIN)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2


# =========================================================
# 前処理 (学習スクリプトと同じ)
# =========================================================
def fix_slice_axis(data):
    """スライス軸(一番短い軸)を最後に持ってきて (H, W, Slices) にする"""
    min_axis = np.argmin(data.shape)
    if min_axis == 0:
        return data.transpose(1, 2, 0)
    elif min_axis == 1:
        return data.transpose(0, 2, 1)
    return data


def bbox_crop(img, mask, pad=8):
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask.shape[1])
    return img[y0:y1, x0:x1].astype(np.float16), mask[y0:y1, x0:x1]


def slice_density(hu):
    """腫瘍内HU -> 平滑化したヒストグラム密度 (積分=1)"""
    hu = np.clip(hu, HU_MIN, HU_MAX - 1e-6)
    h, _ = np.histogram(hu, bins=EDGES)
    h = gaussian_filter1d(h.astype(float), SMOOTH_SIGMA)
    return h / (h.sum() * BIN)


def summarize(hu):
    return dict(
        n_pix=hu.size, mean=hu.mean(), std=hu.std(), median=np.median(hu),
        p05=np.percentile(hu, 5), p95=np.percentile(hu, 95),
        skew=skew(hu), kurt=kurtosis(hu),
        frac_below_m500=(hu < -500).mean(),
        frac_m100_100=((hu >= -100) & (hu <= 100)).mean(),
        frac_above_100=(hu > 100).mean(),
    )


def group_name(v):
    for name, lo, hi in GROUPS:
        if lo <= v < hi:
            return name
    return 'other'


def extract(pairs, mode, split_name):
    """pairsから (密度配列, DataFrame, 切り出し画像リスト) を作る。
    mode='all': 腫瘍スライス全部 / mode='max': 最大面積スライス1枚"""
    dens, rows, crops = [], [], []
    for pair in pairs:
        img_vol = fix_slice_axis(nib.load(pair['image']).get_fdata(dtype=np.float32))
        mask_vol = fix_slice_axis(nib.load(pair['label']).get_fdata(dtype=np.float32)) > 0

        n_slices_img, n_slices_mask = img_vol.shape[2], mask_vol.shape[2]
        slice_areas = mask_vol.sum(axis=(0, 1))
        tumor_indices = np.where(slice_areas > 0)[0]
        if len(tumor_indices) == 0:
            print(f"[警告] 腫瘍スライスなし: {pair['name']} (スキップ)")
            continue
        patient_max = int(slice_areas.max())
        z_ratio = n_slices_img / n_slices_mask  # 学習スクリプトと同じ対応づけ

        use_indices = [int(np.argmax(slice_areas))] if mode == 'max' else [int(i) for i in tumor_indices]
        for mask_idx in use_indices:
            img_idx = min(int(mask_idx * z_ratio), n_slices_img - 1)
            img_slice = img_vol[:, :, img_idx]
            m = mask_vol[:, :, mask_idx]
            if ERODE_ITER > 0:
                m = binary_erosion(m, iterations=ERODE_ITER)
            if m.sum() < MIN_PIX:
                continue
            hu = img_slice[m]
            dens.append(slice_density(hu))
            rows.append(dict(
                split=split_name, case=pair['name'], slice=mask_idx,
                tumor_size=int(slice_areas[mask_idx]), patient_max_size=patient_max,
                **summarize(hu)))
            crops.append(bbox_crop(img_slice, m))
        print(f"[{split_name}] done: {pair['name']}")

    df = pd.DataFrame(rows)
    df['size_group'] = df['patient_max_size'].map(group_name)
    return np.array(dens), df, crops


# =========================================================
# 可視化
# =========================================================
def save_examples(df, crops, dist, k, path):
    """クラスタ中心に近いスライスのCT画像(腫瘍輪郭つき)を保存"""
    fig, axes = plt.subplots(k, N_EXAMPLES, figsize=(2.2 * N_EXAMPLES, 2.4 * k))
    axes = np.array(axes).reshape(k, N_EXAMPLES)
    for ax in axes.ravel():
        ax.axis('off')
    for c in range(k):
        members = np.where(df['cluster'].values == c)[0]
        order = members[np.argsort(dist[members, c])][:N_EXAMPLES]
        for j, i in enumerate(order):
            ax = axes[c, j]
            img_c, m_c = crops[i]
            ax.imshow(np.rot90(img_c.astype(np.float32)), cmap='gray', vmin=HU_MIN, vmax=HU_MAX)
            ax.contour(np.rot90(m_c), levels=[0.5], colors='r', linewidths=0.6)
            ax.set_title(f"c{c} {df.iloc[i]['case'][:12]}\nz={df.iloc[i]['slice']} "
                         f"mean={df.iloc[i]['mean']:.0f}", fontsize=6)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# =========================================================
# main
# =========================================================
def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)

    # ---- 画像とラベルのペア -> Train / Test ----
    all_images = sorted(f for f in os.listdir(IMAGES_DIR) if f.endswith('.nii.gz'))
    valid_pairs = []
    for f in all_images:
        lbl_path = os.path.join(LABELS_DIR, f)
        if os.path.exists(lbl_path):
            valid_pairs.append({'image': os.path.join(IMAGES_DIR, f), 'label': lbl_path, 'name': f})
    train_pairs, test_pairs = train_test_split(valid_pairs, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"有効なペア数: {len(valid_pairs)} / Train: {len(train_pairs)} / Test: {len(test_pairs)}")

    # ---- スライスごとの密度計算 (リサイズ・正規化はせず元のHUのまま) ----
    dens_tr, df_tr, crops_tr = extract(train_pairs, 'all', 'train')
    dens_te, df_te, crops_te = extract(test_pairs, TEST_MODE, 'test')
    print(f"Train: {len(df_tr)}スライス / {df_tr['case'].nunique()}症例, "
          f"Test: {len(df_te)}スライス / {df_te['case'].nunique()}症例 (TEST_MODE={TEST_MODE})")
    np.save(os.path.join(SAVE_ROOT, 'densities_train.npy'), dens_tr)
    np.save(os.path.join(SAVE_ROOT, 'densities_test.npy'), dens_te)

    # ---- クラスタリング: PCA・KMeansはTrainだけで学習し、TestはそれにあてはめるだけにしてTestの情報を漏らさない ----
    pca = PCA(n_components=min(10, dens_tr.shape[1]), random_state=0)
    Z_tr = pca.fit_transform(np.sqrt(dens_tr))   # sqrt変換: 低密度の裾も効かせる
    Z_te = pca.transform(np.sqrt(dens_te))

    if K_FIXED is not None:
        best_k = K_FIXED
    else:
        scores = {}
        for k in range(2, K_MAX + 1):
            labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z_tr)
            scores[k] = silhouette_score(Z_tr, labels)
        best_k = max(scores, key=scores.get)
        print('silhouette:', {k: round(v, 3) for k, v in scores.items()}, '-> k =', best_k)

    km = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(Z_tr)
    df_tr['cluster'] = km.labels_
    df_te['cluster'] = km.predict(Z_te)

    dist_tr = km.transform(Z_tr)
    dist_te = km.transform(Z_te)
    # 割り当てられた重心までの距離が、Trainの大半より遠いTestスライス = 未知の分布型の候補
    thr = np.percentile(dist_tr[np.arange(len(df_tr)), df_tr['cluster'].values], OOD_PERCENTILE)
    df_tr['dist_to_center'] = dist_tr[np.arange(len(df_tr)), df_tr['cluster'].values]
    df_te['dist_to_center'] = dist_te[np.arange(len(df_te)), df_te['cluster'].values]
    df_te['unseen_pattern'] = df_te['dist_to_center'] > thr
    df_tr['unseen_pattern'] = df_tr['dist_to_center'] > thr
    print(f"未知パターン候補 (距離>{thr:.3f}): Train {df_tr['unseen_pattern'].mean():.1%}, "
          f"Test {df_te['unseen_pattern'].mean():.1%}")

    df_all = pd.concat([df_tr, df_te], ignore_index=True)
    df_all.to_csv(os.path.join(SAVE_ROOT, 'slice_stats.csv'), index=False)

    # ---- (a) クラスタごとの平均密度: Train(青, ±SD) とTest(赤破線)を重ねる ----
    fig, axes = plt.subplots(1, best_k, figsize=(4 * best_k, 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for c, ax in enumerate(axes):
        d_tr = dens_tr[df_tr['cluster'].values == c]
        d_te = dens_te[df_te['cluster'].values == c]
        mu, sd = d_tr.mean(0), d_tr.std(0)
        ax.plot(CENTERS, mu, label='Train mean')
        ax.fill_between(CENTERS, np.maximum(mu - sd, 0), mu + sd, alpha=0.3)
        if len(d_te) > 0:
            ax.plot(CENTERS, d_te.mean(0), 'r--', label='Test mean')
        ax.set_title(f"cluster {c} (train={len(d_tr)}, test={len(d_te)})", fontsize=9)
        ax.set_xlabel('HU')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('Probability density')
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, 'cluster_mean_density.png'), dpi=150)
    plt.close()

    # ---- (b) 個別スライスの重ね描き (Train=青, Test=橙) ----
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, best_k, figsize=(4 * best_k, 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for c, ax in enumerate(axes):
        for dens, df_, color in [(dens_tr, df_tr, 'tab:blue'), (dens_te, df_te, 'tab:orange')]:
            idx = np.where(df_['cluster'].values == c)[0]
            if len(idx) == 0:
                continue
            for i in rng.choice(idx, size=min(20, len(idx)), replace=False):
                ax.plot(CENTERS, dens[i], alpha=0.4, lw=0.8, color=color)
        ax.set_title(f'cluster {c}: random slices (blue=train, orange=test)', fontsize=8)
        ax.set_xlabel('HU')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('Probability density')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, 'cluster_individual_slices.png'), dpi=150)
    plt.close()

    # ---- (c) PCA散布図 (Train=丸, Test=×) ----
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(Z_tr[:, 0], Z_tr[:, 1], c=df_tr['cluster'], s=4, cmap='tab10', alpha=0.5)
    plt.scatter(Z_te[:, 0], Z_te[:, 1], c=df_te['cluster'], s=22, cmap='tab10',
                marker='x', linewidths=0.8, vmin=0, vmax=9)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('o: train, x: test')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, 'pca_scatter.png'), dpi=150)
    plt.close()

    # ---- (d) 代表スライスのCT画像 (Train / Test別) ----
    save_examples(df_tr, crops_tr, dist_tr, best_k, os.path.join(SAVE_ROOT, 'cluster_examples_train.png'))
    save_examples(df_te, crops_te, dist_te, best_k, os.path.join(SAVE_ROOT, 'cluster_examples_test.png'))

    # 未知パターン候補のTestスライス (重心から遠い順に最大16枚)
    odd = df_te[df_te['unseen_pattern']].sort_values('dist_to_center', ascending=False).head(16)
    if len(odd) > 0:
        n_col = 8
        n_row = int(np.ceil(len(odd) / n_col))
        fig, axes = plt.subplots(n_row, n_col, figsize=(2.2 * n_col, 2.6 * n_row))
        axes = np.array(axes).reshape(n_row, n_col)
        for ax in axes.ravel():
            ax.axis('off')
        for ax, (i, r) in zip(axes.ravel(), odd.iterrows()):
            ti = df_te.index.get_loc(i)
            img_c, m_c = crops_te[ti]
            ax.imshow(np.rot90(img_c.astype(np.float32)), cmap='gray', vmin=HU_MIN, vmax=HU_MAX)
            ax.contour(np.rot90(m_c), levels=[0.5], colors='r', linewidths=0.6)
            ax.set_title(f"{r['case'][:12]} z={r['slice']}\nc{r['cluster']} d={r['dist_to_center']:.2f}", fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_ROOT, 'test_unseen_pattern_examples.png'), dpi=150)
        plt.close()

    # ---- (e) クラスタ構成の比較 Train vs Test ----
    prop = pd.crosstab(df_all['split'], df_all['cluster'], normalize='index')
    prop.T.plot(kind='bar', figsize=(7, 4), color=['tab:blue', 'tab:orange'][:len(prop)])
    plt.ylabel('Proportion of slices')
    plt.xlabel('cluster')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, 'cluster_proportion_train_vs_test.png'), dpi=150)
    plt.close()

    # ---- (f) 要約CSV ----
    cols = ['n_pix', 'mean', 'std', 'median', 'p05', 'p95', 'skew',
            'frac_below_m500', 'frac_m100_100', 'frac_above_100']
    summ = df_all.groupby(['split', 'cluster'])[cols].mean().round(2)
    summ['n_slices'] = df_all.groupby(['split', 'cluster']).size()
    summ['n_cases'] = df_all.groupby(['split', 'cluster'])['case'].nunique()
    summ.to_csv(os.path.join(SAVE_ROOT, 'cluster_summary.csv'))
    print(summ)

    ct = pd.crosstab([df_all['split'], df_all['cluster']], df_all['size_group'])
    ct.to_csv(os.path.join(SAVE_ROOT, 'cluster_vs_size_group.csv'))
    print(ct)
    pd.crosstab(df_all['case'], df_all['cluster']).to_csv(os.path.join(SAVE_ROOT, 'cluster_vs_case.csv'))

    print(f"保存先: {SAVE_ROOT}")


if __name__ == '__main__':
    main()
