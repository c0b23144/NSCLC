import os
import csv
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 学習スクリプトと同じフォルダに置くこと
from Nsclc_train_ct import IMAGES_DIR, LABELS_DIR, fix_slice_axis

# =========================================================
# 設定
# =========================================================
OUTPUT_DIR = 'output/hu_map'

# True: 評価と同じテスト患者だけ / False: 全患者
TEST_ONLY = True

# HUの色の範囲 (この範囲を色で表示する)
VMIN, VMAX = -1000, 400
CMAP = 'turbo'

# 画像の向きが合わないときに調整 (反時計回りに90度回す回数 0~3)
ROTATE = 0

# 腫瘍の拡大図で、腫瘍の外側に足すピクセル数
CROP_MARGIN = 10


# =========================================================
# 読み込み: 最大面積スライス (評価で使うスライスと同じ)
# =========================================================
def load_max_slice(pair):
    img = fix_slice_axis(
        nib.load(pair['image']).get_fdata(dtype=np.float32)
    )
    mask = fix_slice_axis(
        nib.load(pair['label']).get_fdata(dtype=np.float32)
    ) > 0
    k = int(mask.sum(axis=(0, 1)).argmax())
    return img[:, :, k], mask[:, :, k], k


def orient(a):
    """表示用に転置 (+回転)"""
    return np.rot90(a.T, ROTATE)


def tumor_bbox(mask, margin):
    ys, xs = np.where(mask)
    y0 = max(ys.min() - margin, 0)
    y1 = min(ys.max() + margin + 1, mask.shape[0])
    x0 = max(xs.min() - margin, 0)
    x1 = min(xs.max() + margin + 1, mask.shape[1])
    return y0, y1, x0, x1


# =========================================================
# 描画
# =========================================================
def plot_patient(name, sl, mask, k, out_dir):
    sl_d, mask_d = orient(sl), orient(mask)
    y0, y1, x0, x1 = tumor_bbox(mask_d, CROP_MARGIN)
    tumor_hu = sl[mask]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    # (1) 全体: グレースケール + 腫瘍の輪郭
    axes[0].imshow(sl_d, cmap='gray', vmin=VMIN, vmax=VMAX)
    axes[0].contour(mask_d, levels=[0.5], colors='red', linewidths=1.2)
    axes[0].set_title(
        f'{name} (slice {k}) gray + tumor contour', fontsize=12
    )

    # (2) 全体: HU値を色で表示
    im = axes[1].imshow(sl_d, cmap=CMAP, vmin=VMIN, vmax=VMAX)
    axes[1].contour(mask_d, levels=[0.5], colors='white', linewidths=1.0)
    axes[1].set_title('HU color map (whole slice)', fontsize=12)

    # (3) 腫瘍の拡大図
    lo, hi = np.percentile(tumor_hu, [1, 99])
    crop = sl_d[y0:y1, x0:x1]
    im2 = axes[2].imshow(
        crop, cmap=CMAP, vmin=lo, vmax=hi, interpolation='nearest'
    )
    axes[2].contour(
        mask_d[y0:y1, x0:x1],
        levels=[0.5],
        colors='white',
        linewidths=1.5
    )
    axes[2].set_title(
        f'Tumor zoom (color: {lo:.0f} to {hi:.0f} HU, pixel level)',
        fontsize=12
    )

    for ax in axes:
        ax.axis('off')

    fig.colorbar(im, ax=axes[:2], fraction=0.025, pad=0.02, label='HU')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02, label='HU')
    fig.savefig(
        os.path.join(out_dir, f'{name}.png'),
        dpi=150,
        bbox_inches='tight'
    )
    plt.close(fig)


# =========================================================
# CSV保存: 腫瘍内部 + 腫瘍以外のHU統計
# =========================================================
def save_csv(rows, out_dir):
    csv_path = os.path.join(out_dir, 'hu_map_summary.csv')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            'patient',
            'slice',
            'tumor_pixels',
            'tumor_hu_min',
            'tumor_hu_mean',
            'tumor_hu_max',
            'non_tumor_pixels',
            'non_tumor_hu_min',
            'non_tumor_hu_mean',
            'non_tumor_hu_max'
        ])
        w.writerows(rows)


# =========================================================
# main
# =========================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pairs = []
    for f in sorted(
        f for f in os.listdir(IMAGES_DIR) if f.endswith('.nii.gz')
    ):
        lbl = os.path.join(LABELS_DIR, f)
        if os.path.exists(lbl):
            pairs.append({
                'image': os.path.join(IMAGES_DIR, f),
                'label': lbl,
                'name': f
            })

    if TEST_ONLY:
        _, pairs = train_test_split(
            pairs, test_size=0.2, random_state=42
        )

    print(f'対象患者数: {len(pairs)}')

    rows = []
    for i, p in enumerate(pairs, 1):
        name = p['name'].replace('.nii.gz', '')
        sl, mask, k = load_max_slice(p)

        if i == 1:
            print(f'画像の値の範囲: {sl.min():.1f} ~ {sl.max():.1f}')
            if np.abs(sl).max() <= 5:
                print(
                    '[注意] 値が小さすぎます。HUではなく'
                    '正規化済みの画像の可能性があります。'
                )

        plot_patient(name, sl, mask, k, OUTPUT_DIR)

        # 腫瘍内部のHU値
        tumor_hu = sl[mask]

        # 腫瘍以外のHU値
        # 注意: 現状はスライス内のマスク外すべて
        # (肺野以外の血管・骨・胸壁・空気なども含む)
        non_tumor_hu = sl[~mask]

        rows.append([
            name,
            k,
            int(mask.sum()),
            f'{tumor_hu.min():.0f}',
            f'{tumor_hu.mean():.1f}',
            f'{tumor_hu.max():.0f}',
            int((~mask).sum()),
            f'{non_tumor_hu.min():.0f}',
            f'{non_tumor_hu.mean():.1f}',
            f'{non_tumor_hu.max():.0f}'
        ])

        print(f'\r  {i}/{len(pairs)}', end='')

    print()
    save_csv(rows, OUTPUT_DIR)
    print(f'保存しました: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
