import os
import csv
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =========================================================
# 設定
# =========================================================
IMAGES_DIR = '/workspace/NSCLC_NIfTI2/imagesTr'
LABELS_DIR = '/workspace/NSCLC_NIfTI2/labelsTr/Neoplasm_Primary'
OUT_DIR = 'output/area_sorted'

LOG_Y = False              # Trueにすると縦軸が対数


# =========================================================
# 処理
# =========================================================
def fix_slice_axis(data):
    """スライス軸(一番短い軸)を最後にして (H, W, Slices) にする"""
    min_axis = np.argmin(data.shape)
    if min_axis == 0:
        return data.transpose(1, 2, 0)
    elif min_axis == 1:
        return data.transpose(0, 2, 1)
    return data


def max_slice_area(label_path):
    """腫瘍面積が最大のスライスの腫瘍ピクセル数"""
    mask = fix_slice_axis(nib.load(label_path).get_fdata(dtype=np.float32)) > 0
    return int(mask.sum(axis=(0, 1)).max())


def plot_sorted_areas(names, areas, path):
    order = np.argsort(areas)  # 昇順
    names = [names[i] for i in order]
    areas = np.array(areas)[order]

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.12), 7))
    ax.bar(range(len(names)), areas, color='#2196F3', width=0.85)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_xlim(-0.8, len(names) - 0.2)
    ax.set_xlabel('Image ID (sorted by tumor area)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tumor Area on Max Slice (pixels)', fontsize=12, fontweight='bold')
    ax.set_title(f'NSCLC-Radiomics: Tumor Area per Patient (n={len(names)})',
                 fontsize=14, fontweight='bold')
    if LOG_Y:
        ax.set_yscale('log')
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return names, areas


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    names, areas = [], []
    for f in sorted(f for f in os.listdir(IMAGES_DIR) if f.endswith('.nii.gz')):
        lbl = os.path.join(LABELS_DIR, f)
        if not os.path.exists(lbl):
            continue
        area = max_slice_area(lbl)
        if area == 0:
            print(f"[警告] 腫瘍なし: {f} (スキップ)")
            continue
        names.append(f.replace('.nii.gz', ''))
        areas.append(area)

    print(f"患者数: {len(names)}  面積: {min(areas)} - {max(areas)} px")

    sorted_names, sorted_areas = plot_sorted_areas(names, areas, os.path.join(OUT_DIR, 'area_sorted.png'))

    with open(os.path.join(OUT_DIR, 'area_sorted.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'image_id', 'area_px'])
        for i, (n, a) in enumerate(zip(sorted_names, sorted_areas), start=1):
            w.writerow([i, n, int(a)])

    print(f"保存しました: {OUT_DIR}")


if __name__ == '__main__':
    main()
