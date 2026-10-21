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

LOG_Y = False          # Trueにすると縦軸が対数
SHOW_LABELS_MAX = 150  # 棒がこれ以下なら横軸に画像番号を全部表示。多いと読めないので順位だけ表示


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


def tumor_slice_areas(label_path):
    """腫瘍が写っているスライスの (スライス番号, 腫瘍ピクセル数) のリスト"""
    mask = fix_slice_axis(nib.load(label_path).get_fdata(dtype=np.float32)) > 0
    areas = mask.sum(axis=(0, 1))
    return [(int(z), int(a)) for z, a in enumerate(areas) if a > 0]


def plot_sorted(labels, areas, path, title, xlabel, ylabel):
    """areasの昇順に並べて棒グラフにする。並べ替えたlabels, areasを返す"""
    order = np.argsort(areas, kind='stable')
    labels = [labels[i] for i in order]
    areas = np.array(areas)[order]
    n = len(labels)

    show_labels = n <= SHOW_LABELS_MAX
    fig, ax = plt.subplots(figsize=(max(14, n * 0.12) if show_labels else 14, 7))
    ax.bar(range(n), areas, color='#2196F3', width=1.0 if not show_labels else 0.85,
           linewidth=0 if not show_labels else None)

    if show_labels:
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
    else:
        step = max(1, n // 10)
        ticks = list(range(0, n, step)) + [n - 1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t + 1) for t in ticks], fontsize=10)
        xlabel += '  (IDs are in the CSV)'

    ax.set_xlim(-0.8, n - 0.2)
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    if LOG_Y:
        ax.set_yscale('log')
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return labels, areas


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    patient_ids, patient_areas = [], []   # 患者ごと(最大スライス)
    slice_ids, slice_areas = [], []       # 腫瘍が写っている全スライス: (画像番号, スライス番号)
    for f in sorted(f for f in os.listdir(IMAGES_DIR) if f.endswith('.nii.gz')):
        lbl = os.path.join(LABELS_DIR, f)
        if not os.path.exists(lbl):
            continue
        slices = tumor_slice_areas(lbl)
        if not slices:
            print(f"[警告] 腫瘍なし: {f} (スキップ)")
            continue
        pid = f.replace('.nii.gz', '')
        patient_ids.append(pid)
        patient_areas.append(max(a for _, a in slices))
        for z, a in slices:
            slice_ids.append((pid, z))
            slice_areas.append(a)

    print(f"患者数: {len(patient_ids)} / 腫瘍スライス数: {len(slice_ids)}")
    print(f"スライス面積: {min(slice_areas)} - {max(slice_areas)} px")

    # ---- 全患者・全スライス ----
    labels = [f"{pid}_slice{z}" for pid, z in slice_ids]
    order = np.argsort(slice_areas, kind='stable')
    plot_sorted(labels, slice_areas, os.path.join(OUT_DIR, 'area_sorted_slices.png'),
                f'NSCLC-Radiomics: Tumor Area per Slice (all slices, n={len(labels)})',
                'Slice (sorted by tumor area)', 'Tumor Area on Slice (pixels)')
    with open(os.path.join(OUT_DIR, 'area_sorted_slices.csv'), 'w', newline='', encoding='utf-8') as fcsv:
        w = csv.writer(fcsv)
        w.writerow(['rank', 'image_id', 'slice_idx', 'area_px'])
        for rank, i in enumerate(order, start=1):
            w.writerow([rank, slice_ids[i][0], slice_ids[i][1], slice_areas[i]])

    # ---- 参考: 患者ごと(最大スライス) ----
    sorted_ids, sorted_areas = plot_sorted(
        patient_ids, patient_areas, os.path.join(OUT_DIR, 'area_sorted.png'),
        f'NSCLC-Radiomics: Tumor Area per Patient (max slice, n={len(patient_ids)})',
        'Image ID (sorted by tumor area)', 'Tumor Area on Max Slice (pixels)')
    with open(os.path.join(OUT_DIR, 'area_sorted.csv'), 'w', newline='', encoding='utf-8') as fcsv:
        w = csv.writer(fcsv)
        w.writerow(['rank', 'image_id', 'area_px'])
        for rank, (n, a) in enumerate(zip(sorted_ids, sorted_areas), start=1):
            w.writerow([rank, n, int(a)])

    print(f"保存しました: {OUT_DIR}")


if __name__ == '__main__':
    main()
