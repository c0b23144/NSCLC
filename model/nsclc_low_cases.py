import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Nsclc_train_ct import fix_slice_axis, IMAGES_DIR, LABELS_DIR

# ================= 設定 =================
CSV_PATH  = 'output/ct_metrics_patient/per_patient_metrics.csv'
OUT_DIR   = 'output/low_cases'
METRIC    = 'IoU'     # 低さの判定に使う指標
THRESH    = 0.1        # これ未満を「低い」とする
MAX_CASES = 15         # 1モデルあたり最大何人まで描くか
CT_MODES  = ['lung', 'mediastinal', 'raw']

# 表示用ウィンドウ (学習時のCT_SETTINGSに合わせて書き換える)
WINDOWS = {
    'lung':        (-1350, 150),
    'mediastinal': (-160, 240),
    'raw':         (-1000, 500),
}
HIST_RANGE = (-1024, 600)   # ヒストグラムの横軸 (HU)
# =======================================


def load_case(name):
    """最大腫瘍スライスのCTとマスクを返す"""
    img = fix_slice_axis(nib.load(os.path.join(IMAGES_DIR, name + '.nii.gz')).get_fdata(dtype=np.float32))
    lbl = fix_slice_axis(nib.load(os.path.join(LABELS_DIR, name + '.nii.gz')).get_fdata(dtype=np.float32)) > 0
    z = int(np.argmax(lbl.sum(axis=(0, 1))))
    return img[:, :, z], lbl[:, :, z], z


def show(ax, ct, win):
    ax.imshow(ct.T, cmap='gray', vmin=win[0], vmax=win[1])  # 向きが変なら .T を np.rot90(ct) に
    ax.axis('off')


def plot_model(df, mode):
    col = f'{mode}_{METRIC}'
    low = df[df[col] < THRESH].sort_values(col).head(MAX_CASES)
    n_all_low = int((df[col] < THRESH).sum())
    print(f"\n[{mode}] {METRIC}<{THRESH}: {n_all_low}人 / {len(df)}人 (うち上位{len(low)}人を描画)")
    print(low[['patient', 'tumor_area', col]].to_string(index=False))
    if low.empty:
        return

    win = WINDOWS[mode]
    fig, axes = plt.subplots(len(low), 3, figsize=(14, 3.6 * len(low)), squeeze=False)
    for i, (_, r) in enumerate(low.iterrows()):
        ct, mask, z = load_case(r['patient'])

        show(axes[i, 0], ct, win)
        axes[i, 0].set_title(f"{r['patient']}  {METRIC}={r[col]:.3f}\narea={r['tumor_area']}  slice={z}", fontsize=9)

        show(axes[i, 1], ct, win)
        axes[i, 1].contour(mask.T, levels=[0.5], colors='red', linewidths=1)
        axes[i, 1].set_title('GT (red)', fontsize=9)

        ax = axes[i, 2]
        ax.hist(ct.ravel(), bins=100, range=HIST_RANGE, density=True, alpha=0.4, color='gray', label='whole slice')
        ax.hist(ct[mask], bins=100, range=HIST_RANGE, density=True, alpha=0.7, color='red', label='tumor')
        if mode != 'raw':
            ax.axvline(win[0], color='blue', ls='--', lw=1)
            ax.axvline(win[1], color='blue', ls='--', lw=1, label='window')
        ax.set_xlabel('HU'); ax.set_ylabel('density'); ax.legend(fontsize=7)
        ax.set_title(f"tumor HU mean={ct[mask].mean():.0f}, sd={ct[mask].std():.0f}", fontsize=9)

    plt.suptitle(f'{DISPLAY[mode]}: low {METRIC} cases', fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'low_cases_{mode}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary(df, mode):
    """低い人たち全員 vs それ以外 で、腫瘍内HU分布をまとめて比較"""
    col = f'{mode}_{METRIC}'
    groups = {'low': df[df[col] < THRESH], 'others': df[df[col] >= THRESH]}
    plt.figure(figsize=(7, 4.5))
    for label, color in [('low', 'red'), ('others', 'blue')]:
        vals = []
        for name in groups[label]['patient']:
            ct, mask, _ = load_case(name)
            vals.append(ct[mask])
        if vals:
            plt.hist(np.concatenate(vals), bins=100, range=HIST_RANGE, density=True,
                     alpha=0.5, color=color, label=f'{label} (n={len(vals)})')
    plt.xlabel('Tumor HU'); plt.ylabel('density')
    plt.title(f'{DISPLAY[mode]}: tumor HU, low vs others')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'summary_hist_{mode}.png'), dpi=150)
    plt.close()


DISPLAY = {'lung': 'Lung window', 'mediastinal': 'Mediastinal window', 'raw': 'Raw HU'}

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    for mode in CT_MODES:
        if f'{mode}_{METRIC}' not in df.columns:
            continue
        plot_model(df, mode)
        plot_summary(df, mode)
    print(f"\n保存先: {OUT_DIR}")