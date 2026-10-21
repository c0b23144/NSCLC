import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

IMAGES_DIR = '/workspace/NSCLC_NIfTI2/imagesTr'
LABELS_DIR = '/workspace/NSCLC_NIfTI2/labelsTr/Neoplasm_Primary'

case_ids = ["LUNG018", "LUNG056", "LUNG230"]

TRAIN_RANGE = (-1000, 400)   # 学習コードのクリップ範囲
VIEW_RANGE = (-600, 240)     # 先ほどの範囲


def fix_slice_axis(data):
    min_axis = np.argmin(data.shape)
    if min_axis == 0:
        return data.transpose(1, 2, 0)
    elif min_axis == 1:
        return data.transpose(0, 2, 1)
    return data


def find_file(directory, cid):
    files = sorted(glob.glob(os.path.join(directory, "*.nii.gz")))
    num = cid.replace("LUNG", "")
    hits = [f for f in files if cid in os.path.basename(f)] or \
           [f for f in files if num in os.path.basename(f)]
    if not hits:
        raise FileNotFoundError(f"{cid} が {directory} に見つかりません")
    return hits[0]


def show(ax, slice2d, lo, hi, title):
    clipped = np.clip(slice2d, lo, hi)
    im = ax.imshow(clipped.T, cmap="gray", origin="lower", vmin=lo, vmax=hi)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)


fig, axes = plt.subplots(4, len(case_ids), figsize=(5 * len(case_ids), 19))

for col, cid in enumerate(case_ids):
    img_path = find_file(IMAGES_DIR, cid)
    lbl_path = os.path.join(LABELS_DIR, os.path.basename(img_path))

    img_vol = fix_slice_axis(nib.load(img_path).get_fdata(dtype=np.float32))
    mask_vol = fix_slice_axis(nib.load(lbl_path).get_fdata(dtype=np.float32)) > 0

    slice_areas = mask_vol.sum(axis=(0, 1))
    mask_idx = int(np.argmax(slice_areas))
    z_ratio = img_vol.shape[2] / mask_vol.shape[2]
    img_idx = min(int(mask_idx * z_ratio), img_vol.shape[2] - 1)

    raw = img_vol[:, :, img_idx]

    # 1段目: 元のCT値(全範囲)
    show(axes[0, col], raw, raw.min(), raw.max(),
         f"{cid} slice {img_idx}\noriginal [{raw.min():.0f}, {raw.max():.0f}]")

    # 2段目: 学習時の範囲
    show(axes[1, col], raw, *TRAIN_RANGE,
         f"train range {TRAIN_RANGE}")

    # 3段目: 先ほどの範囲
    show(axes[2, col], raw, *VIEW_RANGE,
         f"view range {VIEW_RANGE}")

    # 4段目: ヒストグラム + 両方の範囲
    ax = axes[3, col]
    ax.hist(raw.ravel(), bins=200, color="gray", edgecolor="none")
    ax.set_yscale("log")
    ax.axvline(TRAIN_RANGE[0], color="blue", linestyle=":", label=f"train {TRAIN_RANGE}")
    ax.axvline(TRAIN_RANGE[1], color="blue", linestyle=":")
    ax.axvline(VIEW_RANGE[0], color="red", linestyle="--", label=f"view {VIEW_RANGE}")
    ax.axvline(VIEW_RANGE[1], color="red", linestyle="--")
    ax.set_xlabel("CT value (HU)")
    ax.set_ylabel("pixels (log)")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("lung_range_compare.png", dpi=150)
plt.show()