import nibabel as nib
import matplotlib.pyplot as plt

# CT
ct_path = r"E:\NSCLC_NIfTI\imagesTr\LUNG1-001.nii.gz"

# mask
mask_path = r"E:\NSCLC_NIfTI\labelsTr\LUNG1-001.nii.gz"

# 読み込み
ct_img = nib.load(ct_path)
mask_img = nib.load(mask_path)

ct = ct_img.get_fdata()
mask = mask_img.get_fdata()

print("CT shape:", ct.shape)
print("MASK shape:", mask.shape)

# 真ん中slice
slice_idx = ct.shape[2] // 2
plt.figure(figsize=(8,8))

# CT

plt.imshow(ct[:, :, slice_idx], cmap="gray")

# mask overlay
plt.imshow(
    mask[:, :, slice_idx],
    cmap="Reds",
    alpha=0.5
)

plt.title("CT + Tumor Mask")
plt.axis("off")

plt.show()

# 横並び表示
fig, axes = plt.subplots(1, 2, figsize=(12,6))

# 左: CTだけ
axes[0].imshow(
    ct[:, :, slice_idx],
    cmap="gray",
    vmin=-1000,
    vmax=400
)

axes[0].set_title("CT")

axes[0].axis("off")

# 右: overlay
axes[1].imshow(
    ct[:, :, slice_idx],
    cmap="gray",
    vmin=-1000,
    vmax=400
)

axes[1].imshow(
    mask[:, :, slice_idx],
    cmap="Reds",
    alpha=0.5
)

axes[1].set_title("CT + Tumor Mask")

axes[1].axis("off")

plt.tight_layout()

plt.show()