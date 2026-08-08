import os
import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from monai.networks.nets import SegResNet
from monai.losses import DiceCELoss
from sklearn.model_selection import train_test_split

# 保存先
os.makedirs(
    "checkpoints_segres",
    exist_ok=True
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Dataset (3D)
class NSCLCNiftiDataset(Dataset):

    def __init__(self, pairs):

        self.pairs = pairs

    def __len__(self):

        return len(self.pairs)

    def fix_axis(self, data):

        shape = data.shape
        min_axis = np.argmin(shape)

        if min_axis == 0:
            data = data.transpose(1,2,0)

        elif min_axis == 1:
            data = data.transpose(0,2,1)

        return data

    def __getitem__(self, idx):

        pair = self.pairs[idx]

        img = nib.load(
            pair["image"]
        ).get_fdata()

        mask = nib.load(
            pair["label"]
        ).get_fdata()

        img = self.fix_axis(img)
        mask = self.fix_axis(mask)

        # CT normalization
        img = np.clip(
            img,
            -1000,
            400
        )

        img = (img + 1000) / 1400

        # numpy → tensor
        img = torch.from_numpy(
            img
        ).float()

        mask = torch.from_numpy(
            mask
        ).float()

        tumor_size = torch.sum(
            mask > 0
        ).item()

        # [H,W,D]
        # ↓
        # [1,H,W,D]
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)

        # 3D resize
        img = F.interpolate(
            img.unsqueeze(0),
            size=(96,96,96),
            mode="trilinear",
            align_corners=False
        ).squeeze(0)

        mask = F.interpolate(
            mask.unsqueeze(0),
            size=(96,96,96),
            mode="nearest"
        ).squeeze(0)

        return {
            "image": img,
            "label": mask,
            "tumor_size": tumor_size
        }

images_dir = "NSCLC_NIfTI2/imagesTr"
labels_dir = "NSCLC_NIfTI2/labelsTr/Neoplasm_Primary"

pairs=[]

for f in sorted(os.listdir(images_dir)):
    if f.endswith(".nii.gz"):

        img_path = os.path.join(
            images_dir,f
        )

        label_path = os.path.join(
            labels_dir,f
        )

        if os.path.exists(label_path):
            pairs.append(
                {
                    "image":img_path,
                    "label":label_path
                }
            )

train_pairs, test_pairs = train_test_split(
    pairs,
    test_size=0.2,
    random_state=42
)

train_dataset = NSCLCNiftiDataset(
    train_pairs
)

test_dataset = NSCLCNiftiDataset(
    test_pairs
)

# 腫瘍サイズ分割
train_sizes = []

for i in range(len(train_dataset)):
    item = train_dataset[i]
    train_sizes.append(
        item["tumor_size"]
    )

# 腫瘍サイズを保存したリスト
test_sizes = []

for i in range(len(test_dataset)):
    item = test_dataset[i]
    test_sizes.append(
        item["tumor_size"]
    )

threshold = np.median(train_sizes)

# ヒストグラム
plt.figure(figsize=(10,5))


plt.hist(
    train_sizes,
    bins=30,
    alpha=0.6,
    label=f"Train (n={len(train_sizes)})",
    edgecolor="black"
)

plt.hist(
    test_sizes,
    bins=30,
    alpha=0.6,
    label=f"Test (n={len(test_sizes)})",
    edgecolor="black"
)

plt.axvline(
    threshold,
    linestyle="dashed",
    linewidth=2,
    label=f"Threshold {int(threshold)} voxel"
)

plt.xlabel("Tumor Size (voxel)")
plt.ylabel("Number of cases")
plt.title("3D Tumor Size Distribution")
plt.legend()
plt.grid(axis="y")

plt.savefig(
    "checkpoints_segres/tumor_hist.png"
)

plt.close()

print("threshold:", threshold)

train_large_pairs = []
train_small_pairs = []

for i, pair in enumerate(train_pairs):

    if train_sizes[i] >= threshold:
        train_large_pairs.append(pair)

    else:
        train_small_pairs.append(pair)

test_large_pairs = []
test_small_pairs = []

for i, pair in enumerate(test_pairs):

    if test_sizes[i] >= threshold:
        test_large_pairs.append(pair)

    else:
        test_small_pairs.append(pair)

print("train large:", len(train_large_pairs))
print("train small:", len(train_small_pairs))
print("test large:", len(test_large_pairs))
print("test small:", len(test_small_pairs))

train_dataset_large = NSCLCNiftiDataset(
    train_large_pairs
)

train_dataset_small = NSCLCNiftiDataset(
    train_small_pairs
)

test_dataset_large = NSCLCNiftiDataset(
    test_large_pairs
)

test_dataset_small = NSCLCNiftiDataset(
    test_small_pairs
)

train_loader_large = DataLoader(
    train_dataset_large,
    batch_size=4,
    shuffle=True
)

train_loader_small = DataLoader(
    train_dataset_small,
    batch_size=4,
    shuffle=True
)

test_loader_large = DataLoader(
    test_dataset_large,
    batch_size=4
)

test_loader_small = DataLoader(
    test_dataset_small,
    batch_size=4
)

# 評価関数
def evaluate(model, loader):

    model.eval()

    dice_list=[]
    iou_list=[]
    acc_list=[]
    rec_list=[]
    pre_list=[]

    with torch.no_grad():

        for batch in loader:

            images=batch["image"].to(device)
            masks=batch["label"].to(device)

            outputs=model(images)

            # 2class
            pred=torch.argmax(
                outputs,
                dim=1
            )

            pred=pred.float()
            masks=masks.squeeze(1)


            TP=((pred==1)&(masks==1)).sum().item()
            TN=((pred==0)&(masks==0)).sum().item()
            FP=((pred==1)&(masks==0)).sum().item()
            FN=((pred==0)&(masks==1)).sum().item()


            dice=(2*TP)/(2*TP+FP+FN+1e-7)

            iou=TP/(TP+FP+FN+1e-7)

            acc=(TP+TN)/(TP+TN+FP+FN+1e-7)

            recall=TP/(TP+FN+1e-7)

            precision=TP/(TP+FP+1e-7)

            dice_list.append(dice)
            iou_list.append(iou)
            acc_list.append(acc)
            rec_list.append(recall)
            pre_list.append(precision)

    print("DICE:", np.mean(dice_list))

    print("IoU:", np.mean(iou_list))

    print("ACC:", np.mean(acc_list))

    print("REC:", np.mean(rec_list))

    print("PRE:", np.mean(pre_list))

# SegResNet
model = SegResNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    init_filters=32,
    blocks_down=(1,2,2,4),
    blocks_up=(1,1,1),
    dropout_prob=0.2
)

model.load_state_dict(
    torch.load(
        "checkpoints_segres/large_epoch_100.pth",
        map_location=device
    )
)

model=model.to(device)

# 評価実行
evaluate(model, test_loader_large)

plt.figure(figsize=(10,5))

counts, bins, patches = plt.hist(
    test_sizes,
    bins=30,
    color='orange',
    alpha=0.6,
    edgecolor='black',
    label=f'Test Data (n={len(test_sizes)})'
)

centers = (bins[:-1] + bins[1:]) / 2

errors = np.sqrt(counts)

plt.errorbar(
    centers,
    counts,
    yerr=errors,
    fmt='k.',
    capsize=4
)

plt.axvline(
    threshold,
    color='red',
    linestyle='dashed',
    linewidth=2,
    label=f'Threshold ({int(threshold)})'
)

plt.xlabel("Tumor Volume (Voxel)")
plt.ylabel("Number of Patients")
plt.title("Test Tumor Volume Distribution")
plt.legend()

plt.savefig("test_hist_errorbar.png")
plt.show()

