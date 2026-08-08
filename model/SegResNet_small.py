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
    "checkpoints_segres_small",
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

test_sizes = []

for i in range(len(test_dataset)):
    item = test_dataset[i]
    test_sizes.append(
        item["tumor_size"]
    )

threshold = np.median(train_sizes)


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

# Whole Body CT weight
ckpt = torch.load(
    "bundles/wholeBody_ct_segmentation/models/model_lowres.pt",
    map_location="cpu"
)

print(type(ckpt))

if isinstance(ckpt, dict):
    print(ckpt.keys())

pretrained_dict = ckpt

model_dict = model.state_dict()

# output layerなど違う部分を除外

pretrained_dict = {
    k:v
    for k,v in pretrained_dict.items()

    if k in model_dict
    and v.shape == model_dict[k].shape

}

model_dict.update(
    pretrained_dict
)

model.load_state_dict(
    model_dict
)

print(
    f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers"
)
model = model.to(device)

# Loss
loss_fn = DiceCELoss(
    to_onehot_y=True,
    softmax=True
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

loss_history = []

#train
epochs = 100

for epoch in range(epochs):

    model.train()
    total_loss=0

    for batch in train_loader_small:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = loss_fn(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader_small)

    loss_history.append(avg_loss)

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss={avg_loss:.4f}"
    )

    # 10 epochごと保存
    if (epoch+1)%10==0:
        torch.save(
            model.state_dict(),
            f"checkpoints_segres_small/small_epoch_{epoch+1}.pth"
        )

import matplotlib.pyplot as plt


plt.figure(figsize=(8,5))

plt.plot(
    loss_history,
    label="Train Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Training Loss")

plt.legend()

plt.grid()

plt.savefig(
    "checkpoints_segres_small/loss_curve.png"
)

plt.close()
