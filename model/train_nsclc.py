import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from TransAttUnet import UNet_Attention_Transformer_Multiscale

class NSCLCNiftiDataset(Dataset):
    def __init__(self, root_dir, file_list, target_size=(512, 512)):
        self.img_dir = os.path.join(root_dir, 'imagesTr')
        self.mask_dir = os.path.join(root_dir, 'labelsTr')
        self.file_names = file_list
        self.target_size = target_size
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        # NIfTI読み込み
        img_path = os.path.join(self.img_dir, self.file_name[idx])
        mask_path = os.path.join(self.mask_dir, self.file_names[idx])

        img_nii = nib.load(img_path).get_fdata()
        mask_nii = nib.load(mask_path).get_fdata()

        # 3Dから2Dスライス(腫瘍がいちばん大きいスライスとる)
        # 配列の中で最大値となってる要素のうち先頭のインデックスを返す(argmax)
        max_slice = np.argmax(np.sum(mask_nii), axis=(0, 1))
        img_slice = img_nii[:, :, max_slice]
        mask_slice = mask_nii[:, :, max_slice]

        # CT値の正規化(?わからん -1000~400程度にクリップ)
        # 肺の微細な構造を鮮明に観察するための肺野条件らしい
        img_slice = np.clip(img_slice, -1000, 400)
        #0-1の正規化
        img_slice = (img_slice + 1000) / 1400

        # リサイズとテンソル化
        img_tensor = torch.from_numpy(img_slice).float().unsqueeze(0) # [1, H, W]
        mask_tensor = torch.from_numpy(mask_slice).float().unsqueeze(0)

        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
        mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=self.target_size, mode='nearest').squeeze(0)

        return img_tensor, mask_tensor

if __name__ == '__main__':
    root_dir = 'NSCLC_NIfTI'

    # 全てのファイル名を取得
    all_files = sorted([f for f in os.listdir(os.path.join(root_dir, 'imagesTr')) if f.endswith('.nii.gz')])

    # train 8  test 2で分割
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

    print(f"--- データ分割完了 ---")
    print(f"学習用 (train): {len(train_files)} 症例")
    print(f"テスト用 (test): {len(test_files)} 症例")

# 学習設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = NSCLCNiftiDataset(root_dir=root_dir, file_list=train_files)
test_dataset = NSCLCNiftiDataset(root_dir=root_dir, file_list=test_files)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = UNet_Attention_Transformer_Multiscale(n_channels=1, n_classes=1).to(device)
#最適化アルゴリズム
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
# 損失関数
criterion = nn.BCEWithLogitsLoss()

num_epochs = 50

print("学習を開始...")
for epoch in range(num_epochs):
    # 学習
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        # 予測と計算
        outputs = model(images)
        loss = criterion(outputs, masks)
