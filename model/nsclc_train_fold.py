import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import albumentations as A

from sklearn.model_selection import KFold
from torchsummary import summary
from TransAttUnet import UNet_Attention_Transformer_Multiscale
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, jaccard_score, precision_score, recall_score

# クラスの上の階層に共通関数として1つだけ定義する
def fix_slice_axis(data):
    shape = data.shape
    min_axis = np.argmin(shape)
    if min_axis == 0:     # (Slices, H, W) -> (H, W, Slices)
        return data.transpose(1, 2, 0)
    elif min_axis == 1:   # (H, Slices, W) -> (H, W, Slices)
        return data.transpose(0, 2, 1)
    return data           # すでに (H, W, Slices)

class NSCLCNiftiDataset(Dataset):
    def __init__(self, pairs, target_size=(512, 512), is_train=False):

        self.pairs = pairs
        self.target_size = target_size
        self.is_train = is_train

        # データを増やすために、腫瘍面積上位2枚と
        # うまく予測できていない小さい腫瘍のデータを増やすために
        # 上から4枚目の3枚をつかう

        self.extended_pairs = []

        if self.is_train:
            print(f"Trainデータのスライス拡張を処理")
            for pair in pairs:
                mask_path = pair['label']
                mask_nii = nib.load(mask_path).get_fdata()

                mask_nii = fix_slice_axis(mask_nii) # 呼び出すだけにスッキリ化
                
                # 全スライスの腫瘍面積計算
                num_slices = mask_nii.shape[2]
                slice_areas = [np.sum(mask_nii[:, :, i] > 0) for i in range(num_slices)]

                # 腫瘍が写ってるスライスのインデックスを、面積が大きい順にソート
                valid_slice_indices = [i for i, area in enumerate(slice_areas) if area > 0]
                sorted_indices = sorted(valid_slice_indices, key=lambda i: slice_areas[i], reverse=True)

                # 上位2枚と、4枚目(小さい腫瘍にも対応したいため)
                chosen_indices = []
                if len(sorted_indices) >= 1: chosen_indices.append(sorted_indices[0]) # 1位
                if len(sorted_indices) >= 2: chosen_indices.append(sorted_indices[1]) # 2位
                if len(sorted_indices) >= 4: chosen_indices.append(sorted_indices[3]) # 4位（インデックスは3）
                elif len(sorted_indices) >= 3: chosen_indices.append(sorted_indices[2]) # 4位がない特殊な場合は3位でカバー

                # 3枚それぞれの情報を新しいペアとして登録
                for rank, mask_idx in enumerate(chosen_indices):
                    new_pair = pair.copy()
                    new_pair['chosen_mask_idx'] = mask_idx
                    new_pair['tumor_size'] = slice_areas[mask_idx]

                    # 元の名前の末尾に、何番目のスライスか識別子を付けておく
                    new_pair['extended_name'] = f"{pair['name']}_slice_rank{rank}"
                    self.extended_pairs.append(new_pair)
        
        else:
            # Testデータは最大スライス1枚のみ
            for pair in pairs:
                mask_path = pair['label']
                mask_nii = nib.load(mask_path).get_fdata()

                mask_nii = fix_slice_axis(mask_nii) # 呼び出すだけにスッキリ化
                
                # 全スライスの腫瘍面積計算
                num_slices = mask_nii.shape[2]
                slice_areas = [np.sum(mask_nii[:, :, i] > 0) for i in range(num_slices)]
                max_mask_idx = int(np.argmax(slice_areas))

                new_pair = pair.copy()
                new_pair['chosen_mask_idx'] = max_mask_idx
                new_pair['tumor_size'] = slice_areas[max_mask_idx]
                new_pair['extended_name'] = pair['name']
                self.extended_pairs.append(new_pair)

        # DA定義
        # if self.is_train:
        #     self.da_transform = A.Compose([
        #         A.HorizontalFlip(p=0.5),
        #         A.VerticalFlip(p=0.5),
        #         A.RandomRotate90(p=0.5),
        #         A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5),
        #     ])
    
    def __len__(self):
        return len(self.extended_pairs)
    
    def __getitem__(self, idx):
        pair = self.extended_pairs[idx]
        file_id = pair['extended_name']
        img_path = pair['image']
        mask_path = pair['label']
        max_mask_idx = pair['chosen_mask_idx']
        tumor_size_2d = pair['tumor_size']

        img_nii = nib.load(img_path).get_fdata()
        mask_nii = nib.load(mask_path).get_fdata()

        img_nii = fix_slice_axis(img_nii)
        mask_nii = fix_slice_axis(mask_nii)

        # --- 【ここが重要】画像側のスライス枚数に合わせてリスケールする ---
        num_slices_mask = mask_nii.shape[2]
        num_slices_img = img_nii.shape[2]
        # 画像とマスクの枚数が違う場合でも、相対的な位置を計算する
        z_ratio = num_slices_img / num_slices_mask
        max_img_idx = int(max_mask_idx * z_ratio)
        
        # 最後に念のためガードレールをかける
        max_img_idx = min(max_img_idx, num_slices_img - 1)

        img_slice = img_nii[:, :, max_img_idx]
        mask_slice = mask_nii[:, :, max_mask_idx]

        # CT値の正規化(?わからん -1000~400程度にクリップ)
        # 肺の微細な構造を鮮明に観察するための肺野条件らしい
        img_slice = np.clip(img_slice, -1000, 400)
        #0-1の正規化
        img_slice = (img_slice + 1000) / 1400

        # DA
        # if self.is_train:
        #     augmented = self.da_transform(image=img_slice, mask=mask_slice)
        #     img_slice = augmented['image']
        #     mask_slice = augmented['mask']

        # リサイズとテンソル化
        img_tensor = torch.from_numpy(img_slice).float().unsqueeze(0) # [1, H, W]
        mask_tensor = torch.from_numpy(mask_slice).float().unsqueeze(0)

        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
        mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=self.target_size, mode='nearest').squeeze(0)
        # Datasetの __getitem__ 内
        #img_slice = img_nii[:, :, max_img_idx] # 135番が抜かれる
        #print(f"DEBUG: {file_id} のスライス {max_img_idx} を抽出しました。形: {img_slice.shape}")

        # ここで一度表示して、専用コードの135番と見比べる
        #import matplotlib.pyplot as plt
        #plt.imshow(img_slice, cmap='gray')
        #plt.title("Dataset抽出直後 (Resize前)")
        #plt.show()

        # スライスのピクセル数(腫瘍面積)

        return img_tensor, mask_tensor, file_id, tumor_size_2d

# BCEとDice Lossを混ぜる
class MixLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(MixLoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCELoss()
    
    def forward(self, outputs, targets):
        targets  = targets.float()
        
        probs = torch.sigmoid(outputs)
        
        # 計算不可能にならないように範囲指定
        probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)

        # BCE Lossの計算
        bce_loss = self.bce(probs, targets)
        
        # Dice Lossの計算
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()

        dice_loss = 1.0 - ((2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth))

        total_loss = 0.5 * bce_loss + 0.5 * dice_loss

        return total_loss
    
if __name__ == '__main__':
    num_epochs = 250
    root_dir = '/workspace/NSCLC_NIfTI2'

    images_dir = '/workspace/NSCLC_NIfTI2/imagesTr'
    labels_dir = '/workspace/NSCLC_NIfTI2/labelsTr/Neoplasm_Primary'

    # 全てのファイル名を取得
    all_images = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])

    # 画像とラベルのフルパスのペアをリストに
    valid_pairs = []
    for f in all_images:
        img_path = os.path.join(images_dir, f)
        lbl_path = os.path.join(labels_dir, f)

        # ラベルが存在するか
        if os.path.exists(lbl_path):
            # ペアを辞書形式で保存
            valid_pairs.append({
                'image': img_path,
                'label': lbl_path,
                'name': f
            })
    print(f"有効なペア数: {len(valid_pairs)}")

    valid_pairs = np.array(valid_pairs)

    # K分割
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_splits = {}

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(valid_pairs)):
        fold_key = f"fold_{fold_idx + 1}"
        
        # valid_pairs[i]['image'] と valid_pairs[i]['label'] を取得
        fold_splits[fold_key] = {
            "train_images": [valid_pairs[i]['image'] for i in train_idx],
            "train_masks": [valid_pairs[i]['label'] for i in train_idx],
            "val_images": [valid_pairs[i]['image'] for i in val_idx],
            "val_masks": [valid_pairs[i]['label'] for i in val_idx],
        }

    # JSONファイルに保存
    with open("dataset_splits.json", "w") as f:
        json.dump(fold_splits, f, indent=4)

    print("分割情報を 'dataset_splits.json' に保存しました。")

    print(f"--- {k_folds}分割交差検証を開始します ---")



    # ここから
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for fold, (train_idx, val_idx) in enumerate(kf.split(valid_pairs)):
        print(f"\n========== Fold {fold + 1}/{k_folds} ==========")

        #  該当するFoldのインデックスを使って、Train/Valペアを切り出す
        train_pairs_fold = valid_pairs[train_idx].tolist()
        val_pairs_fold = valid_pairs[val_idx].tolist()

        # このFold専用のデータセットを作成し、閾値（中央値）を計算
        train_measure_ds = NSCLCNiftiDataset(train_pairs_fold, is_train=True)
        val_measure_ds = NSCLCNiftiDataset(val_pairs_fold, is_train=False)

        train_sizes = np.array([train_measure_ds[i][3] for i in range(len(train_measure_ds))])
        val_sizes = np.array([val_measure_ds[i][3] for i in range(len(val_measure_ds))])
        all_sizes = np.concatenate([train_sizes, val_sizes])
        threshold = np.median(all_sizes)

        # 閾値を使って Large データを仕分け、DataLoaderを作成
        train_pairs_large = [p for p in train_measure_ds.extended_pairs if p['tumor_size'] >= threshold]
        val_pairs_large = [p for p in val_measure_ds.extended_pairs if p['tumor_size'] >= threshold]

        train_ds_large = NSCLCNiftiDataset([], is_train=False)
        train_ds_large.extended_pairs = train_pairs_large

        val_ds_large = NSCLCNiftiDataset([], is_train=False)
        val_ds_large.extended_pairs = val_pairs_large

        batch_size = 8
        train_loader_large = DataLoader(train_ds_large, batch_size=batch_size, shuffle=True)
        val_loader_large = DataLoader(val_ds_large, batch_size=batch_size, shuffle=False)

        # 最も重要：Foldごとに「新品のモデル」と「オプティマイザ」を用意する
        model = UNet_Attention_Transformer_Multiscale(n_channels=1, n_classes=1).to(device)
        optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
        criterion = MixLoss()

        # 保存先フォルダ名を Fold ごとに動的変更
        save_dir = f"checkpoints_large_fold{fold+1}"
        os.makedirs(save_dir, exist_ok=True)

        # Foldごとの学習・検証ループ（修正箇所2の `for fold, ...:` の中に配置）
        train_loss_history = []
        val_loss_history = []
        # 勾配蓄積
        accumulation_steps = 3

        try:
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0.0
                optimizer.zero_grad()

                for i, (images, masks, file_ids, sizes) in enumerate(train_loader_large):
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                    loss_scaled = loss / accumulation_steps
                    loss_scaled.backward()

                    if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader_large):
                        optimizer.step()
                        optimizer.zero_grad()

                    train_loss += loss.item()

                avg_train_loss = train_loss / len(train_loader_large)

                # 検証（変数名を val_loader_large に変更）
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for images, masks, file_ids, sizes in val_loader_large:
                        images, masks = images.to(device), masks.to(device)
                        outputs = model(images)
                        v_loss = criterion(outputs, masks)
                        val_loss += v_loss.item()

                avg_val_loss = val_loss / len(val_loader_large)

                train_loss_history.append(avg_train_loss)
                val_loss_history.append(avg_val_loss)

                if (epoch + 1) % 10 == 0:
                    print(f"Fold {fold+1} | Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                    # 修正箇所2で定義した save_dir（Foldごとのフォルダ）にモデルを保存
                    save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
                    torch.save(model.state_dict(), save_path)

            # グラフも Fold ごとのフォルダ内に保存
            plt.figure(figsize=(10, 5))
            plt.plot(train_loss_history, label='Train Loss')
            plt.plot(val_loss_history, label='Val Loss')
            plt.title(f'Fold {fold+1} Loss History')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "loss_history.png"))
            plt.close()
        # for epoch ループの直後に except を追加する
        except KeyboardInterrupt:
            print(f"\n[中断] Fold {fold+1} の学習が手動で停止されました。")
            print(f"Fold {fold+1} の中断時点の状態を保存します...")
            
            # 中断時点のモデル重みを保存
            interrupted_model_path = os.path.join(save_dir, "model_interrupted.pth")
            torch.save(model.state_dict(), interrupted_model_path)

            # 中断時点までのグラフを保存
            if len(train_loss_history) > 0:
                plt.figure(figsize=(10, 5))
                plt.plot(train_loss_history, label='Train Loss')
                plt.plot(val_loss_history, label='Val Loss')
                plt.title(f'Fold {fold+1} Loss History (Interrupted at Epoch {len(train_loss_history)})')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(save_dir, "loss_history_interrupted.png"))
                plt.close()

# 重みロード
#model.load_state_dict(torch.load('/workspace/checkpoints_small_epoch200/model_epoch_200.pth')) # 保存されたファイル名

#最適化アルゴリズム 1e-4

# 30エポックごとに学習率を0.1倍に
# scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
# 損失関数

print("すべての処理が終了しました。")