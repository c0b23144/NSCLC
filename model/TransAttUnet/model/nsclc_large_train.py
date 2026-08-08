import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import albumentations as A

from torchsummary import summary
from TransAttUnet import UNet_Attention_Transformer_Multiscale
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, jaccard_score, precision_score, recall_score

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

                shape = mask_nii.shape
                min_axis = np.argmin(shape)
                if min_axis == 0:
                    mask_nii = mask_nii.transpose(1, 2, 0)
                elif min_axis == 1:
                    mask_nii = mask_nii.transpose(0, 2, 1)
                
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

                # 軸の修正
                shape = mask_nii.shape
                min_axis = np.argmin(shape)
                if min_axis == 0:
                    mask_nii = mask_nii.transpose(1, 2, 0)
                elif min_axis == 1:
                    mask_nii = mask_nii.transpose(0, 2, 1)
                
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

        def fix_slice_axis(data):
            shape = data.shape
            # 3つの軸の中で、512より小さい（枚数と思われる）軸のインデックスを探す
            min_axis = np.argmin(shape)
            if min_axis == 0:   # (Slices, H, W) -> (H, W, Slices)
                return data.transpose(1, 2, 0)
            elif min_axis == 1: # (H, Slices, W) -> (H, W, Slices)
                return data.transpose(0, 2, 1)
            return data         # すでに (H, W, Slices)

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

if __name__ == '__main__':
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

    # train 8  test 2で分割
    train_pairs, test_pairs = train_test_split(valid_pairs, test_size=0.2, random_state=42)

    print(f"--- データ分割完了 ---")

    print(f"学習用ペア数: {len(train_pairs)}")
    print(f"テスト用ペア数: {len(test_pairs)}")

    # 確認：テスト用の1番目のデータが正しくペアになっているか
    print(f"\n[確認] テスト用データの1番目:")
    print(f"名前: {test_pairs[0]['name']}")
    print(f"画像: {test_pairs[0]['image']}")
    print(f"ラベル: {test_pairs[0]['label']}")

    # Datasetのインスタンス化
    train_dataset = NSCLCNiftiDataset(train_pairs, is_train=True)
    test_dataset = NSCLCNiftiDataset(test_pairs, is_train=False)

    print(f"\n--- Dataset作成完了 ---")
    print(f"学習用: {len(train_dataset)}件, テスト用: {len(test_dataset)}件")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print(f"DataLoader準備完了")

    # 動作確認
    img, mask, name, size = train_dataset[0]
    print(f"\n[動作確認成功]")
    print(f"患者ID {name}")
    print(f"画像の形: {img.shape}")
    print(f"マスクの形: {mask.shape}")

# trainとtestを腫瘍の大きさ(大, 小)で分ける
# 測定用のデータセット
train_measure_dataset = NSCLCNiftiDataset(train_pairs, is_train=True)
test_measure_dataset = NSCLCNiftiDataset(test_pairs, is_train=False)

# Datasetから計算済みのサイズだけ回収(最大面積の腫瘍とってるため)
train_sizes = np.array([train_measure_dataset[i][3]for i in range(len(train_measure_dataset))])
test_sizes = np.array([test_measure_dataset[i][3] for i in range(len(test_measure_dataset))])
all_sizes = np.concatenate([train_sizes, test_sizes])

# 閾値(中央値)の決定とヒストグラム
threshold = np.median(all_sizes)

plt.figure(figsize=(10, 5))
plt.hist(train_sizes, bins=30, color='royalblue', alpha=0.6, label=f'Train Data (n={len(train_sizes)})', edgecolor='black')
plt.hist(test_sizes, bins=30, color='orange', alpha=0.6, label=f'Test Data (n={len(test_sizes)})', edgecolor='black')
plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold ({int(threshold)} px)')
plt.title('Tumor Size Distribution (Direct from Dataset)', fontsize=14)
plt.xlabel('Tumor Size (Pixels on Max Slice)')
plt.ylabel('Number of Patients')
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.savefig("checkpoints_large_epoch250/hist.png")
print("保存しました。")

print(f"割り出された閾値: {int(threshold)} ピクセル")

# 3倍に拡張されているextended_pairsを直接使って、閾値で仕分ける
train_extended_pairs_large = [p for p in train_measure_dataset.extended_pairs if p['tumor_size'] >= threshold]
train_extended_pairs_small = [p for p in train_measure_dataset.extended_pairs if p['tumor_size'] < threshold]

test_extended_pairs_large = [p for p in test_measure_dataset.extended_pairs if p['tumor_size'] >= threshold]
test_extended_pairs_small = [p for p in test_measure_dataset.extended_pairs if p['tumor_size'] < threshold]

# 仕分けたリストを使って、新しくDatasetを作る
# trainはすでに拡張済みのリストを渡すため、これ以上増えないように is_train=False でインスタンス化
train_ds_large = NSCLCNiftiDataset([], is_train=False)  # 空で初期化
train_ds_large.extended_pairs = train_extended_pairs_large

train_ds_small = NSCLCNiftiDataset([], is_train=False)
train_ds_small.extended_pairs = train_extended_pairs_small

test_ds_large = NSCLCNiftiDataset([], is_train=False)
test_ds_large.extended_pairs = test_extended_pairs_large

test_ds_small = NSCLCNiftiDataset([], is_train=False)
test_ds_small.extended_pairs = test_extended_pairs_small


# サイズ別DataLoaderの作成
batch_size = 8
train_loader_large = DataLoader(train_ds_large, batch_size=batch_size, shuffle=True)
train_loader_small = DataLoader(train_ds_small, batch_size=batch_size, shuffle=True)
test_loader_large = DataLoader(test_ds_large, batch_size=batch_size, shuffle=False)
test_loader_small = DataLoader(test_ds_small, batch_size=batch_size, shuffle=False)


# 拡張された後の正確な件数をプリント
print(f"train_loader_large: {len(train_ds_large)}件 / test_loader_large: {len(test_ds_large)}件")
print(f"train_loader_small: {len(train_ds_small)}件 / test_loader_small: {len(test_ds_small)}件")

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

# 学習設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet_Attention_Transformer_Multiscale(n_channels=1, n_classes=1).to(device)

# 重みロード
#model.load_state_dict(torch.load('/workspace/checkpoints_small_epoch200/model_epoch_200.pth')) # 保存されたファイル名

#最適化アルゴリズム 1e-4
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
# 30エポックごとに学習率を0.1倍に
# scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
# 損失関数
criterion = MixLoss()

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# small_epochは50回追加学習してるから気を付ける
num_epochs = 250

train_loss_history = []
val_loss_history = []

print("学習を開始...")
# 勾配蓄積回数
accumulation_steps = 3

try:
    # small largeを変えるときに、画像を保存するときの場所も変える（loss, hist）
    for epoch in range(num_epochs):
        # 学習
        model.train()
        train_loss = 0.0

        # 蓄積を開始する前に勾配リセット
        optimizer.zero_grad()

        for i, (images, masks, file_ids, sizes) in enumerate(train_loader_large):
            images, masks = images.to(device), masks.to(device)

            # 予測と計算
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Lossを蓄積回数で割る
            loss_scaled = loss / accumulation_steps

            # 勾配の計算
            loss_scaled.backward()
            
            # 重みの更新
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader_large):
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader_large)

        # 検証(テストデータでの評価)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks, file_ids, sizes in test_loader_large:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                v_loss = criterion(outputs, masks)
                val_loss += v_loss.item()

        avg_val_loss = val_loss / len(test_loader_large)

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        # 進捗を表示
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # モデルの保存
        save_dir = "checkpoints_large_epoch250"
        os.makedirs(save_dir, exist_ok=True)

        # 10エポックごとに重みを保存
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--- モデルを保存しました: {save_path}")

        # 【おまけ】毎エポックごとに上書き保存しておけば、いつ止めても最新のグラフが見られます
        if (epoch + 1) % 1 == 0:
            plt.figure(figsize=(10, 5))
            plt.plot(train_loss_history, label='Train Loss')
            plt.plot(val_loss_history, label='Val Loss')
            plt.title('Training and Validation Loss (Live)')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig("checkpoints_large_epoch250/loss_history.png")
            plt.close() # メモリリーク防止のために必ず閉じる

except KeyboardInterrupt:
    print("\n[中断] 学習が途中で停止されました。")

finally:
    # 正常終了時、またはCtrl+Cでの中断時、どちらでも必ずここが実行されます
    if len(train_loss_history) > 0:
        print("これまでの学習曲線をグラフに保存しています...")
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_history, label='Train Loss')
        plt.plot(val_loss_history, label='Val Loss')
        plt.title('Training and Validation Loss (Final/Interrupted)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        os.makedirs("checkpoints_large_epoch250", exist_ok=True)
        plt.savefig("checkpoints_large_epoch250/loss_history.png")
        plt.close()
        print("学習曲線のグラフを保存しました。")
    else:
        print("1エポックも完了していないため、グラフは生成されませんでした。")

print("すべての処理が終了しました。")