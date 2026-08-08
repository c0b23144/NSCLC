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

    # 不一致ファイルがあるかどうか
    # testデータの並び替え(順序が違うだけでテストデータの不一致になってしまうのを避ける)
    current_test_names = sorted([p['name'] for p in test_pairs])

    # ログ保存用フォルダ
    log_dir = "/workspace/split_check/"
    os.makedirs(log_dir, exist_ok=True)
    master_log_path = os.path.join(log_dir, "test_files_master.txt")

    print("\n" + "="*60)
    
    # 初回実行時(基準となるデータを保存)
    if not os.path.exists(master_log_path):
        with open(master_log_path, "w") as f:
            f.write("\n".join(current_test_names))
        print("[初回実行] 今回のテストデータをマスターとして保存しました")
        print("次回以降、このデータとズレがないかチェックされます")
    
    # 2回目以降の実行時(前回のデータとズレがないかチェック)
    else:
        with open(master_log_path, "r") as f:
            master_test_names = sorted([line.strip() for line in f.readlines() if line.strip()])
        
        # ズレがある場合
        if current_test_names != master_test_names:
            print("[警告: データのズレを検知しました]")
            print("今回分割されたデータが前回と異なっています")
            print("-" * 40)

            # ずれている中身を比較
            set_master = set(master_test_names)
            set_current = set(current_test_names)

            # 前回にあって今回消えたもの
            missing_files = sorted(list(set_master - set_current))
            # 前回（マスター）になくて、今回新しく入ってしまったもの
            added_files = sorted(list(set_current - set_master))

            print(f"前回（マスター）から消えてしまったファイル (計 {len(missing_files)} 件):")
            for name in missing_files[:5]:  # 多すぎる場合に備えて先頭5件を表示
                print(f"  - {name}")
            if len(missing_files) > 5:
                print(f"  ... 他 {len(missing_files) - 5} 件")
                
            print(f"\n今回新しく混入してしまったファイル (計 {len(added_files)} 件):")
            for name in added_files[:5]:   # 多すぎる場合に備えて先頭5件を表示
                print(f"  - {name}")
            if len(added_files) > 5:
                print(f"  ... 他 {len(added_files) - 5} 件")
            print("-" * 40)

            print("安全のため、前回のデータを使うよう修復・上書きします")

            # マスターファイルに登録されているファイル名だけをvalid_pairsから強制的に抽出して復元
            test_pairs = [p for p in valid_pairs if p['name'] in set_master]
            train_pairs = [p for p in valid_pairs if p['name'] not in set_master]

            print(f"→ 自動修復完了：Testデータ（{len(test_pairs)}件）、Trainデータ（{len(train_pairs)}件）を完全に同期しました。")
        
        # ズレがない場合
        else:
            print("【 判定： 安全（前回のデータと完全に一致しています） 】")
            print("データはズレていません。安心してそのまま評価を続けてください。")
    print("="*60 + "\n")

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

# 
actual_bins = np.linspace(all_sizes.min(), all_sizes.max(), 31)

plt.figure(figsize=(10, 5))
plt.hist(train_sizes, bins=actual_bins, color='royalblue', alpha=0.6, label=f'Train Data (n={len(train_sizes)})', edgecolor='black')
plt.hist(test_sizes, bins=actual_bins, color='orange', alpha=0.6, label=f'Test Data (n={len(test_sizes)})', edgecolor='black')
plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold ({int(threshold)} px)')
plt.title('Tumor Size Distribution (Direct from Dataset)', fontsize=14)
plt.xlabel('Tumor Size (Pixels on Max Slice)')
plt.ylabel('Number of Patients')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.yscale('log')

plt.savefig("output/hist_bins/hist.png")
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
#model.load_state_dict(torch.load('/workspace/TransAttUnet/model/checkpoints/model_epoch_50.pth')) # 保存されたファイル名

#最適化アルゴリズム 1e-4
optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-4)
# 30エポックごとに学習率を0.1倍に
# scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
# 損失関数
criterion = MixLoss()

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 評価

#epoch_num = 20         # 評価したいエポックの数値
metrics_list = ["DICE", "IoU", "ACC", "REC", "PRE"]

print(f"\n重みをロードして、大・小別 × 各指標の評価グラフを生成")

# 5つの重みを重ねる評価・プロット（エラーバーなし・同色/線種違いルール適用）
# 1. 指定された名前（small, large, mix, mix + small, mix + large）で重みパスを定義
weight_configs = {
    "small": '/workspace/checkpoints_small_epoch200/model_epoch_50.pth',
    "large": '/workspace/checkpoints_large_epoch250/model_epoch_250.pth',
    "mix": '/workspace/TransAttUnet/model/checkpoints/model_epoch_50.pth',
    "mix + small": '/workspace/checkpoints_small/model_epoch_20.pth',
    "mix + large": '/workspace/checkpoints_large/model_epoch_20.pth'
}

# 指標リストとビンの定義
metrics_list = ["DICE", "IoU", "ACC", "REC", "PRE"]
bin_edges = actual_bins
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 全モデルのプロットデータを格納する辞書
all_weight_results = {}

print("\n--- 5つの重みモデルの正確な評価を開始（small/largeを完全分離） ---")

for weight_name, weight_path in weight_configs.items():
    print(f"\n重みをロード中... 【 {weight_name} 】")
    
    if not os.path.exists(weight_path):
        print(f"[警告] 重みファイルが見つかりません: {weight_path}")
        continue
        
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # データセット（small / large）ごとに個別に結果を回収するための辞書
    collected_data = {
        "small_data": {"sizes": [], "scores": {m: [] for m in metrics_list}},
        "large_data": {"sizes": [], "scores": {m: [] for m in metrics_list}}
    }

    # 各ローダーを個別に回す
    for loader_type, eval_loader in [("small_data", test_loader_small), ("large_data", test_loader_large)]:
        with torch.no_grad():
            for data in eval_loader:
                images, masks, _, sizes = data
                images = images.to(device)
                outputs = model(images)
                preds = (torch.sigmoid(outputs) > 0.5).float()

                for j in range(images.size(0)):
                    p = preds[j].cpu().numpy().flatten()
                    t = masks[j].numpy().flatten()
                    t_size = sizes[j].item()

                    p_score = precision_score(t, p, zero_division=0)
                    r_score = recall_score(t, p, zero_division=0)
                    i_score = jaccard_score(t, p, zero_division=0)
                    a_score = accuracy_score(t, p)
                    d_score = (2 * i_score) / (i_score + 1) if i_score > 0 else 0.0

                    if np.sum(t) == 0 and np.sum(p) == 0:
                        d_score, i_score = 1.0, 1.0
                    
                    collected_data[loader_type]["sizes"].append(t_size)
                    collected_data[loader_type]["scores"]["DICE"].append(d_score)
                    collected_data[loader_type]["scores"]["IoU"].append(i_score)
                    collected_data[loader_type]["scores"]["ACC"].append(a_score)
                    collected_data[loader_type]["scores"]["REC"].append(r_score)
                    collected_data[loader_type]["scores"]["PRE"].append(p_score)

    # ビンごとの「平均値」を、データソース別に計算
    weight_metrics_summary = {}
    for metric_name in metrics_list:
        small_means = []
        large_means = []
        
        # 30ビンに対して集計
        for j in range(30):
            low, high = bin_edges[j], bin_edges[j+1]
            is_last_bin = (j == 29)
            
            # small_data の集計
            s_sizes = np.array(collected_data["small_data"]["sizes"])
            s_scores = np.array(collected_data["small_data"]["scores"][metric_name])
            if is_last_bin:
                s_in_bin = (s_sizes >= low) & (s_sizes <= high)
            else:
                s_in_bin = (s_sizes >= low) & (s_sizes < high)
            
            if np.sum(s_in_bin) > 0:
                small_means.append(np.mean(s_scores[s_in_bin]))
            else:
                small_means.append(np.nan)
                
            # large_data の集計
            l_sizes = np.array(collected_data["large_data"]["sizes"])
            l_scores = np.array(collected_data["large_data"]["scores"][metric_name])
            if is_last_bin:
                l_in_bin = (l_sizes >= low) & (l_sizes <= high)
            else:
                l_in_bin = (l_sizes >= low) & (l_sizes < high)
                
            if np.sum(l_in_bin) > 0:
                large_means.append(np.mean(l_scores[l_in_bin]))
            else:
                large_means.append(np.nan)
                
        # 1本化した全体平均（1枚目の折れ線グラフ用）
        combined_means = []
        for j in range(30):
            s_val = small_means[j]
            l_val = large_means[j]
            if not np.isnan(s_val) and not np.isnan(l_val):
                combined_means.append((s_val + l_val) / 2.0) # 両方あれば平均
            elif not np.isnan(s_val):
                combined_means.append(s_val)
            else:
                combined_means.append(l_val) # なければ片方の値（両方NaNならNaNになる）

        weight_metrics_summary[metric_name] = {
            "small_mean": np.array(small_means),
            "large_mean": np.array(large_means),
            "combined_mean": np.array(combined_means)
        }
        
    all_weight_results[weight_name] = weight_metrics_summary

print("--- 評価およびデータの完全分離集計が完了しました ---")

# ========================================================================
# 2. 【プロットフェーズ】
# 1枚目（全体折れ線）：mixを含めた5モデルで全体の推移を描写
# 2枚目（Bin 3棒グラフ）：mixを除外した4モデルのみですっきり比較
# ========================================================================
# データが存在するビンを特定
valid_bins_indices = []
for j in range(30):
    has_data = False
    for model_key in weight_configs.keys():
        if model_key in all_weight_results:
            val = all_weight_results[model_key]["DICE"]["combined_mean"][j]
            if not np.isnan(val):
                has_data = True
                break
    if has_data:
        valid_bins_indices.append(j)

num_final_bins = len(valid_bins_indices)
x_indices = np.arange(num_final_bins)

bin3_final_idx = None
for idx, j in enumerate(valid_bins_indices):
    if bin_edges[j] <= threshold <= bin_edges[j+1]:
        bin3_final_idx = idx
        break

final_bin_labels = []
for idx, j in enumerate(valid_bins_indices):
    low = int(bin_edges[j])
    high = int(bin_edges[j+1])
    range_str = f"{low}-{high} px"
    if idx == bin3_final_idx:
        final_bin_labels.append(f"{range_str}\n(Split)")
    else:
        final_bin_labels.append(range_str)

# 1枚目用スタイル（5モデルすべて）
plot_styles = {
    "small":       {"color": "#8BC34A", "marker": "o", "linestyle": "-"},
    "mix + small": {"color": "#8BC34A", "marker": "v", "linestyle": "--"},
    "large":       {"color": "#E65100", "marker": "s", "linestyle": "-"},
    "mix + large": {"color": "#E65100", "marker": "D", "linestyle": "--"},
    "mix":         {"color": "#2196F3", "marker": "^", "linestyle": "-"}
}
model_order_all = ["small", "mix + small", "large", "mix + large", "mix"]

# 2枚目（Bin 3詳細）用モデルリスト ➔ mix を完全除外！
model_order_bin3 = ["small", "mix + small", "large", "mix + large"]

for metric_name in metrics_list:
    
    # --------------------------------------------------------------------
    # 【1枚目の図】全体推移を示す折れ線グラフ（5モデルすべて）
    # --------------------------------------------------------------------
    plt.figure(figsize=(15, 7.5))
    for idx, model_key in enumerate(model_order_all):
        if model_key not in all_weight_results: continue
        
        raw_combined = all_weight_results[model_key][metric_name]["combined_mean"]
        final_means = [raw_combined[j] for j in valid_bins_indices]
            
        style = plot_styles[model_key]
        plt.plot(
            x_indices, final_means, 
            label=model_key, color=style["color"], marker=style["marker"],
            linestyle=style["linestyle"], linewidth=2.5, markersize=7, alpha=0.9
        )
        
    if bin3_final_idx is not None:
        plt.axvline(bin3_final_idx, color='#D32F2F', linestyle=':', linewidth=2, label=f'Threshold ({int(threshold)} px)')

    plt.xticks(x_indices, final_bin_labels, rotation=45, ha='right', fontsize=9.5)
    plt.xlabel('Tumor Size Ranges (Pixels on Max Slice)', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel(f'Mean {metric_name} Score', fontsize=12, fontweight='bold')
    plt.title(f'Comparison of 5 Models: {metric_name} Performance Trend ({num_final_bins} Bins)', fontsize=14, fontweight='bold', pad=15)
    plt.xlim(-0.5, num_final_bins - 0.5)
    plt.ylim(-0.05, 1.05) if metric_name != "ACC" else plt.ylim(0.9, 1.01)
    plt.legend(loc='upper right', fontsize=10, frameon=True, shadow=True, ncol=6)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    
    save_path_1 = f"/workspace/output/combined_evaluation/combined_models_final_overview_line_{metric_name}.eps"
    plt.savefig(save_path_1, dpi=200)
    plt.close()

    # --------------------------------------------------------------------
    # 【2枚目の図】Bin 3（Split）専用：4モデル（mix除外）の棒グラフ
    # --------------------------------------------------------------------
    if bin3_final_idx is None:
        continue
        
    initial_bin3_idx = valid_bins_indices[bin3_final_idx]
    
    plt.figure(figsize=(11, 6.5))
    x_models = np.arange(len(model_order_bin3))
    bar_width_2 = 0.35  # 2本の棒を並べるための幅
    
    # データの種類に応じた固定カラー
    color_small_data = "#8BC34A"  # small_data（緑）
    color_large_data = "#E65100"  # large_data（オレンジ）
    
    for idx, model_key in enumerate(model_order_bin3):
        if model_key not in all_weight_results: continue
        
        # すべてのモデルから、それぞれのデータの評価値を取得
        s_val = all_weight_results[model_key][metric_name]["small_mean"][initial_bin3_idx]
        l_val = all_weight_results[model_key][metric_name]["large_mean"][initial_bin3_idx]
        
        # ■ ① 左側の棒：small_data の評価スコア（緑色）
        if not np.isnan(s_val):
            plt.bar(
                idx - bar_width_2/2, s_val, width=bar_width_2,
                color=color_small_data, edgecolor='black', linewidth=1.2,
                label="on small_data (≤1289 px)" if idx == 0 else ""
            )
            plt.text(idx - bar_width_2/2, s_val + 0.01, f"{s_val:.2f}", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
            
        # ■ ② 右側の棒：large_data の評価スコア（オレンジ色）
        if not np.isnan(l_val):
            plt.bar(
                idx + bar_width_2/2, l_val, width=bar_width_2,
                color=color_large_data, edgecolor='black', linewidth=1.2,
                label="on large_data (>1289 px)" if idx == 0 else ""
            )
            plt.text(idx + bar_width_2/2, l_val + 0.01, f"{l_val:.2f}", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    plt.xticks(x_models, model_order_bin3, fontsize=11, fontweight='bold')
    plt.xlabel('Evaluation Models (Specialist Focus)', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel(f'Mean {metric_name} Score', fontsize=12, fontweight='bold')
    
    range_str = f"{int(bin_edges[initial_bin3_idx])}-{int(bin_edges[initial_bin3_idx+1])} px"
    plt.title(f'Bin 3 ({range_str}) Zoom-in Analysis: {metric_name} Comparison on Small vs Large Data', fontsize=13, fontweight='bold', pad=15)
    plt.ylim(-0.05, 1.1) if metric_name != "ACC" else plt.ylim(0.9, 1.05)
    
    # 凡例をわかりやすく表示
    plt.legend(loc='upper right', fontsize=10, frameon=True, shadow=True)
    plt.grid(axis='y', linestyle=":", alpha=0.5)
    plt.tight_layout()
    
    save_path_2 = f"/workspace/output/combined_evaluation/combined_models_final_bin3_detail_{metric_name}.eps"
    plt.savefig(save_path_2, dpi=200)
    plt.close()

print("\nすべて修正が完了し、2つのグラフの保存が完了しました！")


# ========================================================================
# プロットフェーズ：プレゼンテーション・プレゼン資料最適化版
# ========================================================================

# 保存先ディレクトリの自動作成
output_dir = "/workspace/output/slide_combined_evaluation"
os.makedirs(output_dir, exist_ok=True)

# プレゼン用共通フォントサイズ設定
# 2. フォントサイズ設定（定義漏れ防止）
# --- プレゼン（スライド）用フォント設定 ---
FONT_TITLE_SLIDE = 24
FONT_LABEL_SLIDE = 20
FONT_TICK_SLIDE = 16
FONT_LEGEND_SLIDE = 18
FONT_BAR_TEXT_SLIDE = 16

# --- 卒業論文（Thesis）用フォント設定 ---
FONT_TITLE_THESIS = 13     
FONT_LABEL_THESIS = 11     
FONT_TICK_THESIS = 9.5     
FONT_LEGEND_THESIS = 10    
LINE_WIDTH_THESIS = 2.0
MARKER_SIZE_THESIS = 7.5

# データが存在するビンを特定
valid_bins_indices = []
for j in range(30):
    has_data = False
    for model_key in weight_configs.keys():
        if model_key in all_weight_results:
            val = all_weight_results[model_key]["DICE"]["combined_mean"][j]
            if not np.isnan(val):
                has_data = True
                break
    if has_data:
        valid_bins_indices.append(j)

num_final_bins = len(valid_bins_indices)
x_indices = np.arange(num_final_bins)

bin3_final_idx = None
for idx, j in enumerate(valid_bins_indices):
    if bin_edges[j] <= threshold <= bin_edges[j+1]:
        bin3_final_idx = idx
        break

final_bin_labels = []
for idx, j in enumerate(valid_bins_indices):
    low = int(bin_edges[j])
    high = int(bin_edges[j+1])
    range_str = f"{low}-{high}"
    if idx == bin3_final_idx:
        final_bin_labels.append(f"{range_str}\n(Split)")
    else:
        final_bin_labels.append(range_str)

# スタイル・カラー設定
plot_styles = {
    "small":       {"color": "#8BC34A", "marker": "o", "linestyle": "-"},
    "mix + small": {"color": "#8BC34A", "marker": "v", "linestyle": "--"},
    "large":       {"color": "#E65100", "marker": "s", "linestyle": "-"},
    "mix + large": {"color": "#E65100", "marker": "D", "linestyle": "--"},
    "mix":         {"color": "#2196F3", "marker": "^", "linestyle": "-"}
}
model_order_all = ["small", "mix + small", "large", "mix + large", "mix"]
model_order_bin3 = ["small", "mix + small", "large", "mix + large"]

color_small_data = "#8BC34A"  
color_large_data = "#E65100"  

for metric_name in metrics_list:
    
    # --------------------------------------------------------------------
    # 【スライド用 1枚目】全体比較（5モデル）
    # --------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 9))
    for model_key in model_order_all:
        if model_key not in all_weight_results: continue
        raw_combined = all_weight_results[model_key][metric_name]["combined_mean"]
        final_means = [raw_combined[j] for j in valid_bins_indices]
        style = plot_styles[model_key]
        ax.plot(x_indices, final_means, label=model_key, color=style["color"], 
                marker=style["marker"], linestyle=style["linestyle"], 
                linewidth=4, markersize=12, alpha=0.9)
        
    if bin3_final_idx is not None:
        ax.axvline(bin3_final_idx, color='#D32F2F', linestyle=':', linewidth=3, label='Threshold Split')

    ax.set_xticks(x_indices)
    ax.set_xticklabels(final_bin_labels, rotation=45, ha='right', fontsize=FONT_TICK_SLIDE)
    ax.set_xlabel('Tumor Size (Pixels)', fontsize=FONT_LABEL_SLIDE, fontweight='bold', labelpad=15)
    ax.set_ylabel(f'Mean {metric_name}', fontsize=FONT_LABEL_SLIDE, fontweight='bold')
    ax.set_title(f'Performance Comparison: {metric_name}', fontsize=FONT_TITLE_SLIDE, fontweight='bold', pad=30)
    
    # 🎓 凡例を枠外右上に配置（重なり防止）
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=FONT_LEGEND_SLIDE, 
              frameon=True, shadow=True, edgecolor='black', facecolor='white')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    # 枠外の凡例が途切れないようにレイアウトを自動調整
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"1_Comparison_All_{metric_name}slide.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --------------------------------------------------------------------
    # 【スライド用 2枚目】Bin 3 詳細比較（4モデル・データ色分け）
    # --------------------------------------------------------------------
    if bin3_final_idx is not None:
        idx_b3 = valid_bins_indices[bin3_final_idx]
        fig, ax = plt.subplots(figsize=(14, 8))
        x_models = np.arange(len(model_order_bin3))
        bw = 0.35
        
        for idx, model_key in enumerate(model_order_bin3):
            if model_key not in all_weight_results: continue
            s_val = all_weight_results[model_key][metric_name]["small_mean"][idx_b3]
            l_val = all_weight_results[model_key][metric_name]["large_mean"][idx_b3]
            
            ax.bar(idx - bw/2, s_val, width=bw, color=color_small_data, edgecolor='black', linewidth=1.5,
                   label="Small Data Segment" if idx == 0 else "")
            ax.bar(idx + bw/2, l_val, width=bw, color=color_large_data, edgecolor='black', linewidth=1.5,
                   label="Large Data Segment" if idx == 0 else "")
            
            ax.text(idx - bw/2, s_val + 0.01, f"{s_val:.2f}", ha='center', va='bottom', fontsize=16, fontweight='bold')
            ax.text(idx + bw/2, l_val + 0.01, f"{l_val:.2f}", ha='center', va='bottom', fontsize=16, fontweight='bold')

        ax.set_xticks(x_models)
        ax.set_xticklabels(model_order_bin3, fontsize=FONT_TICK_SLIDE + 2, fontweight='bold')
        ax.set_ylabel(f'Mean {metric_name}', fontsize=FONT_LABEL_SLIDE, fontweight='bold')
        range_str = f"{int(bin_edges[idx_b3])}-{int(bin_edges[idx_b3+1])} px"
        ax.set_title(f'Strategic Split Analysis (Bin: {range_str})', fontsize=FONT_TITLE_SLIDE, fontweight='bold', pad=30)
        
        # 🎓 凡例を枠外右上に配置
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=FONT_LEGEND_SLIDE, frameon=True, shadow=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, 1.15) if metric_name != "ACC" else ax.set_ylim(0.9, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"2_Bin3_Analysis_{metric_name}slide.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # --------------------------------------------------------------------
    # 【スライド用 3枚目】Mix モデル単体
    # --------------------------------------------------------------------
    if "mix" in all_weight_results:
        fig, ax = plt.subplots(figsize=(16, 9))
        raw_mix = all_weight_results["mix"][metric_name]["combined_mean"]
        final_mix = [raw_mix[j] for j in valid_bins_indices]
        
        ax.plot(x_indices, final_mix, label="Baseline: Mix Model", color="#2196F3", 
                marker='^', linestyle='-', linewidth=5, markersize=15, alpha=1.0)
        
        if bin3_final_idx is not None:
            ax.axvline(bin3_final_idx, color='#D32F2F', linestyle=':', linewidth=3, label='Threshold Split')

        ax.set_xticks(x_indices)
        ax.set_xticklabels(final_bin_labels, rotation=45, ha='right', fontsize=FONT_TICK_SLIDE)
        ax.set_xlabel('Tumor Size (Pixels)', fontsize=FONT_LABEL_SLIDE, fontweight='bold', labelpad=15)
        ax.set_ylabel(f'Mean {metric_name}', fontsize=FONT_LABEL_SLIDE, fontweight='bold')
        ax.set_title(f'Mix Model Performance Profile: {metric_name}', fontsize=FONT_TITLE_SLIDE, fontweight='bold', pad=30)
        
        # 🎓 凡例を枠外右上に配置
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=FONT_LEGEND_SLIDE + 2, frameon=True, shadow=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"3_Mix_Model_Focus_{metric_name}slide.png"), dpi=300, bbox_inches='tight')
        plt.close()

print(f"\nプレゼン用高解像度グラフ（凡例拡大版）の生成が完了しました！")
print(f"保存先: {output_dir}")

# ========================================================================
# 追記部分：論文原稿（Thesis）向け Mixモデル単体専用プロット
# ========================================================================
print("\n--- 卒論原稿（Thesis）用：Mixモデル単体グラフの生成を開始します ---")
output_dir = "/workspace/output/combined_evaluation"

# 論文本文用フォントサイズ・描画スタイル設定
FONT_TITLE_THESIS = 13     # グラフタイトル
FONT_LABEL_THESIS = 11     # 軸ラベル (X軸/Y軸)
FONT_TICK_THESIS = 9.5     # 目盛り文字
FONT_LEGEND_THESIS = 10    # 凡例

LINE_WIDTH_THESIS = 2.0
MARKER_SIZE_THESIS = 7.5

for metric_name in metrics_list:
    if "mix" in all_weight_results:
        fig, ax = plt.subplots(figsize=(12, 6.5))
        
        raw_mix = all_weight_results["mix"][metric_name]["combined_mean"]
        final_mix = [raw_mix[j] for j in valid_bins_indices]
        
        # Mixモデルの描画
        ax.plot(
            x_indices, final_mix, 
            label="Baseline: Mix Model", color="#2196F3", 
            marker='^', linestyle='-', 
            linewidth=LINE_WIDTH_THESIS, 
            markersize=MARKER_SIZE_THESIS, 
            alpha=1.0
        )
        
        # 閾値（Split）線
        if bin3_final_idx is not None:
            ax.axvline(
                bin3_final_idx, color='#D32F2F', linestyle=':', 
                linewidth=1.5, label='Threshold Split'
            )

        # 軸・タイトル・凡例の設定（論文本文サイズに最適化）
        ax.set_xticks(x_indices)
        ax.set_xticklabels(final_bin_labels, rotation=45, ha='right', fontsize=FONT_TICK_THESIS)
        ax.set_xlabel('Tumor Size (Pixels)', fontsize=FONT_LABEL_THESIS, fontweight='bold', labelpad=10)
        ax.set_ylabel(f'Mean {metric_name}', fontsize=FONT_LABEL_THESIS, fontweight='bold')
        ax.set_title(f'Mix Model Performance Profile: {metric_name}', fontsize=FONT_TITLE_THESIS, fontweight='bold', pad=15)
        
        ax.legend(loc='upper right', fontsize=FONT_LEGEND_THESIS, frameon=True, shadow=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        
        # 論文用 Mix単体グラフの保存
        save_path_thesis = os.path.join(output_dir, f"3_Mix_Model_Focus_{metric_name}_thesis.eps")
        plt.savefig(save_path_thesis, dpi=300)
        plt.close()

print(f"\n論文用のMixモデル単体グラフの生成が完了しました！")
print(f"出力完了先: {output_dir}")