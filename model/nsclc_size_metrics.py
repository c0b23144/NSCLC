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

epoch_num = 20         # 評価したいエポックの数値
metrics_list = ["DICE", "IoU", "ACC", "REC", "PRE"]

print(f"\n[Epoch {epoch_num}] の重みをロードして、大・小別 × 各指標の評価グラフを生成")

# 1つの図には1つの重み。重み（モデル）ごとにループを回す
for model_weight_type in ["large", "small"]:
    print(f"[Epoch {epoch_num}] 重みロード: 【 {model_weight_type.upper()} WEIGHT MODEL 】")

    if model_weight_type == "large":
        epoch_weight_path = f'/workspace/checkpoints_large/model_epoch_{epoch_num}.pth'
        #epoch_weight_path = f'/workspace/TransAttUnet/model/checkpoints/model_epoch_50.pth'
    else:
        epoch_weight_path = f'/workspace/checkpoints_small/model_epoch_{epoch_num}.pth'
        #epoch_weight_path = f'/workspace/TransAttUnet/model/checkpoints/model_epoch_50.pth'

    if not os.path.exists(epoch_weight_path):
        raise FileNotFoundError(f"重みファイルが見つかりません: {epoch_weight_path}")
    
    model.load_state_dict(torch.load(epoch_weight_path, map_location=device))
    model.eval()

    test_collected_sizes = []
    test_collected_scores = {m: [] for m in metrics_list}

    # smallとlargeのテストデータローダー両方から順番にデータを集めて合体させる
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
                    
                    score_dict = {"DICE": d_score, "IoU": i_score, "ACC": a_score, "REC": r_score, "PRE": p_score}
                    test_collected_sizes.append(t_size)
                    for m in metrics_list:
                        test_collected_scores[m].append(score_dict[m])

    test_collected_sizes = np.array(test_collected_sizes)
    for m in metrics_list:
        test_collected_scores[m] = np.array(test_collected_scores[m])

    # 指標ごとにグラフを作成（この重みに対する1枚の図を作る）
    for metric_name in metrics_list:
        print(f"【{model_weight_type}重み】指標 [ {metric_name} ] のグラフを生成中...")
        
        bin_edges = actual_bins
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = np.diff(bin_edges)

        plt.figure(figsize=(14, 6))
        current_metric_scores = test_collected_scores[metric_name]

        # 30個のビンを1つずつ回してプロット
        for j in range(30):
            low, high = bin_edges[j], bin_edges[j+1]
            
            if j < 29:
                in_bin = (test_collected_sizes >= low) & (test_collected_sizes < high)
            else:
                in_bin = (test_collected_sizes >= low) & (test_collected_sizes <= high)
            
            scores_in_bin = current_metric_scores[in_bin]
            count = len(scores_in_bin)

            if count > 0:
                mean = np.mean(scores_in_bin)
                std_err = np.std(scores_in_bin) / np.sqrt(count) if count > 1 else 0.0
            else:
                mean, std_err = np.nan, np.nan

            # 【4色カラーロジック】モデルの種類 × データのサイズ で完全に色分け
            bin_center = (low + high) / 2
            
            if model_weight_type == "small":
                if bin_center < threshold:
                    bar_color = '#8BC34A'   # 黄緑 (Small Weight Model が Small Data を推論)
                    text_color = '#558B2F'
                    label_name = 'Small Weight - Small Data' if j == 0 else ""
                else:
                    bar_color = '#2E7D32'   # 深緑 (Small Weight Model が Large Data を推論)
                    text_color = '#1B5E20'
                    label_name = 'Small Weight - Large Data' if j == 15 else ""
            else:  # large_weight モデルの場合
                if bin_center < threshold:
                    bar_color = '#FFE082'   # 薄オレンジ (Large Weight Model が Small Data を推論)
                    text_color = '#FFB300'
                    label_name = 'Large Weight - Small Data' if j == 0 else ""
                else:
                    bar_color = '#E65100'   # 濃いオレンジ (Large Weight Model が Large Data を推論)
                    text_color = '#BF360C'
                    label_name = 'Large Weight - Large Data' if j == 15 else ""

            # 棒グラフを元の形式で隙間なく真っ直ぐ並べて描画
            bar = plt.bar(bin_centers[j], mean, yerr=std_err, width=bin_widths[j],
                            capsize=3, color=bar_color, edgecolor='black', alpha=0.8,
                            label=label_name, error_kw={'ecolor': 'black', 'lw': 1.0})

            # 各ビンの真上にサンプル数 (n=...) を表示
            if count > 0 and not np.isnan(mean):
                err_top_normal = mean + std_err if (std_err > 0 and not np.isnan(std_err)) else mean
                plt.text(bin_centers[j], err_top_normal + 0.03, f'n={count}',
                            ha='center', va='bottom', fontsize=8, color=text_color, rotation=90)
            

            # ========================================================================
            # このビンの中にSmallとLargeが「両方とも」実際に存在しているかチェック
                sizes_in_bin = test_collected_sizes[in_bin]
                scores_in_bin = current_metric_scores[in_bin]
                
                is_small = sizes_in_bin < threshold
                is_large = sizes_in_bin >= threshold

                # 3つ目のビンのように両方ある場合、既存の1本バーを「別々の2本」にアップデートする
                if np.sum(is_small) > 0 and np.sum(is_large) > 0:
                    
                    # 1. さっき描いてしまった「合算された1本のバー」を一旦画面から消去（クリア）する
                    if 'bar' in locals() and bar is not None:
                        for patch in bar:
                            patch.remove()
                    
                    # 2. 幅を半分にして、左右に綺麗に並べる位置を計算
                    width_sub = bin_widths[j] / 2
                    pos_small = bin_centers[j] - (width_sub / 2)
                    pos_large = bin_centers[j] + (width_sub / 2)

                    # 3. 【Small側】を別個に計算して描画
                    scores_s = scores_in_bin[is_small]
                    count_s = len(scores_s)
                    mean_s = np.mean(scores_s)
                    std_s = np.std(scores_s) / np.sqrt(count_s) if count_s > 1 else 0.0

                    color_s = '#8BC34A' if model_weight_type == "small" else '#FFE082'
                    txt_color_s = '#558B2F' if model_weight_type == "small" else '#FFB300'
                    label_s = f"{model_weight_type.capitalize()} Weight - Small Data"

                    bar_s = plt.bar(pos_small, mean_s, yerr=std_s, width=width_sub,
                                    capsize=2, color=color_s, edgecolor='black', alpha=0.8,
                                    error_kw={'ecolor': 'black', 'lw': 1.0})
                    if not any(lbl == label_s for lbl in plt.gca().get_legend_handles_labels()[1]):
                        bar_s.set_label(label_s)

                    # 4. 【Large側】を別個に計算して描画（これで消えていたLargeが別個に出現！）
                    scores_l = scores_in_bin[is_large]
                    count_l = len(scores_l)
                    mean_l = np.mean(scores_l)
                    std_l = np.std(scores_l) / np.sqrt(count_l) if count_l > 1 else 0.0

                    color_l = '#2E7D32' if model_weight_type == "small" else '#E65100'
                    txt_color_l = '#1B5E20' if model_weight_type == "small" else '#BF360C'
                    label_l = f"{model_weight_type.capitalize()} Weight - Large Data"

                    bar_l = plt.bar(pos_large, mean_l, yerr=std_l, width=width_sub,
                                    capsize=2, color=color_l, edgecolor='black', alpha=0.8,
                                    error_kw={'ecolor': 'black', 'lw': 1.0})
                    if not any(lbl == label_l for lbl in plt.gca().get_legend_handles_labels()[1]):
                        bar_l.set_label(label_l)

                    # 5. サンプル数（n=...）のテキスト位置も左右それぞれに修正する
                    # ※ 元の plt.text の上から、新しい位置に正しい n 数を上書き表示します
                    plt.text(pos_small, mean_s + 0.02, f'n={count_s}',
                             ha='center', va='bottom', fontsize=7, color=txt_color_s, rotation=90)
                    plt.text(pos_large, mean_l + 0.02, f'n={count_l}',
                             ha='center', va='bottom', fontsize=7, color=txt_color_l, rotation=90)
                # ========================================================================

        # 閾値線（中央値）の描画
        plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold ({int(threshold)} px)')
        
        # タイトルとレイアウトの装飾
        plt.title(f'Model: {model_weight_type.upper()} Weight (Epoch {epoch_num}) - {metric_name} Performance', fontsize=14)
        plt.xlabel('Tumor Size (Pixels on Max Slice)')
        plt.ylabel(f'Mean {metric_name} Score')
        plt.xlim(all_sizes.min(), all_sizes.max())
        plt.ylim(0, 1.15)
        plt.legend(loc='upper right')
        plt.grid(axis='y', alpha=0.3)

        # 重みごとにフォルダを分けて保存
        save_dir = f"/workspace/output/hist_bins/{model_weight_type}_weight/epoch_add{epoch_num}/"
        #save_dir = f"/workspace/output/hist_bins/pretrained/"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{model_weight_type}_weight_{metric_name}_epoch{epoch_num}.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    

    # ─── 必要な数値（ビンと平均値）だけをテキストファイルに出力する処理 ───
    output_text_path = f"/workspace/output/hist_bins/bin_metrics_summary_{model_weight_type}.txt"
    
    with open(output_text_path, "w", encoding="utf-8") as f:
        f.write(f"========================================================\n")
        f.write(f" モデルの重み: {model_weight_type.upper()} WEIGHT MODEL\n")
        f.write(f"========================================================\n\n")
        
        bin_edges = actual_bins
        
        for j in range(30):
            low, high = bin_edges[j], bin_edges[j+1]
            if j < 29:
                in_bin = (test_collected_sizes >= low) & (test_collected_sizes < high)
            else:
                in_bin = (test_collected_sizes >= low) & (test_collected_sizes <= high)
            
            sizes_in_bin = test_collected_sizes[in_bin]
            
            # 3つ目のビンのように「SmallとLargeが両方混ざっている」かを判定
            is_small = sizes_in_bin < threshold
            is_large = sizes_in_bin >= threshold
            
            # 出力対象の仕分け（混ざっていれば別々に、通常ならビン全体）
            splits = []
            if np.sum(is_small) > 0 and np.sum(is_large) > 0:
                splits = [("Small Data", is_small), ("Large Data", is_large)]
            else:
                splits = [("Total Data", np.ones_like(in_bin, dtype=bool)[in_bin])]
                
            for label, mask_sub in splits:
                sub_scores_dict = {}
                count_sub = np.sum(mask_sub)
                
                if count_sub > 0:
                    # 各指標の「平均値」だけを計算
                    for m in metrics_list:
                        sub_scores_dict[m] = np.mean(test_collected_scores[m][in_bin][mask_sub])
                    
                    # テキストファイルへの書き込みフォーマット
                    f.write(f"【Bin {j+1:02d}】 範囲: {int(low)}-{int(high)} px ({label}) [n={count_sub}]\n")
                    f.write(f"  -> 平均値: ")
                    metrics_str = ", ".join([f"{m}: {sub_scores_dict[m]:.4f}" for m in metrics_list])
                    f.write(metrics_str + "\n")
                else:
                    # データが1件もないビン
                    f.write(f"【Bin {j+1:02d}】 範囲: {int(low)}-{int(high)} px -> データなし (n=0)\n")
            f.write("\n")
            
    print(f"[{model_weight_type.upper()}] のビン別平均値テキストを保存しました: {output_text_path}")

print("\n4色カラーロジックを搭載した個別重みグラフ（計10枚）の保存がすべて完了しました！")
