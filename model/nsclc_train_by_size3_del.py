import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split

from TransAttUnet import UNet_Attention_Transformer_Multiscale

# =========================================================
# 設定
# =========================================================
IMAGES_DIR = '/workspace/NSCLC_NIfTI2/imagesTr'
LABELS_DIR = '/workspace/NSCLC_NIfTI2/labelsTr/Neoplasm_Primary'
SAVE_ROOT = 'checkpoints_by_size'

TARGET_SIZE = (512, 512)
BATCH_SIZE = 8
ACCUMULATION_STEPS = 3
NUM_EPOCHS = 250  # 全スライス使うとデータ量が増えるので、時間がかかる場合は減らす
LR = 1e-4

# (名前, 下限, 上限)  下限 <= 腫瘍面積 < 上限
# 一番小さい群の下限と一番大きい群の上限は開けておく(35px未満や10652px超が来ても漏れないように)
GROUPS = [
    ('35-919px',     0,    919),
    ('919-1804px',   919,  1804),
    ('1804-10652px', 1804, float('inf')),
]

# 学習する群 (ここに書いた群だけ学習する)
TRAIN_GROUPS = ['919-1804px']

# 学習の初期値として読み込むpth (Noneなら最初から学習)
# ※ 自分が使いたいpthのパスに書き換える
INIT_WEIGHTS = None

# INIT_WEIGHTSが何エポック目まで学習済みのものか (最初から学習するなら0)
# → START_EPOCH+1 エポック目から NUM_EPOCHS エポック目まで続きを学習する
START_EPOCH = 0

# pthから学習を再開するときの保存先フォルダ名の後ろにつける文字
# (元のpthを上書きしないため。例: checkpoints_by_size/1804-10652px_finetune/)
SAVE_SUFFIX = '_finetune2'

# 学習データをどの面積で群分けするか
#   'patient': 患者ごとの最大スライスの腫瘍面積で群分け (その患者の腫瘍スライスは全部同じモデルに入る)
#   'slice'  : スライスごとの腫瘍面積で群分け
GROUP_BY = 'patient'


# =========================================================
# Dataset
# =========================================================
def fix_slice_axis(data):
    """スライス軸(一番短い軸)を最後に持ってきて (H, W, Slices) にする"""
    min_axis = np.argmin(data.shape)
    if min_axis == 0:
        return data.transpose(1, 2, 0)
    elif min_axis == 1:
        return data.transpose(0, 2, 1)
    return data


class NSCLCSliceDataset(Dataset):
    """
    mode='all': 腫瘍が写っているスライスを全部使う (train用)
    mode='max': 腫瘍面積が最大のスライス1枚だけ使う (test用)

    毎回niftiを読み直すと遅いので、最初に前処理してメモリに載せる。
    (画像はfloat16, マスクはuint8で保持)
    """

    def __init__(self, pairs, mode='all', target_size=(512, 512)):
        assert mode in ('all', 'max')
        self.images = []
        self.masks = []
        self.meta = []  # スライスごとの情報
        self.patient_max_sizes = []  # 患者ごとの最大スライス腫瘍面積

        for pair in pairs:
            img_vol = fix_slice_axis(nib.load(pair['image']).get_fdata(dtype=np.float32))
            mask_vol = fix_slice_axis(nib.load(pair['label']).get_fdata(dtype=np.float32)) > 0

            n_slices_img = img_vol.shape[2]
            n_slices_mask = mask_vol.shape[2]

            slice_areas = mask_vol.sum(axis=(0, 1))
            tumor_indices = np.where(slice_areas > 0)[0]
            if len(tumor_indices) == 0:
                print(f"[警告] 腫瘍スライスなし: {pair['name']} (スキップ)")
                continue

            patient_max = int(slice_areas.max())
            self.patient_max_sizes.append(patient_max)

            if mode == 'max':
                use_indices = [int(np.argmax(slice_areas))]
            else:
                use_indices = [int(i) for i in tumor_indices]

            # 画像とマスクのスライス枚数が違う場合に備えて相対位置で対応づける
            z_ratio = n_slices_img / n_slices_mask

            for mask_idx in use_indices:
                img_idx = min(int(mask_idx * z_ratio), n_slices_img - 1)

                img_slice = np.clip(img_vol[:, :, img_idx], -1000, 400)
                img_slice = (img_slice + 1000) / 1400  # 0-1に正規化
                mask_slice = mask_vol[:, :, mask_idx].astype(np.float32)

                img_t = torch.from_numpy(img_slice).float()[None, None]    # [1,1,H,W]
                mask_t = torch.from_numpy(mask_slice).float()[None, None]
                img_t = F.interpolate(img_t, size=target_size, mode='bilinear', align_corners=False)
                mask_t = F.interpolate(mask_t, size=target_size, mode='nearest')

                self.images.append(img_t.squeeze(0).half())   # [1,H,W]
                self.masks.append(mask_t.squeeze(0).to(torch.uint8))
                self.meta.append({
                    'name': pair['name'] if mode == 'max' else f"{pair['name']}_slice{mask_idx}",
                    'tumor_size': int(slice_areas[mask_idx]),
                    'patient_max_size': patient_max,
                })

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        m = self.meta[idx]
        return self.images[idx].float(), self.masks[idx].float(), m['name'], m['tumor_size']


def select_indices(dataset, lo, hi, key):
    return [i for i, m in enumerate(dataset.meta) if lo <= m[key] < hi]


# =========================================================
# Loss (BCE + Dice)
# =========================================================
class MixLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, outputs, targets):
        targets = targets.float()
        probs = torch.sigmoid(outputs)
        probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)

        bce_loss = self.bce(probs, targets)

        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice_loss = 1.0 - ((2. * intersection + self.smooth) /
                           (probs_flat.sum() + targets_flat.sum() + self.smooth))

        return 0.5 * bce_loss + 0.5 * dice_loss


# =========================================================
# 1群ぶんの学習
# =========================================================
def save_loss_plot(train_hist, val_hist, path, title, start_epoch=0):
    epochs = range(start_epoch + 1, start_epoch + len(train_hist) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_hist, label='Train Loss')
    plt.plot(epochs, val_hist, label='Val Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()


def train_one_group(group_name, train_ds, test_ds, device):
    save_dir = os.path.join(SAVE_ROOT, group_name + (SAVE_SUFFIX if INIT_WEIGHTS else ''))
    os.makedirs(save_dir, exist_ok=True)

    # 最後のバッチが1枚だけになるとBatchNormで落ちることがあるので、その場合だけ捨てる
    drop_last = (len(train_ds) % BATCH_SIZE == 1)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=drop_last)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 群ごとに新しいモデル・optimizerを作る(重みは共有しない)
    model = UNet_Attention_Transformer_Multiscale(n_channels=1, n_classes=1).to(device)

    # pthがあれば重みを読み込んでそこから学習を始める
    if INIT_WEIGHTS:
        state = torch.load(INIT_WEIGHTS, map_location=device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        state = {k.replace('module.', '', 1): v for k, v in state.items()}  # DataParallel対策
        model.load_state_dict(state)
        print(f"[{group_name}] 重みを読み込みました: {INIT_WEIGHTS}")

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    start_epoch = START_EPOCH if INIT_WEIGHTS else 0
    # 追加
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS - start_epoch,  # 残りエポック数
    eta_min=1e-6
)
    criterion = MixLoss()

    train_hist, val_hist = [], []
    best_val = float('inf')

    print(f"\n===== [{group_name}] 学習開始: train {len(train_ds)}枚 / test {len(test_ds)}枚 =====")

    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            # ---- 学習 ----
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()

            for i, (images, masks, _, _) in enumerate(train_loader):
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                (loss / ACCUMULATION_STEPS).backward()

                if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # ---- 検証 ----
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks, _, _ in test_loader:
                    images, masks = images.to(device), masks.to(device)
                    val_loss += criterion(model(images), masks).item()
            avg_val_loss = val_loss / len(test_loader) if len(test_loader) > 0 else float('nan')

            # 追加
            scheduler.step() 

            train_hist.append(avg_train_loss)
            val_hist.append(avg_val_loss)
            print(f"[{group_name}] Epoch [{epoch+1}/{NUM_EPOCHS}] "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # ---- 保存 ----
            if (epoch + 1) % 10 == 0:
                path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), path)
                print(f"--- モデルを保存しました: {path}")

            if avg_val_loss < best_val:  # nanのときは比較がFalseになるので保存されない
                best_val = avg_val_loss
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

            save_loss_plot(train_hist, val_hist,
                           os.path.join(save_dir, "loss_history.png"),
                           f"Training and Validation Loss ({group_name})",
                           start_epoch=start_epoch)

    except KeyboardInterrupt:
        print(f"\n[中断] [{group_name}] の学習が途中で停止されました。")
        raise
    finally:
        if len(train_hist) > 0:
            torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))
            save_loss_plot(train_hist, val_hist,
                           os.path.join(save_dir, "loss_history.png"),
                           f"Training and Validation Loss ({group_name})",
                           start_epoch=start_epoch)


# =========================================================
# データ分布の可視化 (縦軸は対数)
# =========================================================
def plot_distribution(train_ds, test_ds):
    train_slice_sizes = [m['tumor_size'] for m in train_ds.meta]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 左: 患者ごとの最大スライス面積 (群分けの基準)
    ax = axes[0]
    all_max = train_ds.patient_max_sizes + test_ds.patient_max_sizes
    bins = np.linspace(min(all_max), max(all_max), 31)
    ax.hist(train_ds.patient_max_sizes, bins=bins, color='royalblue', alpha=0.6,
            label=f'Train (n={len(train_ds.patient_max_sizes)} patients)', edgecolor='black')
    ax.hist(test_ds.patient_max_sizes, bins=bins, color='orange', alpha=0.6,
            label=f'Test (n={len(test_ds.patient_max_sizes)} patients)', edgecolor='black')
    ax.set_title('Tumor Size per Patient (Max Slice)')
    ax.set_xlabel('Tumor Size (Pixels on Max Slice)')
    ax.set_ylabel('Number of Patients (log)')

    # 右: 学習に使う全スライスの面積
    ax = axes[1]
    ax.hist(train_slice_sizes, bins=30, color='seagreen', alpha=0.7,
            label=f'Train slices (n={len(train_slice_sizes)})', edgecolor='black')
    ax.set_title('Tumor Size per Slice (All Train Slices)')
    ax.set_xlabel('Tumor Size (Pixels on Slice)')
    ax.set_ylabel('Number of Slices (log)')

    for ax in axes:
        ax.set_yscale('log')
        for _, lo, _ in GROUPS[1:]:
            ax.axvline(lo, color='red', linestyle='dashed', linewidth=2)
        ax.legend()
        ax.grid(axis='y', which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, 'hist.png'), dpi=150)
    plt.close()
    print(f"分布図を保存しました: {os.path.join(SAVE_ROOT, 'hist.png')}")


# =========================================================
# main
# =========================================================
def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 画像とラベルのペア ----
    all_images = sorted(f for f in os.listdir(IMAGES_DIR) if f.endswith('.nii.gz'))
    valid_pairs = []
    for f in all_images:
        lbl_path = os.path.join(LABELS_DIR, f)
        if os.path.exists(lbl_path):
            valid_pairs.append({'image': os.path.join(IMAGES_DIR, f), 'label': lbl_path, 'name': f})
    print(f"有効なペア数: {len(valid_pairs)}")

    # train 8 : test 2 (患者単位で分割。元のスクリプトと同じseed)
    train_pairs, test_pairs = train_test_split(valid_pairs, test_size=0.2, random_state=42)
    print(f"学習用ペア数: {len(train_pairs)} / テスト用ペア数: {len(test_pairs)}")

    # ---- Dataset (trainは腫瘍スライス全部、testは最大スライス1枚) ----
    print("Dataset作成中...")
    train_ds = NSCLCSliceDataset(train_pairs, mode='all', target_size=TARGET_SIZE)
    test_ds = NSCLCSliceDataset(test_pairs, mode='max', target_size=TARGET_SIZE)
    print(f"学習用スライス数: {len(train_ds)} / テスト用スライス数: {len(test_ds)}")

    # ---- 腫瘍サイズ分布の確認 ----
    plot_distribution(train_ds, test_ds)

    # ---- 群ごとにデータを仕分けて、それぞれ別モデルを学習 ----
    train_key = 'patient_max_size' if GROUP_BY == 'patient' else 'tumor_size'
    test_key = 'patient_max_size'  # testは最大スライス1枚なので tumor_size と同じ

    group_datasets = []
    for name, lo, hi in GROUPS:
        tr_idx = select_indices(train_ds, lo, hi, train_key)
        te_idx = select_indices(test_ds, lo, hi, test_key)
        group_datasets.append((name, Subset(train_ds, tr_idx), Subset(test_ds, te_idx)))
        print(f"[{name}] train: {len(tr_idx)}枚 / test: {len(te_idx)}枚")

    for name, tr_subset, te_subset in group_datasets:
        if name not in TRAIN_GROUPS:
            print(f"[{name}] TRAIN_GROUPSに含まれないのでスキップします")
            continue
        if len(tr_subset) == 0:
            print(f"[{name}] 学習データが0枚なのでスキップします")
            continue
        train_one_group(name, tr_subset, te_subset, device)

    print("すべての処理が終了しました。")


if __name__ == '__main__':
    main()
