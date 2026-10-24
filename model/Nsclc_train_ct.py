import os
import sys
import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
 
from TransAttUnet import UNet_Attention_Transformer_Multiscale

# =========================================================
# 設定
# =========================================================
IMAGES_DIR = '/workspace/NSCLC_NIfTI2/imagesTr'
LABELS_DIR = '/workspace/NSCLC_NIfTI2/labelsTr/Neoplasm_Primary'
 
# 保存先: SAVE_ROOT/<ct_mode>/ に保存される
SAVE_ROOT = 'checkpoints_ct'
 
# ---- 学習するCT条件 ----
# 1つだけ:       ['mediastinal']
# 3つまとめて:   ['lung', 'mediastinal', 'raw']
# (コマンドラインで指定した場合はそちらが優先: python nsclc_train_ct.py lung)
CT_MODES = ['lung', 'mediastinal', 'raw']
 
# CT条件の定義
#   window: ウィンドウ(level, width)の範囲外は切り捨てて 0-1 に正規化
#   fixed : クリップなし。全患者共通の固定値で 0-1 に線形正規化 (範囲外は0-1からはみ出る)
CT_SETTINGS = {
    'lung':        {'type': 'window', 'level': -600, 'width': 1500},   # level -550~-700, width 1000~1500
    'mediastinal': {'type': 'window', 'level': 40,   'width': 350},    # level 20~60,     width 300~400
    'raw':         {'type': 'fixed',  'hu_min': -1024, 'hu_max': 3071},
}
 
TARGET_SIZE = (512, 512)
BATCH_SIZE = 8
ACCUMULATION_STEPS = 3
NUM_EPOCHS = 250
LR = 1e-4
 
# 学習の初期値として読み込むpth。キーはCT条件。無ければ最初から学習
# 例: INIT_WEIGHTS = {'lung': 'checkpoints_ct/lung/model_epoch_250.pth'}
INIT_WEIGHTS = {}
 
# INIT_WEIGHTSが何エポック目まで学習済みか (最初から学習するなら0)
START_EPOCH = 0
 
# pthから再開するときの保存先フォルダ名の後ろにつける文字 (元のpthを上書きしないため)
SAVE_SUFFIX = '_finetune'
 
 
# =========================================================
# CT値の変換
# =========================================================
def apply_ct_range(img_slice, ct_mode):
    s = CT_SETTINGS[ct_mode]
    if s['type'] == 'window':
        lo = s['level'] - s['width'] / 2
        hi = s['level'] + s['width'] / 2
        return (np.clip(img_slice, lo, hi) - lo) / (hi - lo)
    return (img_slice - s['hu_min']) / (s['hu_max'] - s['hu_min'])
 
 
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
    ct_mode   : 'lung' / 'mediastinal' / 'raw'  (CT_SETTINGSのキー)
    """
 
    def __init__(self, pairs, ct_mode, mode='all', target_size=(512, 512)):
        assert mode in ('all', 'max')
        assert ct_mode in CT_SETTINGS, f"ct_modeが不正です: {ct_mode}"
        self.ct_mode = ct_mode
        self.images = []
        self.masks = []
        self.meta = []
        self.patient_max_sizes = []
 
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
 
            z_ratio = n_slices_img / n_slices_mask
 
            for mask_idx in use_indices:
                img_idx = min(int(mask_idx * z_ratio), n_slices_img - 1)
 
                img_slice = apply_ct_range(img_vol[:, :, img_idx], ct_mode)
                mask_slice = mask_vol[:, :, mask_idx].astype(np.float32)
 
                img_t = torch.from_numpy(img_slice.astype(np.float32))[None, None]
                mask_t = torch.from_numpy(mask_slice).float()[None, None]
                img_t = F.interpolate(img_t, size=target_size, mode='bilinear', align_corners=False)
                mask_t = F.interpolate(mask_t, size=target_size, mode='nearest')
 
                self.images.append(img_t.squeeze(0).half())
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
# 1つのCT条件ぶんの学習
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


def train_one_mode(ct_mode, train_ds, test_ds, device):
    init_weights = INIT_WEIGHTS.get(ct_mode)

    save_dir = os.path.join(SAVE_ROOT, ct_mode + (SAVE_SUFFIX if init_weights else ''))
    os.makedirs(save_dir, exist_ok=True)

    drop_last = (len(train_ds) % BATCH_SIZE == 1)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=drop_last)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # CT条件ごとに新しいモデル・optimizerを作る(重みは共有しない)
    model = UNet_Attention_Transformer_Multiscale(n_channels=1, n_classes=1).to(device)

    if init_weights:
        state = torch.load(init_weights, map_location=device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
        model.load_state_dict(state)
        print(f"[{ct_mode}] 重みを読み込みました: {init_weights}")

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    start_epoch = START_EPOCH if init_weights else 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - start_epoch, eta_min=1e-6)
    criterion = MixLoss()

    train_hist, val_hist = [], []
    best_val = float('inf')
 
    print(f"\n===== [{ct_mode}] 学習開始: train {len(train_ds)}枚 / test {len(test_ds)}枚 =====")

    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
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
 
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks, _, _ in test_loader:
                    images, masks = images.to(device), masks.to(device)
                    val_loss += criterion(model(images), masks).item()
            avg_val_loss = val_loss / len(test_loader) if len(test_loader) > 0 else float('nan')
 
            scheduler.step()
 
            train_hist.append(avg_train_loss)
            val_hist.append(avg_val_loss)
            print(f"[{ct_mode}] Epoch [{epoch+1}/{NUM_EPOCHS}] "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
 
            if (epoch + 1) % 10 == 0:
                path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), path)
                print(f"--- モデルを保存しました: {path}")

            if avg_val_loss < best_val:
                best_val = avg_val_loss
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

            save_loss_plot(train_hist, val_hist,
                           os.path.join(save_dir, "loss_history.png"),
                           f"Training and Validation Loss ({ct_mode})",
                           start_epoch=start_epoch)

    except KeyboardInterrupt:
        print(f"\n[中断] [{ct_mode}] の学習が途中で停止されました。")
        raise
    finally:
        if len(train_hist) > 0:
            torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))
            save_loss_plot(train_hist, val_hist,
                           os.path.join(save_dir, "loss_history.png"),
                           f"Training and Validation Loss ({ct_mode})",
                           start_epoch=start_epoch)


# =========================================================
# 腫瘍サイズ分布の確認用 (学習の分け方には使わない)
# =========================================================
def plot_distribution(train_ds, test_ds):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
 
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
 
    ax = axes[1]
    train_slice_sizes = [m['tumor_size'] for m in train_ds.meta]
    ax.hist(train_slice_sizes, bins=30, color='seagreen', alpha=0.7,
            label=f'Train slices (n={len(train_slice_sizes)})', edgecolor='black')
    ax.set_title('Tumor Size per Slice (All Train Slices)')
    ax.set_xlabel('Tumor Size (Pixels on Slice)')
    ax.set_ylabel('Number of Slices (log)')
 
    for ax in axes:
        ax.set_yscale('log')
        ax.legend()
        ax.grid(axis='y', which='both', alpha=0.3)
 
    plt.tight_layout()
    path = os.path.join(SAVE_ROOT, 'hist.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"分布図を保存しました: {path}")
 
 
# =========================================================
# main
# =========================================================
def parse_ct_modes():
    """コマンドライン引数があればそれを優先。'all' なら3つ全部。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('ct', nargs='*',
                        help="lung / mediastinal / raw のどれか(複数可)。'all'で3つ全部。省略時は設定のCT_MODES")
    args = parser.parse_args()
 
    if not args.ct:
        modes = list(CT_MODES)
    elif 'all' in args.ct:
        modes = list(CT_SETTINGS.keys())
    else:
        modes = args.ct
 
    for m in modes:
        if m not in CT_SETTINGS:
            sys.exit(f"不明なCT条件: {m} (使えるのは {list(CT_SETTINGS.keys())} と all)")
    return modes
 
 
def main():
    ct_modes = parse_ct_modes()
    print(f"学習するCT条件: {ct_modes}")
 
    os.makedirs(SAVE_ROOT, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    all_images = sorted(f for f in os.listdir(IMAGES_DIR) if f.endswith('.nii.gz'))
    valid_pairs = []
    for f in all_images:
        lbl_path = os.path.join(LABELS_DIR, f)
        if os.path.exists(lbl_path):
            valid_pairs.append({'image': os.path.join(IMAGES_DIR, f), 'label': lbl_path, 'name': f})
    print(f"有効なペア数: {len(valid_pairs)}")
 
    # train 8 : test 2 (患者単位で分割。seedは固定なのでCT条件が違っても同じ分割)
    train_pairs, test_pairs = train_test_split(valid_pairs, test_size=0.2, random_state=42)
    print(f"学習用ペア数: {len(train_pairs)} / テスト用ペア数: {len(test_pairs)}")
 
    for n, ct_mode in enumerate(ct_modes):
        print(f"\n########## CT条件: {ct_mode} {CT_SETTINGS[ct_mode]} ##########")
 
        # メモリ節約のため、CT条件ごとにDatasetを作って、学習が終わったら捨てる
        print("Dataset作成中...")
        train_ds = NSCLCSliceDataset(train_pairs, ct_mode, mode='all', target_size=TARGET_SIZE)
        test_ds = NSCLCSliceDataset(test_pairs, ct_mode, mode='max', target_size=TARGET_SIZE)
        print(f"学習用スライス数: {len(train_ds)} / テスト用スライス数: {len(test_ds)}")
 
        if n == 0:
            plot_distribution(train_ds, test_ds)
 
        if len(train_ds) == 0:
            print(f"[{ct_mode}] 学習データが0枚なのでスキップします")
            continue
 
        train_one_mode(ct_mode, train_ds, test_ds, device)
 
        del train_ds, test_ds
        if device.type == 'cuda':
            torch.cuda.empty_cache()
 
    print("すべての処理が終了しました。")
 
 
if __name__ == '__main__':
    main()
 