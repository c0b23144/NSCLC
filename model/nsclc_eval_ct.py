import os
import sys
import csv
import argparse
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from TransAttUnet import UNet_Attention_Transformer_Multiscale
# データ読み込み部分は学習スクリプトと同じものを使う (同じフォルダに置くこと)
# ファイル名の大文字小文字に注意 (GitHub上は Nsclc_train_ct.py)
from Nsclc_train_ct import (
    NSCLCSliceDataset, fix_slice_axis, IMAGES_DIR, LABELS_DIR, TARGET_SIZE, CT_SETTINGS
)

# =========================================================
# 設定
# =========================================================
# 評価するCT条件 (コマンドラインで指定した場合はそちらが優先)
EVAL_CT_MODES = ['lung', 'mediastinal', 'raw']

# 評価する重み (自分のパスに合わせて書き換える)
# 学習時と同じCT条件で作ったpthを指定すること
WEIGHTS = {
    'lung':        'checkpoints_ct/lung/model_epoch_250.pth',
    'mediastinal': 'checkpoints_ct/mediastinal/model_epoch_250.pth',
    'raw':         'checkpoints_ct/raw/model_epoch_250.pth',
}

OUTPUT_DIR = 'output/ct_metrics'
N_BINS = 30  # 腫瘍サイズのビン数
METRICS = ['DICE', 'IoU', 'ACC', 'REC', 'PRE']

STYLES = {
    'lung':        {'color': '#2196F3', 'marker': 'o', 'linestyle': '-'},
    'mediastinal': {'color': '#E65100', 'marker': '^', 'linestyle': '-'},
    'raw':         {'color': '#4CAF50', 'marker': 's', 'linestyle': '-'},
}

DISPLAY_NAMES = {
    'lung': 'Lung window',
    'mediastinal': 'Mediastinal window',
    'raw': 'Raw HU (all values)',
}


# =========================================================
# ビン (全患者の最大スライス面積の最小〜最大を N_BINS 分割)
# =========================================================
def patient_max_area(label_path):
    mask = fix_slice_axis(nib.load(label_path).get_fdata(dtype=np.float32)) > 0
    return int(mask.sum(axis=(0, 1)).max())


def make_bin_edges(pairs, n_bins):
    sizes = [patient_max_area(p['label']) for p in pairs]
    sizes = [s for s in sizes if s > 0]
    return np.linspace(min(sizes), max(sizes), n_bins + 1)


def assign_bins(sizes, bin_edges):
    """各サイズが何番目のビンに入るか。[low, high) で、最後のビンだけ上端を含む"""
    idx = np.searchsorted(bin_edges, sizes, side='right') - 1
    return np.clip(idx, 0, len(bin_edges) - 2)


# =========================================================
# 評価
# =========================================================
def compute_metrics(pred, target):
    """pred, target: bool配列(1枚ぶん)"""
    tp = np.sum(pred & target)
    fp = np.sum(pred & ~target)
    fn = np.sum(~pred & target)
    tn = np.sum(~pred & ~target)

    both_empty = (tp + fp + fn) == 0
    return {
        'DICE': 1.0 if both_empty else 2 * tp / (2 * tp + fp + fn),
        'IoU':  1.0 if both_empty else tp / (tp + fp + fn),
        'ACC':  (tp + tn) / (tp + fp + fn + tn),
        'REC':  tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'PRE':  tp / (tp + fp) if (tp + fp) > 0 else 0.0,
    }


def load_weights(model, path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state)


def evaluate_model(model, loader, device):
    """テスト全スライスのスライスごとの指標を返す"""
    scores = {m: [] for m in METRICS}
    model.eval()
    with torch.no_grad():
        for images, masks, _, _ in loader:
            outputs = model(images.to(device))
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(bool)
            targets = masks.numpy() > 0.5
            for p, t in zip(preds, targets):
                result = compute_metrics(p.flatten(), t.flatten())
                for m in METRICS:
                    scores[m].append(result[m])
    return {m: np.array(v) for m, v in scores.items()}


def mean_per_bin(values, bin_idx, n_bins):
    """ビンごとの平均。データがないビンはNaN"""
    out = np.full(n_bins, np.nan)
    for j in range(n_bins):
        in_bin = bin_idx == j
        if in_bin.any():
            out[j] = values[in_bin].mean()
    return out


# =========================================================
# グラフ・CSV
# =========================================================
def plot_metric(metric, results, bin_edges, counts, valid_bins, out_dir):
    x = np.arange(len(valid_bins))
    labels = [f"{int(bin_edges[j])}-{int(bin_edges[j + 1])}\n(n={counts[j]})" for j in valid_bins]

    plt.figure(figsize=(15, 7.5))
    for name, res in results.items():
        y = [res['per_bin'][metric][j] for j in valid_bins]
        plt.plot(x, y, label=DISPLAY_NAMES[name], linewidth=2.5, markersize=7, **STYLES[name])

    plt.xticks(x, labels, rotation=45, ha='right', fontsize=9.5)
    plt.xlabel('Tumor Size Ranges (Pixels on Max Slice)', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel(f'Mean {metric} Score', fontsize=12, fontweight='bold')
    plt.title(f'CT-condition Models: {metric} by Tumor Size ({len(valid_bins)} Bins)',
              fontsize=14, fontweight='bold', pad=15)
    plt.xlim(-0.5, len(valid_bins) - 0.5)
    plt.ylim(0.9, 1.01) if metric == 'ACC' else plt.ylim(-0.05, 1.05)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f'ct_models_{metric}.png'), dpi=200)
    plt.savefig(os.path.join(out_dir, f'ct_models_{metric}.eps'))
    plt.close()


def save_csv_per_bin(results, bin_edges, counts, valid_bins, out_dir):
    with open(os.path.join(out_dir, 'ct_models_metrics_per_bin.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'bin_low', 'bin_high', 'n_test'] + list(results.keys()))
        for metric in METRICS:
            for j in valid_bins:
                writer.writerow([metric, f"{bin_edges[j]:.1f}", f"{bin_edges[j + 1]:.1f}", counts[j]] +
                                [f"{res['per_bin'][metric][j]:.4f}" for res in results.values()])


def save_csv_overall(results, n_test, out_dir):
    """サイズで分けない、テスト全体の平均"""
    with open(os.path.join(out_dir, 'ct_models_metrics_overall.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ct_mode', 'n_test'] + METRICS)
        for name, res in results.items():
            writer.writerow([name, n_test] + [f"{res['overall'][m]:.4f}" for m in METRICS])


# =========================================================
# main
# =========================================================
def parse_ct_modes():
    parser = argparse.ArgumentParser()
    parser.add_argument('ct', nargs='*',
                        help="lung / mediastinal / raw のどれか(複数可)。'all'で3つ全部。省略時は設定のEVAL_CT_MODES")
    args = parser.parse_args()

    if not args.ct:
        modes = list(EVAL_CT_MODES)
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
    print(f"評価するCT条件: {ct_modes}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 学習時と同じ分割でテストデータを作る ----
    valid_pairs = []
    for f in sorted(f for f in os.listdir(IMAGES_DIR) if f.endswith('.nii.gz')):
        lbl = os.path.join(LABELS_DIR, f)
        if os.path.exists(lbl):
            valid_pairs.append({'image': os.path.join(IMAGES_DIR, f), 'label': lbl, 'name': f})
    _, test_pairs = train_test_split(valid_pairs, test_size=0.2, random_state=42)
    print(f"テスト用ペア数: {len(test_pairs)}")

    # ---- ビン (train+test全患者の最大スライス面積) ----
    bin_edges = make_bin_edges(valid_pairs, N_BINS)
    print(f"ビン範囲: {bin_edges[0]:.0f} - {bin_edges[-1]:.0f} px ({N_BINS}分割)")

    results = {}
    counts = None
    n_test = 0

    for ct_mode in ct_modes:
        path = WEIGHTS.get(ct_mode)
        if path is None or not os.path.exists(path):
            print(f"[警告] [{ct_mode}] 重みファイルが見つかりません: {path} (スキップ)")
            continue

        # 学習時と同じCT条件でテストデータを作る (ここがずれるとIoUが0になる)
        print(f"\n評価中: {ct_mode} {CT_SETTINGS[ct_mode]}  ({path})")
        test_ds = NSCLCSliceDataset(test_pairs, ct_mode, mode='max', target_size=TARGET_SIZE)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

        test_sizes = np.array([m['tumor_size'] for m in test_ds.meta])
        bin_idx = assign_bins(test_sizes, bin_edges)
        if counts is None:
            counts = np.bincount(bin_idx, minlength=N_BINS)
            n_test = len(test_ds)

        model = UNet_Attention_Transformer_Multiscale(n_channels=1, n_classes=1).to(device)
        load_weights(model, path, device)
        scores = evaluate_model(model, test_loader, device)

        results[ct_mode] = {
            'per_bin': {m: mean_per_bin(scores[m], bin_idx, N_BINS) for m in METRICS},
            'overall': {m: float(scores[m].mean()) for m in METRICS},
        }
        print("  全体平均: " + ", ".join(f"{m}={results[ct_mode]['overall'][m]:.4f}" for m in METRICS))

        del model, test_ds, test_loader
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if not results:
        print("評価できるモデルがありませんでした。WEIGHTSのパスを確認してください。")
        return

    # テストデータが1枚もないビンは飛ばす
    valid_bins = [j for j in range(N_BINS) if counts[j] > 0]
    print(f"\nテストデータがあるビン: {len(valid_bins)} / {N_BINS}")

    # ---- グラフとCSV ----
    for metric in METRICS:
        plot_metric(metric, results, bin_edges, counts, valid_bins, OUTPUT_DIR)
    save_csv_per_bin(results, bin_edges, counts, valid_bins, OUTPUT_DIR)
    save_csv_overall(results, n_test, OUTPUT_DIR)
    print(f"保存しました: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
