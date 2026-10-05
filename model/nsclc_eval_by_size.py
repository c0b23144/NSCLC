import os
import csv
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
from nsclc_train_by_size import (
    NSCLCSliceDataset, fix_slice_axis, IMAGES_DIR, LABELS_DIR, TARGET_SIZE, GROUPS
)

# =========================================================
# 設定
# =========================================================
# 評価する3つのpth (自分のパスに合わせて書き換える)
WEIGHTS = {
    '35-919px':     'checkpoints_by_size/35-919px/model_epoch_250.pth',
    '919-1804px':   'checkpoints_by_size/919-1804px/model_epoch_250.pth',
    '1804-10652px': 'checkpoints_by_size/1804-10652px_finetune/model_epoch_250.pth',
}

OUTPUT_DIR = 'output/size_metrics'
N_BINS = 30  # ヒストグラムのビン数 (学習スクリプトのhist.pngと同じ)
METRICS = ['DICE', 'IoU', 'ACC', 'REC', 'PRE']

STYLES = {
    '35-919px':     {'color': '#8BC34A', 'marker': 'o', 'linestyle': '-'},
    '919-1804px':   {'color': '#2196F3', 'marker': '^', 'linestyle': '-'},
    '1804-10652px': {'color': '#E65100', 'marker': 's', 'linestyle': '-'},
}


# =========================================================
# ビン (ヒストグラムと同じ: 全患者の最大スライス面積の最小〜最大を30分割)
# =========================================================
def patient_max_area(label_path):
    mask = fix_slice_axis(nib.load(label_path).get_fdata(dtype=np.float32)) > 0
    return int(mask.sum(axis=(0, 1)).max())


def make_bin_edges(pairs, n_bins):
    sizes = [patient_max_area(p['label']) for p in pairs]
    sizes = [s for s in sizes if s > 0]  # 腫瘍がない患者はヒストグラムにも入らない
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
# グラフ
# =========================================================
def plot_metric(metric, results, bin_edges, counts, valid_bins, out_dir):
    x = np.arange(len(valid_bins))
    labels = [f"{int(bin_edges[j])}-{int(bin_edges[j + 1])}\n(n={counts[j]})" for j in valid_bins]

    plt.figure(figsize=(15, 7.5))
    for name, res in results.items():
        y = [res[metric][j] for j in valid_bins]
        plt.plot(x, y, label=name, linewidth=2.5, markersize=7, **STYLES[name])

    # 群の境界 (919px, 1804px) を、ビンの中の位置に合わせて点線で入れる
    pos_of_bin = {j: k for k, j in enumerate(valid_bins)}
    for t in [g[1] for g in GROUPS[1:]]:
        j = int(assign_bins(np.array([t]), bin_edges)[0])
        if j in pos_of_bin:
            frac = (t - bin_edges[j]) / (bin_edges[j + 1] - bin_edges[j])
            plt.axvline(pos_of_bin[j] - 0.5 + frac, color='#D32F2F', linestyle=':',
                        linewidth=2, label=f'Group boundary ({int(t)} px)')

    plt.xticks(x, labels, rotation=45, ha='right', fontsize=9.5)
    plt.xlabel('Tumor Size Ranges (Pixels on Max Slice)', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel(f'Mean {metric} Score', fontsize=12, fontweight='bold')
    plt.title(f'Size-specialist Models: {metric} by Tumor Size ({len(valid_bins)} Bins)',
              fontsize=14, fontweight='bold', pad=15)
    plt.xlim(-0.5, len(valid_bins) - 0.5)
    plt.ylim(0.9, 1.01) if metric == 'ACC' else plt.ylim(-0.05, 1.05)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f'size_models_{metric}.png'), dpi=200)
    plt.savefig(os.path.join(out_dir, f'size_models_{metric}.eps'))
    plt.close()


def save_csv(results, bin_edges, counts, valid_bins, out_dir):
    with open(os.path.join(out_dir, 'size_models_metrics.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'bin_low', 'bin_high', 'n_test'] + list(results.keys()))
        for metric in METRICS:
            for j in valid_bins:
                writer.writerow([metric, f"{bin_edges[j]:.1f}", f"{bin_edges[j + 1]:.1f}", counts[j]] +
                                [f"{res[metric][j]:.4f}" for res in results.values()])


# =========================================================
# main
# =========================================================
def main():
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

    # ---- ビンはヒストグラムと同じ (train+test全患者の最大スライス面積) ----
    bin_edges = make_bin_edges(valid_pairs, N_BINS)
    print(f"ビン範囲: {bin_edges[0]:.0f} - {bin_edges[-1]:.0f} px ({N_BINS}分割)")

    test_ds = NSCLCSliceDataset(test_pairs, mode='max', target_size=TARGET_SIZE)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    test_sizes = np.array([m['tumor_size'] for m in test_ds.meta])
    bin_idx = assign_bins(test_sizes, bin_edges)
    counts = np.bincount(bin_idx, minlength=N_BINS)

    # テストデータが1枚もないビンは飛ばす
    valid_bins = [j for j in range(N_BINS) if counts[j] > 0]
    print(f"テストデータがあるビン: {len(valid_bins)} / {N_BINS}")

    # ---- 3つのモデルをそれぞれ全テストデータで評価 ----
    results = {}
    for name, path in WEIGHTS.items():
        if not os.path.exists(path):
            print(f"[警告] 重みファイルが見つかりません: {path} (スキップ)")
            continue
        print(f"評価中: {name}  ({path})")
        model = UNet_Attention_Transformer_Multiscale(n_channels=1, n_classes=1).to(device)
        load_weights(model, path, device)
        scores = evaluate_model(model, test_loader, device)
        results[name] = {m: mean_per_bin(scores[m], bin_idx, N_BINS) for m in METRICS}
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if not results:
        print("評価できるモデルがありませんでした。WEIGHTSのパスを確認してください。")
        return

    # ---- グラフとCSV ----
    for metric in METRICS:
        plot_metric(metric, results, bin_edges, counts, valid_bins, OUTPUT_DIR)
    save_csv(results, bin_edges, counts, valid_bins, OUTPUT_DIR)
    print(f"保存しました: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
