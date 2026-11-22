import os
import csv
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from TransAttUnet import UNet_Attention_Transformer_Multiscale
from nsclc_train_by_size3_3ch import (
    NSCLCSliceDataset,
    fix_slice_axis,
    IMAGES_DIR,
    LABELS_DIR,
    TARGET_SIZE,
    GROUPS,
)

# =========================================================
# 設定
# =========================================================
WEIGHTS = {
    # まずは学習済みの2モデルだけ評価
    '35-919px': 'checkpoints_by_size_3ch/35-919px/best_model.pth',
    '919-1804px': 'checkpoints_by_size_3ch/919-1804px/best_model.pth',
    # 学習したら、ここを有効化して大サイズも評価可能
    '1804-10652px': 'checkpoints_by_size_3ch/1804-10652px/best_model.pth',
}

OUTPUT_DIR = 'output/size_metrics_3ch'
N_BINS = 30
METRICS = ['DICE', 'IoU', 'ACC', 'REC', 'PRE']

STYLES = {
    '35-919px': {
        'color': '#8BC34A',
        'marker': 'o',
        'linestyle': '-',
    },
    '919-1804px': {
        'color': '#2196F3',
        'marker': '^',
        'linestyle': '-',
    },
    '1804-10652px': {
        'color': '#E65100',
        'marker': 's',
        'linestyle': '-',
    },
}


# =========================================================
# ビン生成
# =========================================================
def patient_max_area(label_path):
    mask = fix_slice_axis(
        nib.load(label_path).get_fdata(dtype=np.float32)
    ) > 0
    return int(mask.sum(axis=(0, 1)).max())


def make_bin_edges(pairs, n_bins):
    sizes = [patient_max_area(p['label']) for p in pairs]
    sizes = [s for s in sizes if s > 0]

    if len(sizes) == 0:
        raise RuntimeError("腫瘍のある患者が存在しません")

    return np.linspace(min(sizes), max(sizes), n_bins + 1)


def assign_bins(sizes, bin_edges):
    """各サイズがどのビンに入るかを返す。[low, high)"""
    idx = np.searchsorted(bin_edges, sizes, side='right') - 1
    return np.clip(idx, 0, len(bin_edges) - 2)


# =========================================================
# 指標計算
# =========================================================
def compute_metrics(pred, target):
    """predとtargetはbool配列。スライスごとの指標を計算する。"""
    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.sum(pred & target)
    fp = np.sum(pred & ~target)
    fn = np.sum(~pred & target)
    tn = np.sum(~pred & ~target)

    both_empty = (tp + fp + fn) == 0

    return {
        'DICE': 1.0 if both_empty else 2 * tp / (2 * tp + fp + fn),
        'IoU': 1.0 if both_empty else tp / (tp + fp + fn),
        'ACC': (tp + tn) / (tp + fp + fn + tn),
        'REC': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'PRE': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
    }


def load_weights(model, path, device):
    state = torch.load(path, map_location=device)

    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']

    state = {
        k.replace('module.', '', 1): v
        for k, v in state.items()
    }
    model.load_state_dict(state)


def evaluate_model(model, loader, device):
    scores = {metric: [] for metric in METRICS}

    model.eval()
    with torch.no_grad():
        for images, masks, _, _ in loader:
            outputs = model(images.to(device))
            preds = (
                torch.sigmoid(outputs) > 0.5
            ).cpu().numpy().astype(bool)
            targets = masks.numpy() > 0.5

            for pred, target in zip(preds, targets):
                values = compute_metrics(
                    pred.flatten(),
                    target.flatten(),
                )

                for metric in METRICS:
                    scores[metric].append(values[metric])

    return {
        metric: np.array(values)
        for metric, values in scores.items()
    }


def mean_per_bin(values, bin_idx, n_bins):
    output = np.full(n_bins, np.nan)

    for bin_number in range(n_bins):
        mask = bin_idx == bin_number
        if mask.any():
            output[bin_number] = values[mask].mean()

    return output


# =========================================================
# グラフ
# =========================================================
def plot_metric(metric, results, bin_edges, counts, valid_bins, out_dir):
    x = np.arange(len(valid_bins))
    labels = [
        f"{int(bin_edges[j])}-{int(bin_edges[j + 1])}\n(n={counts[j]})"
        for j in valid_bins
    ]

    plt.figure(figsize=(15, 7.5))

    for name, result in results.items():
        y = [result[metric][j] for j in valid_bins]
        plt.plot(
            x,
            y,
            label=name,
            linewidth=2.5,
            markersize=7,
            **STYLES[name],
        )

    pos_of_bin = {
        original_bin: position
        for position, original_bin in enumerate(valid_bins)
    }

    for _, threshold, _ in GROUPS[1:]:
        if threshold == float('inf'):
            continue

        bin_number = int(
            assign_bins(
                np.array([threshold]),
                bin_edges,
            )[0]
        )

        if bin_number in pos_of_bin:
            fraction = (
                threshold - bin_edges[bin_number]
            ) / (
                bin_edges[bin_number + 1]
                - bin_edges[bin_number]
            )

            position = (
                pos_of_bin[bin_number]
                - 0.5
                + fraction
            )

            plt.axvline(
                position,
                color='#D32F2F',
                linestyle=':',
                linewidth=2,
                label=f'Group boundary ({int(threshold)} px)',
            )

    plt.xticks(
        x,
        labels,
        rotation=45,
        ha='right',
        fontsize=9.5,
    )
    plt.xlabel(
        'Tumor Size Ranges (Pixels on Max Slice)',
        fontsize=12,
        fontweight='bold',
        labelpad=10,
    )
    plt.ylabel(
        f'Mean {metric} Score',
        fontsize=12,
        fontweight='bold',
    )
    plt.title(
        f'Size-specialist Models: {metric} by Tumor Size '
        f'({len(valid_bins)} Bins)',
        fontsize=14,
        fontweight='bold',
        pad=15,
    )

    plt.xlim(-0.5, len(valid_bins) - 0.5)
    plt.ylim(0.9, 1.01) if metric == 'ACC' else plt.ylim(-0.05, 1.05)

    plt.legend(
        bbox_to_anchor=(1.01, 1),
        loc='upper left',
        fontsize=10,
        frameon=True,
        shadow=True,
    )
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, f'size_models_3ch_{metric}.png'),
        dpi=200,
    )
    plt.savefig(
        os.path.join(out_dir, f'size_models_3ch_{metric}.eps'),
    )
    plt.close()


def save_csv(results, bin_edges, counts, valid_bins, out_dir):
    csv_path = os.path.join(
        out_dir,
        'size_models_3ch_metrics.csv',
    )

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        writer.writerow(
            ['metric', 'bin_low', 'bin_high', 'n_test']
            + list(results.keys())
        )

        for metric in METRICS:
            for bin_number in valid_bins:
                row = [
                    metric,
                    f'{bin_edges[bin_number]:.1f}',
                    f'{bin_edges[bin_number + 1]:.1f}',
                    counts[bin_number],
                ]

                row += [
                    f'{result[metric][bin_number]:.4f}'
                    for result in results.values()
                ]

                writer.writerow(row)

    return csv_path


def print_group_metrics(test_ds, test_pairs, test_sizes):
    print('\nテスト患者数:', len(test_pairs))
    print('テスト最大スライス数:', len(test_ds))

    for group_name, lower, upper in GROUPS:
        selected = (
            (test_sizes >= lower)
            & (test_sizes < upper)
        )
        print(
            f'[{group_name}] '
            f'test={selected.sum()}'
        )


# =========================================================
# main
# =========================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print('device:', device)

    # 学習スクリプトと同じ分割
    valid_pairs = []

    for filename in sorted(
        f for f in os.listdir(IMAGES_DIR)
        if f.endswith('.nii.gz')
    ):
        label_path = os.path.join(LABELS_DIR, filename)

        if os.path.exists(label_path):
            valid_pairs.append({
                'image': os.path.join(IMAGES_DIR, filename),
                'label': label_path,
                'name': filename,
            })

    _, test_pairs = train_test_split(
        valid_pairs,
        test_size=0.2,
        random_state=42,
    )

    # 全患者の最大スライス面積からビンを作る
    bin_edges = make_bin_edges(valid_pairs, N_BINS)
    print(
        f'ビン範囲: '
        f'{bin_edges[0]:.0f}-{bin_edges[-1]:.0f} px '
        f'({N_BINS}分割)'
    )

    test_ds = NSCLCSliceDataset(
        test_pairs,
        mode='max',
        target_size=TARGET_SIZE,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=8,
        shuffle=False,
    )

    test_sizes = np.array([
        metadata['tumor_size']
        for metadata in test_ds.meta
    ])

    bin_idx = assign_bins(test_sizes, bin_edges)
    counts = np.bincount(
        bin_idx,
        minlength=N_BINS,
    )

    valid_bins = [
        bin_number
        for bin_number in range(N_BINS)
        if counts[bin_number] > 0
    ]

    print(
        f'テストデータがあるビン: '
        f'{len(valid_bins)} / {N_BINS}'
    )

    print_group_metrics(test_ds, test_pairs, test_sizes)

    # モデルごとに全テスト患者を評価
    results = {}

    for name, path in WEIGHTS.items():
        if not os.path.exists(path):
            print(
                f'[警告] 重みファイルが見つかりません: '
                f'{path} (スキップ)'
            )
            continue

        print(f'評価中: {name} ({path})')

        model = UNet_Attention_Transformer_Multiscale(
            n_channels=3,
            n_classes=1,
        ).to(device)

        load_weights(model, path, device)

        scores = evaluate_model(
            model,
            test_loader,
            device,
        )

        results[name] = {
            metric: mean_per_bin(
                scores[metric],
                bin_idx,
                N_BINS,
            )
            for metric in METRICS
        }

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if not results:
        print(
            '評価できるモデルがありませんでした。'
            'WEIGHTS のパスを確認してください。'
        )
        return

    for metric in METRICS:
        plot_metric(
            metric,
            results,
            bin_edges,
            counts,
            valid_bins,
            OUTPUT_DIR,
        )

    csv_path = save_csv(
        results,
        bin_edges,
        counts,
        valid_bins,
        OUTPUT_DIR,
    )

    print(f'保存しました: {csv_path}')


if __name__ == '__main__':
    main()