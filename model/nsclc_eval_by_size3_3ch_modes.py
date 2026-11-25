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
from nsclc_train_by_size3_3ch import (
    NSCLCSliceDataset,
    fix_slice_axis,
    IMAGES_DIR,
    LABELS_DIR,
    TARGET_SIZE,
)

# =========================================================
# 設定
# =========================================================
# "all", "two", "three" のいずれかを指定
EVAL_MODE = 'two'

# 各モードで使用する重み。
# 実際の保存先が異なる場合は、ここを変更してください。
WEIGHTS_BY_MODE = {
    'all': {
        'All': 'checkpoints_by_all_size_3ch/All/best_model.pth',
    },
    'two': {
        'Small': 'checkpoints_by_2size_3ch/Small/best_model.pth',
        'Large': 'checkpoints_by_2size_3ch/Large/best_model.pth',
    },
    'three': {
        'Small': 'checkpoints_by_size_3ch/35-919px/best_model.pth',
        'Middle': 'checkpoints_by_size_3ch/919-1804px/best_model.pth',
        'Large': 'checkpoints_by_size_3ch/1804-10652px/best_model.pth',
    },
}

# 群境界（腫瘍最大スライス面積、px）
GROUP_BOUNDARIES = {
    'all': [],
    'two': [1804],
    'three': [919, 1804],
}

OUTPUT_DIRS = {
    'all': 'output/size_metrics_3ch_modes/All',
    'two': 'output/size_metrics_3ch_modes/two',
    'three': 'output/size_metrics_3ch_modes/three',
}
N_BINS = 30
BATCH_SIZE = 8
METRICS = ['DICE', 'IoU', 'ACC', 'REC', 'PRE']

STYLES = {
    'All': {
        'color': '#2196F3',
        'marker': 'o',
        'linestyle': '-',
    },
    'Small': {
        'color': '#8BC34A',
        'marker': 'o',
        'linestyle': '-',
    },
    'Middle': {
        'color': '#2196F3',
        'marker': '^',
        'linestyle': '-',
    },
    'Large': {
        'color': '#E65100',
        'marker': 's',
        'linestyle': '-',
    },
}


# =========================================================
# サイズビン
# =========================================================
def patient_max_area(label_path):
    mask = fix_slice_axis(
        nib.load(label_path).get_fdata(dtype=np.float32)
    ) > 0
    return int(mask.sum(axis=(0, 1)).max())


def make_bin_edges(pairs, n_bins):
    sizes = [
        patient_max_area(pair['label'])
        for pair in pairs
    ]
    sizes = [size for size in sizes if size > 0]

    if not sizes:
        raise RuntimeError('腫瘍のある患者が存在しません')

    low = min(sizes)
    high = max(sizes)

    # 全患者の最大スライス面積が同じ場合にゼロ幅のbinを避ける
    if low == high:
        high = low + 1

    return np.linspace(low, high, n_bins + 1)


def assign_bins(sizes, bin_edges):
    """各サイズを [low, high) のbinに割り当てる。"""
    idx = np.searchsorted(bin_edges, sizes, side='right') - 1
    return np.clip(idx, 0, len(bin_edges) - 2)


# =========================================================
# 指標計算
# =========================================================
def compute_metrics(pred, target):
    """predとtargetはbool配列。1スライス分の指標を計算する。"""
    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.sum(pred & target)
    fp = np.sum(pred & ~target)
    fn = np.sum(~pred & target)
    tn = np.sum(~pred & ~target)

    both_empty = (tp + fp + fn) == 0

    return {
        'DICE': (
            1.0 if both_empty
            else 2 * tp / (2 * tp + fp + fn)
        ),
        'IoU': (
            1.0 if both_empty
            else tp / (tp + fp + fn)
        ),
        'ACC': (tp + tn) / (tp + fp + fn + tn),
        'REC': tp / (tp + fn) if tp + fn > 0 else 0.0,
        'PRE': tp / (tp + fp) if tp + fp > 0 else 0.0,
    }


def load_weights(model, path, device):
    state = torch.load(path, map_location=device)

    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']

    state = {
        key.replace('module.', '', 1): value
        for key, value in state.items()
    }
    model.load_state_dict(state)


def evaluate_model(model, loader, device):
    scores = {
        metric: []
        for metric in METRICS
    }

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
        metric: np.asarray(values)
        for metric, values in scores.items()
    }


def mean_per_bin(values, bin_idx, n_bins):
    means = np.full(n_bins, np.nan)

    for bin_number in range(n_bins):
        selected = bin_idx == bin_number
        if selected.any():
            means[bin_number] = values[selected].mean()

    return means


# =========================================================
# グラフ・CSV
# =========================================================
def plot_metric(
    mode,
    metric,
    results,
    bin_edges,
    counts,
    valid_bins,
    out_dir,
):
    x = np.arange(len(valid_bins))
    labels = [
        f'{int(bin_edges[j])}-{int(bin_edges[j + 1])}\n'
        f'(n={counts[j]})'
        for j in valid_bins
    ]

    plt.figure(figsize=(15, 7.5))

    # AllモードはAllモデルの1本、
    # twoモードはSmall/Largeの2本、
    # threeモードはSmall/Middle/Largeの3本を描画
    for model_name, model_results in results.items():
        y = [
            model_results[metric][bin_number]
            for bin_number in valid_bins
        ]

        plt.plot(
            x,
            y,
            label=model_name,
            linewidth=2.5,
            markersize=7,
            **STYLES[model_name],
        )

    # 群境界を、サイズビン軸上に赤い点線で表示
    position_by_bin = {
        bin_number: position
        for position, bin_number in enumerate(valid_bins)
    }

    for threshold in GROUP_BOUNDARIES[mode]:
        bin_number = int(
            assign_bins(
                np.array([threshold]),
                bin_edges,
            )[0]
        )

        if bin_number not in position_by_bin:
            continue

        bin_width = (
            bin_edges[bin_number + 1]
            - bin_edges[bin_number]
        )
        if bin_width <= 0:
            continue

        fraction = (
            threshold - bin_edges[bin_number]
        ) / bin_width

        boundary_x = (
            position_by_bin[bin_number] - 0.5 + fraction
        )

        plt.axvline(
            boundary_x,
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
        f'{mode.capitalize()} Mode: {metric} by Tumor Size '
        f'({len(valid_bins)} Bins)',
        fontsize=14,
        fontweight='bold',
        pad=15,
    )

    plt.xlim(-0.5, len(valid_bins) - 0.5)
    if metric == 'ACC':
        plt.ylim(0.9, 1.01)
    else:
        plt.ylim(-0.05, 1.05)

    plt.legend(
        bbox_to_anchor=(1.01, 1),
        loc='upper left',
        fontsize=10,
        frameon=True,
        shadow=True,
    )
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    png_path = os.path.join(
        out_dir,
        f'{mode}_mode_{metric}.png',
    )
    eps_path = os.path.join(
        out_dir,
        f'{mode}_mode_{metric}.eps',
    )

    plt.savefig(png_path, dpi=200)
    plt.savefig(eps_path)
    plt.close()


def save_csv(
    mode,
    results,
    bin_edges,
    counts,
    valid_bins,
    out_dir,
):
    csv_path = os.path.join(
        out_dir,
        f'{mode}_mode_metrics.csv',
    )

    with open(
        csv_path,
        'w',
        newline='',
        encoding='utf-8',
    ) as file:
        writer = csv.writer(file)

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
                    int(counts[bin_number]),
                ]

                row.extend(
                    f'{model_results[metric][bin_number]:.4f}'
                    for model_results in results.values()
                )
                writer.writerow(row)

    return csv_path


# =========================================================
# main
# =========================================================
def main():
    if EVAL_MODE not in WEIGHTS_BY_MODE:
        raise ValueError(
            f'EVAL_MODEは all / two / three のいずれかです: '
            f'{EVAL_MODE}'
        )

    OUTPUT_DIR = OUTPUT_DIRS[EVAL_MODE]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print('device:', device)
    print('evaluation mode:', EVAL_MODE)

    # 学習スクリプトと同じ患者分割
    valid_pairs = []

    for filename in sorted(
        name
        for name in os.listdir(IMAGES_DIR)
        if name.endswith('.nii.gz')
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

    # 既存PNGと同じく、全患者の最大スライス面積で30個のbinを作る
    bin_edges = make_bin_edges(valid_pairs, N_BINS)
    print(
        f'bin range: {bin_edges[0]:.0f}-{bin_edges[-1]:.0f} px '
        f'({N_BINS} bins)'
    )

    test_ds = NSCLCSliceDataset(
        test_pairs,
        mode='max',
        target_size=TARGET_SIZE,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Datasetのmetaにあるtumor_sizeは最大腫瘍スライスの面積
    test_sizes = np.asarray([
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

    print('test patients:', len(test_ds))
    print(
        'bins with test data:',
        f'{len(valid_bins)} / {N_BINS}',
    )

    results = {}

    # 指定モードのモデルをそれぞれロードし、
    # 同じテスト患者全員に対して評価する
    for model_name, weight_path in WEIGHTS_BY_MODE[EVAL_MODE].items():
        if not os.path.exists(weight_path):
            print(
                f'[警告] 重みファイルが見つかりません: '
                f'{weight_path}'
            )
            continue

        print(f'evaluating {model_name}: {weight_path}')

        model = UNet_Attention_Transformer_Multiscale(
            n_channels=3,
            n_classes=1,
        ).to(device)

        load_weights(model, weight_path, device)

        scores = evaluate_model(
            model,
            test_loader,
            device,
        )

        results[model_name] = {
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
        raise RuntimeError(
            '評価できるモデルがありません。'
            'WEIGHTS_BY_MODEの重みパスを確認してください。'
        )

    for metric in METRICS:
        plot_metric(
            EVAL_MODE,
            metric,
            results,
            bin_edges,
            counts,
            valid_bins,
            OUTPUT_DIR,
        )

    csv_path = save_csv(
        EVAL_MODE,
        results,
        bin_edges,
        counts,
        valid_bins,
        OUTPUT_DIR,
    )

    print('CSV saved:', csv_path)
    print('PNG/EPS saved to:', OUTPUT_DIR)


if __name__ == '__main__':
    main()