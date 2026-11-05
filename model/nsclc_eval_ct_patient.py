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

OUTPUT_DIR = 'output/ct_metrics_patient'
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
# 患者ごとの腫瘍面積 (最大スライス)
# =========================================================
def patient_max_area(label_path):
    mask = fix_slice_axis(nib.load(label_path).get_fdata(dtype=np.float32)) > 0
    return int(mask.sum(axis=(0, 1)).max())


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


# =========================================================
# グラフ・CSV (横軸=患者ID、腫瘍面積の小さい順)
# =========================================================
def plot_per_patient(metric, results, names, sizes, out_dir):
    order = np.argsort(sizes, kind='stable')
    x = np.arange(len(names))
    labels = [f"{names[i]}\n({sizes[i]})" for i in order]  # ID + 面積

    plt.figure(figsize=(max(12, len(names) * 0.45), 7.5))
    for name, res in results.items():
        y = res['scores'][metric][order]
        plt.plot(x, y, label=DISPLAY_NAMES[name], linewidth=1.2, markersize=7,
                 alpha=0.9, **STYLES[name])

    plt.xticks(x, labels, rotation=90, fontsize=8)
    plt.xlabel('Patient ID (sorted by tumor area on max slice, small → large)',
               fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel(f'{metric} Score', fontsize=12, fontweight='bold')
    plt.title(f'CT-condition Models: {metric} per Patient', fontsize=14, fontweight='bold', pad=15)
    plt.xlim(-0.5, len(names) - 0.5)
    plt.ylim(0.9, 1.01) if metric == 'ACC' else plt.ylim(-0.05, 1.05)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f'per_patient_{metric}.png'), dpi=200)
    plt.savefig(os.path.join(out_dir, f'per_patient_{metric}.eps'))
    plt.close()


def save_csv_per_patient(results, names, sizes, out_dir):
    with open(os.path.join(out_dir, 'per_patient_metrics.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['patient', 'tumor_area']
        for ct in results:
            header += [f'{ct}_{m}' for m in METRICS]
        writer.writerow(header)
        for i in np.argsort(sizes, kind='stable'):
            row = [names[i], sizes[i]]
            for res in results.values():
                row += [f"{res['scores'][m][i]:.4f}" for m in METRICS]
            writer.writerow(row)


def save_csv_overall(results, n_test, out_dir):
    """患者で分けない、テスト全体の平均"""
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

    # ---- 患者ID と 腫瘍面積 (test_pairs と同じ順番) ----
    names = [p['name'].replace('.nii.gz', '') for p in test_pairs]
    areas = np.array([patient_max_area(p['label']) for p in test_pairs])

    results = {}
    for ct_mode in ct_modes:
        path = WEIGHTS.get(ct_mode)
        if path is None or not os.path.exists(path):
            print(f"[警告] [{ct_mode}] 重みファイルが見つかりません: {path} (スキップ)")
            continue

        # 学習時と同じCT条件でテストデータを作る (ここがずれるとIoUが0になる)
        print(f"\n評価中: {ct_mode} {CT_SETTINGS[ct_mode]}  ({path})")
        test_ds = NSCLCSliceDataset(test_pairs, ct_mode, mode='max', target_size=TARGET_SIZE)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
        assert len(test_ds) == len(test_pairs), "1患者1スライスの前提が崩れています"

        model = UNet_Attention_Transformer_Multiscale(n_channels=1, n_classes=1).to(device)
        load_weights(model, path, device)
        scores = evaluate_model(model, test_loader, device)

        results[ct_mode] = {
            'scores': scores,
            'overall': {m: float(scores[m].mean()) for m in METRICS},
        }
        print("  全体平均: " + ", ".join(f"{m}={results[ct_mode]['overall'][m]:.4f}" for m in METRICS))

        del model, test_ds, test_loader
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if not results:
        print("評価できるモデルがありませんでした。WEIGHTSのパスを確認してください。")
        return

    # ---- グラフとCSV ----
    for metric in METRICS:
        plot_per_patient(metric, results, names, areas, OUTPUT_DIR)
    save_csv_per_patient(results, names, areas, OUTPUT_DIR)
    save_csv_overall(results, len(test_pairs), OUTPUT_DIR)
    print(f"保存しました: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
