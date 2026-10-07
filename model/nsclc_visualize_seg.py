import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from sklearn.model_selection import train_test_split

from TransAttUnet import UNet_Attention_Transformer_Multiscale
# 同じフォルダに nsclc_train_by_size.py と nsclc_eval_by_size.py を置くこと
from nsclc_train_by_size import NSCLCSliceDataset, IMAGES_DIR, LABELS_DIR, TARGET_SIZE
from nsclc_eval_by_size import WEIGHTS, N_BINS, make_bin_edges, assign_bins, load_weights, compute_metrics

# =========================================================
# 設定
# =========================================================
# モデル名 → WEIGHTS のキー (小さい順に small / middle / large)
MODEL_NAMES = {
    'small':  '35-919px',
    'middle': '919-1804px',
    'large':  '1804-10652px',
}

# 見たい (モデル, ビン) の組。ビンは「グラフの横軸ラベルの下端」で指定する
#   例: 77-518 のビンなら 77
CASES = [
    ('small', 77),     # small  × 77-518
    ('large', 1841),    # large  × 518-959
    ('large', 2723), 
    ('large', 4928),   # large  × 1841-2282
    ('large', 10220),  # large  × 10220-10661
]

SAMPLES_PER_BIN = 6   # 1ビンあたり何枚見るか (ビン内の患者数がこれより少なければ全部)
SEED = 0              # 見る患者のランダム選択。同じビンなら、どのモデルでも同じ患者が選ばれる
CROP = 160            # 拡大表示の切り出しサイズ (512x512にリサイズした後のpx)
OUT_DIR = 'output/seg_examples'


# =========================================================
# 補助
# =========================================================
def find_bin(bin_edges, low):
    for j in range(N_BINS):
        if int(bin_edges[j]) == low:
            return j
    available = [int(bin_edges[j]) for j in range(N_BINS)]
    raise ValueError(f"下端が {low} のビンがありません。選べる値: {available}")


def overlay_tp_fp_fn(img, gt, pred):
    """TP=緑, FP=赤, FN=青 を重ねたRGB画像"""
    rgb = np.stack([img, img, img], axis=-1).astype(np.float32)
    for mask, color in ((gt & pred, (0.0, 1.0, 0.0)),
                        (~gt & pred, (1.0, 0.0, 0.0)),
                        (gt & ~pred, (0.0, 0.4, 1.0))):
        rgb[mask] = 0.45 * rgb[mask] + 0.55 * np.array(color, dtype=np.float32)
    return rgb


def crop_window(gt, crop):
    """GTの重心を中心にした切り出し位置 (y0, x0)"""
    h, w = gt.shape
    ys, xs = np.nonzero(gt)
    cy, cx = int(ys.mean()), int(xs.mean())
    y0 = int(np.clip(cy - crop // 2, 0, max(h - crop, 0)))
    x0 = int(np.clip(cx - crop // 2, 0, max(w - crop, 0)))
    return y0, x0


def draw_case_figure(rows, title, path, crop=CROP):
    """rows: dict(img, gt, pred, name, size, iou, dice) のリスト"""
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(13.5, 3.3 * n), squeeze=False)
    col_titles = ['Full slice (green: GT / red: Pred)', 'Zoom: GT contour',
                  'Zoom: Prediction contour', 'Zoom: TP / FP / FN']

    for r, row in enumerate(rows):
        img, gt, pred = row['img'], row['gt'], row['pred']
        y0, x0 = crop_window(gt, crop)
        sl = (slice(y0, y0 + crop), slice(x0, x0 + crop))

        # 全体
        ax = axes[r, 0]
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.contour(gt.astype(float), levels=[0.5], colors='lime', linewidths=1.2)
        if pred.any():
            ax.contour(pred.astype(float), levels=[0.5], colors='red', linewidths=1.2)
        ax.add_patch(Rectangle((x0, y0), crop, crop, fill=False, edgecolor='white', linewidth=1.2))
        ax.set_xlim(-0.5, img.shape[1] - 0.5)
        ax.set_ylim(img.shape[0] - 0.5, -0.5)
        ax.set_ylabel(f"{row['name']}\nsize={row['size']}px\nIoU={row['iou']:.2f}  Dice={row['dice']:.2f}",
                      fontsize=9, rotation=0, ha='right', va='center', labelpad=8)

        # 拡大: GT
        ax = axes[r, 1]
        ax.imshow(img[sl], cmap='gray', vmin=0, vmax=1)
        ax.contour(gt[sl].astype(float), levels=[0.5], colors='lime', linewidths=1.5)

        # 拡大: 予測
        ax = axes[r, 2]
        ax.imshow(img[sl], cmap='gray', vmin=0, vmax=1)
        if pred[sl].any():
            ax.contour(pred[sl].astype(float), levels=[0.5], colors='red', linewidths=1.5)

        # 拡大: TP/FP/FN
        axes[r, 3].imshow(overlay_tp_fp_fn(img[sl], gt[sl], pred[sl]))

        for c in range(4):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=10)

    fig.legend(handles=[Patch(color='lime', label='TP'), Patch(color='red', label='FP'),
                        Patch(color=(0.0, 0.4, 1.0), label='FN')],
               loc='upper right', fontsize=10, ncol=3)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.08, 0, 1, 0.97])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =========================================================
# main
# =========================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 学習・評価時と同じ分割、同じビン ----
    valid_pairs = []
    for f in sorted(f for f in os.listdir(IMAGES_DIR) if f.endswith('.nii.gz')):
        lbl = os.path.join(LABELS_DIR, f)
        if os.path.exists(lbl):
            valid_pairs.append({'image': os.path.join(IMAGES_DIR, f), 'label': lbl, 'name': f})
    _, test_pairs = train_test_split(valid_pairs, test_size=0.2, random_state=42)

    bin_edges = make_bin_edges(valid_pairs, N_BINS)
    test_ds = NSCLCSliceDataset(test_pairs, mode='max', target_size=TARGET_SIZE)
    test_sizes = np.array([m['tumor_size'] for m in test_ds.meta])
    bin_idx = assign_bins(test_sizes, bin_edges)

    # ---- ビンごとに見る患者を決める (モデルによらず同じ患者) ----
    chosen = {}
    for _, low in CASES:
        j = find_bin(bin_edges, low)
        if j in chosen:
            continue
        candidates = np.where(bin_idx == j)[0]
        rng = np.random.default_rng(SEED)
        k = min(SAMPLES_PER_BIN, len(candidates))
        pick = rng.choice(candidates, size=k, replace=False)
        chosen[j] = sorted(pick.tolist(), key=lambda i: test_sizes[i])
        print(f"ビン {int(bin_edges[j])}-{int(bin_edges[j + 1])}: テスト{len(candidates)}枚中 {k}枚を選択")

    # ---- モデルごとに推論して図を作る ----
    for model_key in dict.fromkeys(m for m, _ in CASES):  # 順番を保って重複を除く
        weight_path = WEIGHTS[MODEL_NAMES[model_key]]
        if not os.path.exists(weight_path):
            print(f"[警告] 重みファイルが見つかりません: {weight_path} (スキップ)")
            continue
        model = UNet_Attention_Transformer_Multiscale(n_channels=1, n_classes=1).to(device)
        load_weights(model, weight_path, device)
        model.eval()

        for m_key, low in CASES:
            if m_key != model_key:
                continue
            j = find_bin(bin_edges, low)
            rows = []
            for i in chosen[j]:
                img_t, mask_t, name, size = test_ds[i]
                with torch.no_grad():
                    out = model(img_t.unsqueeze(0).to(device))
                pred = (torch.sigmoid(out)[0, 0] > 0.5).cpu().numpy()
                gt = mask_t[0].numpy() > 0.5
                score = compute_metrics(pred.flatten(), gt.flatten())
                rows.append({
                    'img': img_t[0].numpy(), 'gt': gt, 'pred': pred,
                    'name': name.replace('.nii.gz', ''), 'size': int(size),
                    'iou': float(score['IoU']), 'dice': float(score['DICE']),
                })
                print(f"  [{m_key}] {name}  size={int(size)}  IoU={score['IoU']:.3f}")

            bin_label = f"{int(bin_edges[j])}-{int(bin_edges[j + 1])}px"
            title = f"{m_key} model ({MODEL_NAMES[m_key]})  x  bin {bin_label}"
            path = os.path.join(OUT_DIR, f"{m_key}_bin{bin_label}.png")
            draw_case_figure(rows, title, path)
            print(f"保存しました: {path}")

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
