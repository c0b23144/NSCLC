"""
PRE(適合率) と REC(再現率) を比較しやすくする評価スクリプト。
 
nsclc_eval_ct_patient.py と同じフォルダに置いて実行する。
モデルの読み込み・評価は nsclc_eval_ct_patient.py の関数を再利用している。
 
    python nsclc_eval_pre_rec.py            # EVAL_CT_MODES の条件を評価
    python nsclc_eval_pre_rec.py lung raw   # 指定した条件だけ
    python nsclc_eval_pre_rec.py all        # 3条件すべて
 
見方:
    REC - PRE > 0  → REC が高い = 過検出 (腫瘍を大きめに塗る / FPが多い)
    REC - PRE < 0  → PRE が高い = 見逃し (腫瘍を小さめに塗る / FNが多い)
"""
import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
 
from TransAttUnet import UNet_Attention_Transformer_Multiscale
from Nsclc_train_ct import (
    NSCLCSliceDataset, IMAGES_DIR, LABELS_DIR, TARGET_SIZE, CT_SETTINGS
)
from nsclc_eval_ct_patient import (
    WEIGHTS, STYLES, DISPLAY_NAMES,
    patient_max_area, load_weights, evaluate_model, parse_ct_modes,
)
 
# =========================================================
# 設定
# =========================================================
OUTPUT_DIR = 'output/ct_pre_rec'
 
# |REC - PRE| がこの値未満なら「バランス良好」とみなす
BALANCE_TOL = 0.05
 
PRE_COLOR = '#D32F2F'   # PRE: 赤
REC_COLOR = '#1976D2'   # REC: 青
 
 
# =========================================================
# 判定
# =========================================================
def tendency(pre, rec, tol=BALANCE_TOL):
    """1スライスぶんのPRE/RECから傾向を判定する"""
    if pre == 0 and rec == 0:
        return 'miss'            # 予測が全く重ならない (見逃し or 空予測)
    diff = rec - pre
    if diff > tol:
        return 'over'            # 過検出 (RECが高い)
    if diff < -tol:
        return 'under'           # 見逃し寄り (PREが高い)
    return 'balanced'
 
 
# =========================================================
# グラフ
# =========================================================
def _patient_labels(names, sizes, order):
    return [f"{names[i]}\n({sizes[i]})" for i in order]
 
 
def plot_pre_rec_line(ct, res, names, sizes, out_dir):
    """1つのCT条件について、PREとRECを同じ軸に重ねる (横軸=患者, 面積の小さい順)"""
    order = np.argsort(sizes, kind='stable')
    x = np.arange(len(names))
    pre = res['scores']['PRE'][order]
    rec = res['scores']['REC'][order]
 
    plt.figure(figsize=(max(12, len(names) * 0.45), 7.5))
    plt.plot(x, pre, color=PRE_COLOR, marker='o', linestyle='-', linewidth=1.3,
             markersize=7, label='PRE (precision)')
    plt.plot(x, rec, color=REC_COLOR, marker='^', linestyle='--', linewidth=1.3,
             markersize=7, label='REC (recall)')
    # 差の部分を塗る: RECが上=青 (過検出), PREが上=赤 (見逃し寄り)
    plt.fill_between(x, pre, rec, where=rec >= pre, color=REC_COLOR, alpha=0.15,
                     interpolate=True, label='REC > PRE (over-seg.)')
    plt.fill_between(x, pre, rec, where=rec < pre, color=PRE_COLOR, alpha=0.15,
                     interpolate=True, label='PRE > REC (under-seg.)')
 
    plt.xticks(x, _patient_labels(names, sizes, order), rotation=90, fontsize=8)
    plt.xlabel('Patient ID (sorted by tumor area on max slice, small → large)',
               fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title(f'PRE vs REC per Patient: {DISPLAY_NAMES[ct]}', fontsize=14, fontweight='bold', pad=15)
    plt.xlim(-0.5, len(names) - 0.5)
    plt.ylim(-0.05, 1.05)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'pre_rec_line_{ct}.png'), dpi=200)
    plt.savefig(os.path.join(out_dir, f'pre_rec_line_{ct}.eps'))
    plt.close()

def plot_pre_rec_scatter(results, out_dir):
    """横軸=PRE, 縦軸=REC の散布図 (1点=1患者, CT条件ごとに色分け)"""
    plt.figure(figsize=(8, 8))
    for ct, res in results.items():
        st = STYLES[ct]
        plt.scatter(res['scores']['PRE'], res['scores']['REC'],
                    color=st['color'], marker=st['marker'], s=55, alpha=0.75,
                    edgecolors='white', linewidths=0.5, label=DISPLAY_NAMES[ct])
        # 平均点 (大きい星)
        plt.scatter(res['overall']['PRE'], res['overall']['REC'],
                    color=st['color'], marker='*', s=380, edgecolors='black', linewidths=1.2,
                    zorder=5)
 
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.6)   # REC = PRE
    plt.text(0.03, 0.93, 'REC > PRE\n(over-segmentation)', fontsize=10, color='gray', va='top')
    plt.text(0.97, 0.05, 'PRE > REC\n(under-segmentation)', fontsize=10, color='gray',
             ha='right', va='bottom')
 
    plt.xlabel('PRE (precision)', fontsize=12, fontweight='bold')
    plt.ylabel('REC (recall)', fontsize=12, fontweight='bold')
    plt.title('PRE vs REC (each dot = patient, star = mean)', fontsize=14, fontweight='bold', pad=15)
    plt.xlim(-0.03, 1.03)
    plt.ylim(-0.03, 1.03)
    plt.gca().set_aspect('equal')
    plt.legend(loc='lower left', fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pre_rec_scatter.png'), dpi=200)
    plt.savefig(os.path.join(out_dir, 'pre_rec_scatter.eps'))
    plt.close()
 
 
def plot_pre_rec_diff(results, names, sizes, out_dir):
    """REC - PRE を患者ごとに表示 (0より上=過検出, 下=見逃し寄り)"""
    order = np.argsort(sizes, kind='stable')
    x = np.arange(len(names))
 
    plt.figure(figsize=(max(12, len(names) * 0.45), 7.5))
    plt.axhspan(-BALANCE_TOL, BALANCE_TOL, color='gray', alpha=0.15,
                label=f'balanced (|diff| < {BALANCE_TOL})')
    plt.axhline(0, color='black', linewidth=1)
    for ct, res in results.items():
        diff = (res['scores']['REC'] - res['scores']['PRE'])[order]
        plt.plot(x, diff, label=DISPLAY_NAMES[ct], linewidth=1.2, markersize=7,
                 alpha=0.9, **STYLES[ct])
 
    plt.xticks(x, _patient_labels(names, sizes, order), rotation=90, fontsize=8)
    plt.xlabel('Patient ID (sorted by tumor area on max slice, small → large)',
               fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('REC − PRE', fontsize=12, fontweight='bold')
    plt.title('REC − PRE per Patient  (+: over-segmentation / −: under-segmentation)',
              fontsize=14, fontweight='bold', pad=15)
    plt.xlim(-0.5, len(names) - 0.5)
    plt.ylim(-1.05, 1.05)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pre_rec_diff.png'), dpi=200)
    plt.savefig(os.path.join(out_dir, 'pre_rec_diff.eps'))
    plt.close()
 
 
def plot_pre_rec_bar(results, out_dir):
    """CT条件ごとの全体平均PRE/RECを棒グラフで並べる"""
    cts = list(results.keys())
    x = np.arange(len(cts))
    w = 0.36
    pre = [results[c]['overall']['PRE'] for c in cts]
    rec = [results[c]['overall']['REC'] for c in cts]
 
    plt.figure(figsize=(8, 6))
    b1 = plt.bar(x - w / 2, pre, w, color=PRE_COLOR, label='PRE (precision)')
    b2 = plt.bar(x + w / 2, rec, w, color=REC_COLOR, label='REC (recall)')
    for bars in (b1, b2):
        for b in bars:
            plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                     f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=10)
 
    plt.xticks(x, [DISPLAY_NAMES[c] for c in cts], fontsize=11)
    plt.ylabel('Mean Score', fontsize=12, fontweight='bold')
    plt.title('Mean PRE vs REC by CT condition', fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0, 1.1)
    plt.legend(fontsize=10, frameon=True, shadow=True)
    plt.grid(True, axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pre_rec_bar.png'), dpi=200)
    plt.savefig(os.path.join(out_dir, 'pre_rec_bar.eps'))
    plt.close()
 
 
# =========================================================
# CSV
# =========================================================
def save_csv_per_patient(results, names, sizes, out_dir):
    """患者ごと: CT条件ごとに PRE / REC / REC-PRE / 傾向 を並べる"""
    with open(os.path.join(out_dir, 'pre_rec_per_patient.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['patient', 'tumor_area']
        for ct in results:
            header += [f'{ct}_PRE', f'{ct}_REC', f'{ct}_REC-PRE', f'{ct}_tendency']
        writer.writerow(header)
        for i in np.argsort(sizes, kind='stable'):
            row = [names[i], sizes[i]]
            for res in results.values():
                p = float(res['scores']['PRE'][i])
                r = float(res['scores']['REC'][i])
                row += [f'{p:.4f}', f'{r:.4f}', f'{r - p:+.4f}', tendency(p, r)]
            writer.writerow(row)
 
 
def save_csv_summary(results, out_dir):
    """CT条件ごとの平均と、傾向ごとの患者数"""
    with open(os.path.join(out_dir, 'pre_rec_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ct_mode', 'n', 'mean_PRE', 'mean_REC', 'mean_REC-PRE',
                         'n_over', 'n_under', 'n_balanced', 'n_miss'])
        for ct, res in results.items():
            pre, rec = res['scores']['PRE'], res['scores']['REC']
            tags = [tendency(float(p), float(r)) for p, r in zip(pre, rec)]
            writer.writerow([
                ct, len(pre),
                f'{pre.mean():.4f}', f'{rec.mean():.4f}', f'{(rec - pre).mean():+.4f}',
                tags.count('over'), tags.count('under'), tags.count('balanced'), tags.count('miss'),
            ])
 
 
def print_summary(results):
    print('\n===== PRE / REC まとめ =====')
    print(f"{'CT条件':<13}{'PRE':>8}{'REC':>8}{'REC-PRE':>10}   over / under / balanced / miss")
    for ct, res in results.items():
        pre, rec = res['scores']['PRE'], res['scores']['REC']
        tags = [tendency(float(p), float(r)) for p, r in zip(pre, rec)]
        print(f"{ct:<13}{pre.mean():>8.4f}{rec.mean():>8.4f}{(rec - pre).mean():>+10.4f}   "
              f"{tags.count('over'):>4} / {tags.count('under'):>5} / {tags.count('balanced'):>8} / {tags.count('miss'):>4}")
 
 
# =========================================================
# main
# =========================================================
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
 
    names = [p['name'].replace('.nii.gz', '') for p in test_pairs]
    areas = np.array([patient_max_area(p['label']) for p in test_pairs])
 
    results = {}
    for ct_mode in ct_modes:
        path = WEIGHTS.get(ct_mode)
        if path is None or not os.path.exists(path):
            print(f"[警告] [{ct_mode}] 重みファイルが見つかりません: {path} (スキップ)")
            continue
 
        print(f"\n評価中: {ct_mode} {CT_SETTINGS[ct_mode]}  ({path})")
        test_ds = NSCLCSliceDataset(test_pairs, ct_mode, mode='max', target_size=TARGET_SIZE)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
        assert len(test_ds) == len(test_pairs), "1患者1スライスの前提が崩れています"
 
        model = UNet_Attention_Transformer_Multiscale(n_channels=1, n_classes=1).to(device)
        load_weights(model, path, device)
        scores = evaluate_model(model, test_loader, device)
 
        results[ct_mode] = {
            'scores': scores,
            'overall': {m: float(v.mean()) for m, v in scores.items()},
        }
 
        del model, test_ds, test_loader
        if device.type == 'cuda':
            torch.cuda.empty_cache()
 
    if not results:
        print("評価できるモデルがありませんでした。WEIGHTSのパスを確認してください。")
        return
 
    # ---- グラフ ----
    for ct, res in results.items():
        plot_pre_rec_line(ct, res, names, areas, OUTPUT_DIR)
    plot_pre_rec_scatter(results, OUTPUT_DIR)
    plot_pre_rec_diff(results, names, areas, OUTPUT_DIR)
    plot_pre_rec_bar(results, OUTPUT_DIR)
 
    # ---- CSV・表示 ----
    save_csv_per_patient(results, names, areas, OUTPUT_DIR)
    save_csv_summary(results, OUTPUT_DIR)
    print_summary(results)
    print(f"\n保存しました: {OUTPUT_DIR}")
 
 
if __name__ == '__main__':
    main()
 