import os
import csv
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split

# =========================================================
# 設定 (学習スクリプトと同じ)
# =========================================================
IMAGES_DIR = '/workspace/NSCLC_NIfTI2/imagesTr'
LABELS_DIR = '/workspace/NSCLC_NIfTI2/labelsTr/Neoplasm_Primary'
OUT_CSV = 'output/group_stats2.csv'

GROUPS = [
    ('35-919px',     0,    919),
    ('919-1804px',   919,  1804),
    ('1804-10652px', 1804, float('inf')),
]
GROUP_BY = 'slice'  # 学習スクリプトと同じ: 'patient' or 'slice'


# =========================================================
# 処理 (ラベルだけ読むので速い)
# =========================================================
def fix_slice_axis(data):
    min_axis = np.argmin(data.shape)
    if min_axis == 0:
        return data.transpose(1, 2, 0)
    elif min_axis == 1:
        return data.transpose(0, 2, 1)
    return data


def tumor_slice_areas(label_path):
    """腫瘍が写っているスライスの腫瘍ピクセル数の配列"""
    mask = fix_slice_axis(nib.load(label_path).get_fdata(dtype=np.float32)) > 0
    areas = mask.sum(axis=(0, 1))
    return areas[areas > 0].astype(int)


def load_patients(pairs):
    patients = []
    for p in pairs:
        areas = tumor_slice_areas(p['label'])
        if len(areas) == 0:
            continue
        patients.append({'name': p['name'], 'areas': areas, 'max': int(areas.max())})
    return patients


def describe(values):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return dict(n=0, mean=np.nan, std=np.nan, min=np.nan, max=np.nan)
    return dict(n=len(v), mean=v.mean(), std=v.std(ddof=1) if len(v) > 1 else np.nan,
                min=v.min(), max=v.max())


def build_group(train, test, lo, hi):
    """1つのモデル(群)の train/test のサイズ配列を返す"""
    if GROUP_BY == 'patient':
        tr_pat = [p for p in train if lo <= p['max'] < hi]
        train_sizes = np.concatenate([p['areas'] for p in tr_pat]) if tr_pat else np.array([])
        n_train_patients = len(tr_pat)
        train_patient_max = [p['max'] for p in tr_pat]
    else:
        parts, pat_max = [], []
        for p in train:
            sel = p['areas'][(p['areas'] >= lo) & (p['areas'] < hi)]
            if len(sel):
                parts.append(sel)
                pat_max.append(p['max'])
        train_sizes = np.concatenate(parts) if parts else np.array([])
        n_train_patients = len(parts)
        train_patient_max = pat_max

    test_sizes = [p['max'] for p in test if lo <= p['max'] < hi]  # testは患者ごとの最大スライス1枚
    return train_sizes, n_train_patients, train_patient_max, test_sizes


def main():
    valid_pairs = []
    for f in sorted(f for f in os.listdir(IMAGES_DIR) if f.endswith('.nii.gz')):
        lbl = os.path.join(LABELS_DIR, f)
        if os.path.exists(lbl):
            valid_pairs.append({'image': os.path.join(IMAGES_DIR, f), 'label': lbl, 'name': f})

    train_pairs, test_pairs = train_test_split(valid_pairs, test_size=0.2, random_state=42)
    train = load_patients(train_pairs)
    test = load_patients(test_pairs)
    print(f"患者数: train {len(train)} / test {len(test)}  (GROUP_BY={GROUP_BY})\n")

    rows = []
    for name, lo, hi in GROUPS:
        tr_sizes, n_tr_pat, tr_pat_max, te_sizes = build_group(train, test, lo, hi)
        d_tr, d_te, d_pm = describe(tr_sizes), describe(te_sizes), describe(tr_pat_max)
        rows.append({
            'model': name,
            'train_patients': n_tr_pat,
            'train_slices': d_tr['n'],
            'test_samples': d_te['n'],
            'train_slice_mean': d_tr['mean'], 'train_slice_std': d_tr['std'],
            'train_slice_min': d_tr['min'], 'train_slice_max': d_tr['max'],
            'train_patientmax_std': d_pm['std'],
            'test_mean': d_te['mean'], 'test_std': d_te['std'],
        })

    # ---- 表示 ----
    print(f"{'model':<14}{'train患者':>9}{'trainスライス':>13}{'test枚数':>9}"
          f"{'train面積 平均':>15}{'標準偏差':>10}{'test面積 平均':>14}{'標準偏差':>10}")
    for r in rows:
        print(f"{r['model']:<14}{r['train_patients']:>9}{r['train_slices']:>13}{r['test_samples']:>9}"
              f"{r['train_slice_mean']:>15.1f}{r['train_slice_std']:>10.1f}"
              f"{r['test_mean']:>14.1f}{r['test_std']:>10.1f}")
    print("\n(標準偏差は不偏標準偏差 ddof=1。面積の単位はpx)")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"保存しました: {OUT_CSV}")


if __name__ == '__main__':
    main()
