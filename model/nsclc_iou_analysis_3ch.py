# -*- coding: utf-8 -*-
"""
NSCLC: サイズ別重み (all / two / three) の IoU + Train/Test 特徴分析 (3ch)

既存の all / two / three の Test 評価・クロス評価を維持し、
Train split と Test split の両方を患者ごとの最大腫瘍面積スライス1枚で評価する。

Train/Test は別画像なので、画像単位の IoU 相関は計算しない。
代わりに各 split 内で Low/Middle/High IoU の画像特徴を比較し、
Train と Test の画像特徴量分布も比較する。

必要ファイル:
  TransAttUnet.py
  nsclc_train_by_size3_3ch.py
  nsclc_eval_by_size3_3ch_modes.py

出力:
  output/size_weight_iou_analysis/
"""

import os
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import mannwhitneyu, kruskal

from TransAttUnet import UNet_Attention_Transformer_Multiscale
from nsclc_train_by_size3_3ch import (
    NSCLCSliceDataset,
    IMAGES_DIR,
    LABELS_DIR,
    TARGET_SIZE,
)
from nsclc_eval_by_size3_3ch_modes import (
    WEIGHTS_BY_MODE,
    GROUP_BOUNDARIES,
    BATCH_SIZE,
    METRICS,
    compute_metrics,
    load_weights,
)

# =========================================================
# 設定
# =========================================================

OUTPUT_DIR = "output/size_weight_iou_analysis"
STRATEGY_ORDER = ["all", "two", "three"]

REPORT_BOUNDARIES = GROUP_BOUNDARIES["three"]
REPORT_LABELS = ["Small", "Middle", "Large"]

STRATEGY_COLORS = {
    "all": "#2196F3",
    "two": "#8BC34A",
    "three": "#E65100",
}

LOW_IOU_THRESHOLD = 0.5
HIGH_IOU_THRESHOLD = 0.7
IOU_THRESHOLDS_TO_REPORT = [0.3, 0.5, 0.7]

BATCH_SIZE_EVAL = BATCH_SIZE
NUM_WORKERS = 0


# =========================================================
# 患者分割・サイズ群
# =========================================================

def make_pairs():
    pairs = []
    for filename in sorted(
        n for n in os.listdir(IMAGES_DIR) if n.endswith(".nii.gz")
    ):
        label_path = os.path.join(LABELS_DIR, filename)
        if os.path.exists(label_path):
            pairs.append({
                "image": os.path.join(IMAGES_DIR, filename),
                "label": label_path,
                "name": filename,
            })

    if len(pairs) < 2:
        raise RuntimeError("有効な画像・ラベルのペアが2件未満です。")

    train_pairs, test_pairs = train_test_split(
        pairs, test_size=0.2, random_state=42
    )
    return train_pairs, test_pairs


def group_of(sizes, boundaries, labels):
    idx = np.searchsorted(
        np.asarray(boundaries), np.asarray(sizes), side="right"
    )
    return np.asarray(labels, dtype=object)[idx]


# =========================================================
# 画像特徴量
# =========================================================

def mask_features(mask, image_3ch):
    """
    512x512 の正解マスクと前処理済み3ch画像から特徴量を計算。
    面積・形状特徴量はリサイズ後マスク上の値。
    元の最大スライス面積は tumor_size として別途保持する。
    """
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    area = int(mask.sum())

    if area == 0:
        return {
            "mask_area_512": 0,
            "perimeter": 0.0,
            "circularity": np.nan,
            "aspect_ratio": np.nan,
            "eccentricity": np.nan,
            "solidity": np.nan,
            "bbox_area": 0,
            "tumor_occupancy": 0.0,
            "component_count": 0,
            "largest_component_ratio": np.nan,
            "ch1_tumor_mean": np.nan,
            "ch1_tumor_std": np.nan,
            "ch2_tumor_mean": np.nan,
            "ch2_tumor_std": np.nan,
            "ch3_tumor_mean": np.nan,
            "ch3_tumor_std": np.nan,
            "ch1_ring_mean": np.nan,
            "ch2_ring_mean": np.nan,
            "ch3_ring_mean": np.nan,
        }

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bbox_h = y1 - y0
    bbox_w = x1 - x0
    bbox_area = bbox_h * bbox_w

    # 8近傍での境界画素数を周囲長の近似値として使う
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
    perimeter = float(np.logical_and(mask, ~eroded).sum())
    circularity = (
        float(4.0 * np.pi * area / (perimeter ** 2))
        if perimeter > 0 else np.nan
    )

    # 画素座標の共分散行列から楕円の偏心率を計算
    coords = np.column_stack([xs.astype(float), ys.astype(float)])
    if len(coords) >= 2:
        cov = np.cov(coords, rowvar=False)
        eig = np.maximum(np.linalg.eigvalsh(cov), 0)
        eccentricity = (
            float(np.sqrt(1.0 - eig[0] / eig[1]))
            if eig[1] > 0 else 0.0
        )
    else:
        eccentricity = 0.0

    # ConvexHull の面積。失敗時は欠損値。
    try:
        hull_area = float(ConvexHull(coords).volume) if len(coords) >= 3 else np.nan
        solidity = float(area / hull_area) if hull_area > 0 else np.nan
    except (QhullError, ValueError):
        solidity = np.nan

    labels, n_components = ndimage.label(mask)
    component_sizes = np.bincount(labels.ravel())[1:]
    largest_component_ratio = (
        float(component_sizes.max() / area) if len(component_sizes) else np.nan
    )

    # 腫瘍の周囲1〜5画素のリング領域（腫瘍内部を除外）
    dilated = ndimage.binary_dilation(mask, iterations=5)
    ring = dilated & ~mask

    result = {
        "mask_area_512": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "aspect_ratio": float(max(bbox_w, bbox_h) / max(1, min(bbox_w, bbox_h))),
        "eccentricity": eccentricity,
        "solidity": solidity,
        "bbox_area": int(bbox_area),
        "tumor_occupancy": float(area / (h * w)),
        "component_count": int(n_components),
        "largest_component_ratio": largest_component_ratio,
    }

    image_3ch = np.asarray(image_3ch)
    for ch in range(min(3, image_3ch.shape[0])):
        values = image_3ch[ch]
        result[f"ch{ch + 1}_tumor_mean"] = float(values[mask].mean())
        result[f"ch{ch + 1}_tumor_std"] = float(values[mask].std())
        result[f"ch{ch + 1}_ring_mean"] = (
            float(values[ring].mean()) if ring.any() else np.nan
        )

    return result


def extract_dataset_features(dataset, split_name):
    rows = []
    for i, meta in enumerate(dataset.meta):
        image = dataset.images[i].float().numpy()
        mask = dataset.masks[i].squeeze().numpy() > 0
        features = mask_features(mask, image)
        rows.append({
            "split": split_name,
            "case": os.path.basename(str(meta["name"])),
            "tumor_size": int(meta["tumor_size"]),
            "patient_max_size": int(meta["patient_max_size"]),
            **features,
        })

    df = pd.DataFrame(rows)
    df["size_group"] = group_of(
        df["tumor_size"], REPORT_BOUNDARIES, REPORT_LABELS
    )
    return df


# =========================================================
# モデル評価
# =========================================================

def evaluate_weight(split_name, mode, model_name, path, loader, device):
    model = UNet_Attention_Transformer_Multiscale(
        n_channels=3, n_classes=1
    ).to(device)
    load_weights(model, path, device)
    model.eval()

    rows = []
    with torch.no_grad():
        for images, masks, names, sizes in loader:
            outputs = model(images.to(device))
            preds = (
                torch.sigmoid(outputs) > 0.5
            ).cpu().numpy().astype(bool)
            targets = masks.numpy() > 0.5

            for k, name in enumerate(names):
                metrics = compute_metrics(
                    preds[k].flatten(), targets[k].flatten()
                )
                rows.append({
                    "split": split_name,
                    "mode": mode,
                    "model": model_name,
                    "case": os.path.basename(str(name)),
                    "tumor_size": int(sizes[k]),
                    **metrics,
                })

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def build_strategy_table(weight_df):
    frames = []

    for mode in STRATEGY_ORDER:
        labels = list(WEIGHTS_BY_MODE[mode].keys())
        part = weight_df[weight_df["mode"] == mode].copy()
        have = set(part["model"].unique())

        if not set(labels).issubset(have):
            print(
                f"[警告] {mode}: 重みが揃っていないため方式評価をスキップ "
                f"({sorted(set(labels) - have)})"
            )
            continue

        # 同一 split 内で、各症例のサイズに応じたモデルの結果を採用
        patients = (
            part[["split", "case", "tumor_size"]]
            .drop_duplicates(["split", "case"])
            .copy()
        )
        patients["model"] = group_of(
            patients["tumor_size"], GROUP_BOUNDARIES[mode], labels
        )

        chosen = patients.merge(
            part.drop(columns=["tumor_size"]),
            on=["split", "case", "model"],
            how="left",
            validate="one_to_one",
        )
        chosen["strategy"] = mode
        frames.append(chosen)

    if not frames:
        raise RuntimeError("方式別の評価結果を作成できません。重みパスを確認してください。")

    result = pd.concat(frames, ignore_index=True)
    result["size_group"] = group_of(
        result["tumor_size"], REPORT_BOUNDARIES, REPORT_LABELS
    )
    return result


# =========================================================
# 集計・CSV
# =========================================================

def summarize(df, keys):
    grouped = df.groupby(keys, dropna=False)
    iou = (
        grouped["IoU"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
        .rename(columns={
            "count": "n",
            "mean": "IoU_mean",
            "std": "IoU_std",
            "median": "IoU_median",
            "min": "IoU_min",
            "max": "IoU_max",
        })
    )
    others = [m for m in METRICS if m != "IoU"]
    other_mean = (
        grouped[others].mean().reset_index()
        .rename(columns={m: f"{m}_mean" for m in others})
    )
    return iou.merge(other_mean, on=keys)


def save_base_csv(weight_df, strategy_df, feature_df):
    # size_groupの重複を防ぐ
    feature_cols_for_merge = [
        col for col in feature_df.columns
        if col not in ["split", "size_group"]
    ]

    weight_out = weight_df.merge(
        feature_df[feature_cols_for_merge],
        on=["case", "tumor_size"],
        how="left",
        validate="many_to_one",
    )

    strategy_out = strategy_df.merge(
        feature_df[feature_cols_for_merge],
        on=["case", "tumor_size"],
        how="left",
        validate="many_to_one",
    )

    # size_groupが存在しない場合に、tumor_sizeから作成
    for df in [weight_out, strategy_out]:
        df["size_group"] = pd.cut(
            df["tumor_size"],
            bins=[-float("inf"), 919, 1804, float("inf")],
            labels=["Small", "Middle", "Large"],
            right=False,
        )

    weight_out.to_csv(
        os.path.join(OUTPUT_DIR, "case_metrics_per_weight.csv"), index=False
    )
    strategy_out.to_csv(
        os.path.join(OUTPUT_DIR, "case_metrics_strategy.csv"), index=False
    )

    strategy_test = strategy_out[strategy_out["split"] == "test"]
    by_size = summarize(strategy_test, ["strategy", "size_group"])
    overall = summarize(strategy_test, ["strategy"])
    per_weight = summarize(
        weight_out[weight_out["split"] == "test"],
        ["mode", "model", "size_group"],
    )

    by_size.to_csv(
        os.path.join(OUTPUT_DIR, "strategy_summary_by_size.csv"), index=False
    )
    overall.to_csv(
        os.path.join(OUTPUT_DIR, "strategy_summary_overall.csv"), index=False
    )
    per_weight.to_csv(
        os.path.join(OUTPUT_DIR, "weight_summary_by_size.csv"), index=False
    )
    return weight_out, strategy_out, by_size, overall, per_weight


def feature_statistical_comparison(df, feature_cols, group_col="iou_group"):
    rows = []
    for feature in feature_cols:
        groups = {
            label: df.loc[df[group_col] == label, feature].dropna().values
            for label in ["Low", "Middle", "High"]
        }
        low = groups["Low"]
        high = groups["High"]

        p_mw = np.nan
        if len(low) >= 2 and len(high) >= 2:
            try:
                p_mw = float(mannwhitneyu(low, high, alternative="two-sided").pvalue)
            except ValueError:
                pass

        valid_groups = [v for v in groups.values() if len(v) >= 2]
        p_kw = np.nan
        if len(valid_groups) >= 2:
            try:
                p_kw = float(kruskal(*valid_groups).pvalue)
            except ValueError:
                pass

        rows.append({
            "feature": feature,
            "Low_n": len(low),
            "Low_mean": float(np.mean(low)) if len(low) else np.nan,
            "Low_median": float(np.median(low)) if len(low) else np.nan,
            "Middle_n": len(groups["Middle"]),
            "Middle_mean": float(np.mean(groups["Middle"])) if len(groups["Middle"]) else np.nan,
            "Middle_median": float(np.median(groups["Middle"])) if len(groups["Middle"]) else np.nan,
            "High_n": len(high),
            "High_mean": float(np.mean(high)) if len(high) else np.nan,
            "High_median": float(np.median(high)) if len(high) else np.nan,
            "Low_vs_High_MannWhitney_p": p_mw,
            "Low_Middle_High_Kruskal_p": p_kw,
        })
    return pd.DataFrame(rows)


# =========================================================
# 分析用プロット
# =========================================================

FEATURE_COLS = [
    "tumor_size",
    "mask_area_512",
    "perimeter",
    "circularity",
    "aspect_ratio",
    "eccentricity",
    "solidity",
    "tumor_occupancy",
    "component_count",
    "largest_component_ratio",
    "ch1_tumor_mean",
    "ch1_tumor_std",
    "ch2_tumor_mean",
    "ch2_tumor_std",
    "ch3_tumor_mean",
    "ch3_tumor_std",
]


def add_iou_groups(df):
    result = df.copy()
    result["iou_group"] = np.select(
        [
            result["IoU"] < LOW_IOU_THRESHOLD,
            result["IoU"] < HIGH_IOU_THRESHOLD,
        ],
        ["Low", "Middle"],
        default="High",
    )
    return result


def plot_split_boxplots(strategy_df):
    for split in ["train", "test"]:
        for strategy in STRATEGY_ORDER:
            sub = strategy_df[
                (strategy_df["split"] == split)
                & (strategy_df["strategy"] == strategy)
            ]
            if sub.empty:
                continue

            groups = REPORT_LABELS + ["Overall"]
            fig, axes = plt.subplots(
                1, len(groups), figsize=(5 * len(groups), 5), squeeze=False
            )
            for ax, group in zip(axes.ravel(), groups):
                part = sub if group == "Overall" else sub[sub["size_group"] == group]
                values = part["IoU"].dropna().values
                if len(values):
                    ax.boxplot(values, showmeans=True)
                ax.set_title(f"{group} (n={len(values)})")
                ax.set_ylabel("IoU")
                ax.set_ylim(-0.05, 1.05)
                ax.grid(axis="y", alpha=0.3)

            fig.suptitle(f"{split.upper()} IoU boxplot - {strategy}")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    OUTPUT_DIR, f"iou_boxplot_{split}_{strategy}.png"
                ),
                dpi=200,
            )
            plt.close()

    # 代表図: three 方式の Train/Test 箱ひげ図
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, split in zip(axes, ["train", "test"]):
        sub = strategy_df[
            (strategy_df["split"] == split)
            & (strategy_df["strategy"] == "three")
        ]
        data = [
            sub.loc[sub["size_group"] == g, "IoU"].dropna().values
            for g in REPORT_LABELS
        ]
        data.append(sub["IoU"].dropna().values)
        ax.boxplot(data, labels=REPORT_LABELS + ["Overall"], showmeans=True)
        ax.set_title(f"{split.upper()} (three strategy)")
        ax.set_ylabel("IoU")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "iou_boxplot_train_test.png"), dpi=200)
    plt.close()


def plot_feature_group_comparison(df, split, strategy):
    sub = df[
        (df["split"] == split) & (df["strategy"] == strategy)
    ].copy()
    if sub.empty:
        return

    sub = add_iou_groups(sub)
    features = [f for f in FEATURE_COLS if f in sub.columns]
    ncols = 4
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows)
    )
    axes = np.asarray(axes).ravel()

    for ax, feature in zip(axes, features):
        data = [
            sub.loc[sub["iou_group"] == g, feature].dropna().values
            for g in ["Low", "Middle", "High"]
        ]
        if any(len(v) for v in data):
            ax.boxplot(data, labels=["Low", "Middle", "High"], showmeans=True)
        ax.set_title(feature, fontsize=9)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes[len(features):]:
        ax.axis("off")

    fig.suptitle(
        f"{split.upper()}: feature comparison by IoU group ({strategy})"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR, f"{split}_iou_feature_comparison_{strategy}.png"
        ),
        dpi=200,
    )
    plt.close()

    stats = feature_statistical_comparison(sub, features)
    stats.insert(0, "strategy", strategy)
    stats.insert(0, "split", split)
    stats.to_csv(
        os.path.join(
            OUTPUT_DIR, f"{split}_feature_statistics_{strategy}.csv"
        ),
        index=False,
    )


def plot_train_test_feature_distribution(feature_df):
    features = [
        f for f in FEATURE_COLS if f in feature_df.columns
    ]
    ncols = 4
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows)
    )
    axes = np.asarray(axes).ravel()

    for ax, feature in zip(axes, features):
        for split, color in [("train", "#2196F3"), ("test", "#E65100")]:
            values = feature_df.loc[
                feature_df["split"] == split, feature
            ].dropna().values
            if len(values):
                ax.hist(
                    values, bins=25, alpha=0.4, density=True,
                    label=split, color=color,
                )
        ax.set_title(feature, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    for ax in axes[len(features):]:
        ax.axis("off")

    fig.suptitle("Train vs Test image feature distributions")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "train_test_feature_distribution.png"),
        dpi=200,
    )
    plt.close()


def plot_iou_vs_area(strategy_df, strategy):
    for split in ["train", "test"]:
        sub = strategy_df[
            (strategy_df["split"] == split)
            & (strategy_df["strategy"] == strategy)
        ]
        if sub.empty:
            continue

        plt.figure(figsize=(7, 5))
        plt.scatter(
            sub["tumor_size"], sub["IoU"],
            alpha=0.6, s=18,
        )
        plt.xscale("log")
        for boundary in REPORT_BOUNDARIES:
            plt.axvline(boundary, color="gray", linestyle="--", linewidth=1)
        plt.xlabel("Tumor area on max slice (pixels, log)")
        plt.ylabel("IoU")
        plt.ylim(-0.05, 1.05)
        plt.title(f"{split.upper()}: tumor area vs IoU ({strategy})")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"iou_vs_tumor_area_{split}.png"),
            dpi=200,
        )
        plt.close()


def plot_train_test_iou_distribution(strategy_df, strategy):
    plt.figure(figsize=(7, 5))
    for split, color in [("train", "#2196F3"), ("test", "#E65100")]:
        sub = strategy_df[
            (strategy_df["split"] == split)
            & (strategy_df["strategy"] == strategy)
        ]
        if not sub.empty:
            plt.hist(
                sub["IoU"].dropna(), bins=np.linspace(0, 1, 21),
                alpha=0.45, label=split, color=color,
            )
    plt.xlabel("IoU")
    plt.ylabel("Number of patients")
    plt.title(f"Train vs Test IoU distribution ({strategy})")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "train_test_iou_distribution.png"),
        dpi=200,
    )
    plt.close()


def plot_original_strategy_graphs(test_strategy, by_size, per_weight):
    # 元コードの strategy 箱ひげ図
    strategies = [
        s for s in STRATEGY_ORDER
        if s in set(test_strategy["strategy"])
    ]
    groups = REPORT_LABELS + ["Overall"]
    fig, axes = plt.subplots(
        1, len(groups), figsize=(5 * len(groups), 5), squeeze=False
    )
    for ax, group in zip(axes.ravel(), groups):
        part = (
            test_strategy if group == "Overall"
            else test_strategy[test_strategy["size_group"] == group]
        )
        data = [
            part.loc[part["strategy"] == s, "IoU"].values
            for s in strategies
        ]
        ax.boxplot(data, showmeans=True)
        ax.set_xticklabels(strategies)
        ax.set_title(f"{group} (n={part['case'].nunique()})")
        ax.set_ylabel("IoU")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "iou_boxplot_strategy.png"), dpi=200)
    plt.close()

    x = np.arange(len(REPORT_LABELS))
    width = 0.8 / max(1, len(strategies))
    plt.figure(figsize=(8, 5))
    for k, strategy in enumerate(strategies):
        sub = by_size[by_size["strategy"] == strategy].set_index("size_group")
        means = [sub["IoU_mean"].get(g, np.nan) for g in REPORT_LABELS]
        stds = [sub["IoU_std"].get(g, np.nan) for g in REPORT_LABELS]
        plt.bar(
            x + k * width, means, width, yerr=stds, capsize=3,
            color=STRATEGY_COLORS[strategy], label=strategy,
        )
    plt.xticks(
        x + width * (len(strategies) - 1) / 2, REPORT_LABELS
    )
    plt.ylabel("Mean IoU")
    plt.ylim(0, 1.05)
    plt.title("Mean IoU by tumor size (routed weights, Test)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mean_iou_bar_strategy.png"), dpi=200)
    plt.close()

    for group in REPORT_LABELS:
        part = test_strategy[test_strategy["size_group"] == group]
        if part.empty:
            continue
        plt.figure(figsize=(7, 5))
        for strategy in strategies:
            values = part.loc[part["strategy"] == strategy, "IoU"]
            if len(values):
                plt.hist(
                    values, bins=np.linspace(0, 1, 21), alpha=0.4,
                    color=STRATEGY_COLORS[strategy], label=strategy,
                )
        plt.xlabel("IoU")
        plt.ylabel("Number of patients")
        plt.title(f"IoU distribution: {group}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"iou_hist_strategy_{group}.png"),
            dpi=200,
        )
        plt.close()

    plt.figure(figsize=(9, 5.5))
    for strategy in strategies:
        part = test_strategy[test_strategy["strategy"] == strategy]
        plt.scatter(
            part["tumor_size"], part["IoU"], s=14, alpha=0.5,
            color=STRATEGY_COLORS[strategy], label=strategy,
        )
    for boundary in REPORT_BOUNDARIES:
        plt.axvline(boundary, color="gray", linestyle="--", linewidth=0.9)
    plt.xscale("log")
    plt.xlabel("Tumor size (pixels on max slice, log)")
    plt.ylabel("IoU")
    plt.ylim(-0.05, 1.05)
    plt.title("Tumor size vs IoU (Test)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "size_vs_iou_strategy.png"), dpi=200
    )
    plt.close()

    # 各重みを全Test患者に適用したクロス評価
    cross = per_weight.copy()
    cross["label"] = cross["mode"] + "/" + cross["model"]
    labels = list(dict.fromkeys(cross["label"]))
    x = np.arange(len(REPORT_LABELS))
    width = 0.8 / max(1, len(labels))
    plt.figure(figsize=(10, 5.5))
    for k, label in enumerate(labels):
        sub = cross[cross["label"] == label].set_index("size_group")
        means = [sub["IoU_mean"].get(g, np.nan) for g in REPORT_LABELS]
        plt.bar(x + k * width, means, width, label=label)
    plt.xticks(
        x + width * (len(labels) - 1) / 2, REPORT_LABELS
    )
    plt.ylabel("Mean IoU")
    plt.ylim(0, 1.05)
    plt.title("Each weight applied to all Test patients")
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "weight_cross_eval_bar.png"), dpi=200
    )
    plt.close()


# =========================================================
# 低IoU症例一覧・閾値別一覧
# =========================================================

def save_low_iou_lists(strategy_df):
    rows = []
    for threshold in IOU_THRESHOLDS_TO_REPORT:
        part = strategy_df[strategy_df["IoU"] < threshold].copy()
        part["threshold"] = threshold
        rows.append(part)

    if rows:
        pd.concat(rows, ignore_index=True).to_csv(
            os.path.join(OUTPUT_DIR, "low_iou_cases_by_threshold.csv"),
            index=False,
        )

    # 既定のIoU<0.5について、Train/Test別の一覧
    for split in ["train", "test"]:
        part = strategy_df[
            (strategy_df["split"] == split)
            & (strategy_df["IoU"] < LOW_IOU_THRESHOLD)
        ]
        part.to_csv(
            os.path.join(OUTPUT_DIR, f"low_iou_cases_{split}.csv"),
            index=False,
        )


# =========================================================
# main
# =========================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("===== NSCLC Train/Test IoU and feature analysis =====")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_pairs, test_pairs = make_pairs()
    print("patients: train =", len(train_pairs), ", test =", len(test_pairs))

    # TrainもTestも同じ条件: 患者ごとの最大腫瘍面積スライス1枚
    train_ds = NSCLCSliceDataset(
        train_pairs, mode="max", target_size=TARGET_SIZE
    )
    test_ds = NSCLCSliceDataset(
        test_pairs, mode="max", target_size=TARGET_SIZE
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE_EVAL, shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE_EVAL, shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("usable max-slice cases: train =", len(train_ds),
          ", test =", len(test_ds))

    # 画像特徴量は正解マスク・入力画像から一度だけ計算
    train_features = extract_dataset_features(train_ds, "train")
    test_features = extract_dataset_features(test_ds, "test")
    feature_df = pd.concat(
        [train_features, test_features], ignore_index=True
    )
    feature_df.to_csv(
        os.path.join(OUTPUT_DIR, "image_features_train_test.csv"),
        index=False,
    )

    # 各best_modelをTrain/Test両方に適用
    all_frames = []
    for mode in STRATEGY_ORDER:
        for model_name, path in WEIGHTS_BY_MODE[mode].items():
            if not os.path.exists(path):
                print(f"[警告] 重みが見つかりません: {path} (skip)")
                continue

            print(f"evaluating {mode}/{model_name}: {path}")
            for split_name, loader in [
                ("train", train_loader),
                ("test", test_loader),
            ]:
                print(f"  split: {split_name}")
                all_frames.append(
                    evaluate_weight(
                        split_name, mode, model_name, path, loader, device
                    )
                )

    if not all_frames:
        raise RuntimeError(
            "評価できるbest_model.pthがありません。WEIGHTS_BY_MODEのパスを確認してください。"
        )

    weight_df = pd.concat(all_frames, ignore_index=True)
    strategy_df = build_strategy_table(weight_df)

    weight_out, strategy_out, by_size, overall, per_weight = save_base_csv(
        weight_df, strategy_df, feature_df
    )

    # Test側の既存集計・図を維持
    test_strategy = strategy_out[strategy_out["split"] == "test"].copy()
    plot_original_strategy_graphs(test_strategy, by_size, per_weight)

    # Train/Testそれぞれの箱ひげ図
    plot_split_boxplots(strategy_out)

    # 特徴量比較: all/two/three方式それぞれで出力
    for strategy in STRATEGY_ORDER:
        for split in ["train", "test"]:
            plot_feature_group_comparison(strategy_out, split, strategy)

    # Train/Testの入力特徴量分布。画像単位で比較するため方式によらない
    plot_train_test_feature_distribution(feature_df)

    # 代表方式threeのIoUと面積、Train/Test IoU分布
    plot_iou_vs_area(strategy_out, "three")
    plot_train_test_iou_distribution(strategy_out, "three")

    # 低IoU症例一覧（方式別の行を保持）
    save_low_iou_lists(strategy_out)

    # Train/TestのIoU群別統計をまとめる
    stat_frames = []
    for split in ["train", "test"]:
        for strategy in STRATEGY_ORDER:
            path = os.path.join(
                OUTPUT_DIR, f"{split}_feature_statistics_{strategy}.csv"
            )
            if os.path.exists(path):
                stat_frames.append(pd.read_csv(path))
    if stat_frames:
        pd.concat(stat_frames, ignore_index=True).to_csv(
            os.path.join(OUTPUT_DIR, "low_high_iou_feature_statistics.csv"),
            index=False,
        )

    print("\n===== Test strategy summary (overall) =====")
    print(overall.to_string(index=False))
    print("\n===== Test strategy summary by size =====")
    print(by_size.to_string(index=False))
    print("\n===== Test cross evaluation by weight and size =====")
    print(per_weight.to_string(index=False))

    print("\n===== Finished =====")
    print("Output directory:", OUTPUT_DIR)
    for filename in sorted(os.listdir(OUTPUT_DIR)):
        print(" -", filename)


if __name__ == "__main__":
    main()
