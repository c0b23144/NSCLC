import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import nibabel as nib
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, jaccard_score, precision_score, recall_score

from TransAttUnet import UNet_Attention_Transformer_Multiscale


class NSCLCNiftiDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, target_size=(512, 512), is_train=False):
        self.pairs = pairs
        self.target_size = target_size
        self.is_train = is_train
        self.extended_pairs = []

        for pair in pairs:
            mask_nii = nib.load(pair['label']).get_fdata()
            shape = mask_nii.shape
            min_axis = np.argmin(shape)
            if min_axis == 0:
                mask_nii = mask_nii.transpose(1, 2, 0)
            elif min_axis == 1:
                mask_nii = mask_nii.transpose(0, 2, 1)

            slice_areas = [np.sum(mask_nii[:, :, i] > 0) for i in range(mask_nii.shape[2])]
            valid_idx = [i for i, area in enumerate(slice_areas) if area > 0]
            if len(valid_idx) == 0:
                continue

            max_mask_idx = max(valid_idx, key=lambda i: slice_areas[i])
            new_pair = pair.copy()
            new_pair['chosen_mask_idx'] = max_mask_idx
            new_pair['tumor_size'] = slice_areas[max_mask_idx]
            new_pair['extended_name'] = pair['name']
            self.extended_pairs.append(new_pair)

    def __len__(self):
        return len(self.extended_pairs)

    def __getitem__(self, idx):
        pair = self.extended_pairs[idx]
        img_path = pair['image']
        mask_path = pair['label']
        max_mask_idx = pair['chosen_mask_idx']
        tumor_size = pair['tumor_size']

        img_nii = nib.load(img_path).get_fdata()
        mask_nii = nib.load(mask_path).get_fdata()

        def fix_slice_axis(data):
            shape = data.shape
            min_axis = np.argmin(shape)
            if min_axis == 0:
                return data.transpose(1, 2, 0)
            elif min_axis == 1:
                return data.transpose(0, 2, 1)
            return data

        img_nii = fix_slice_axis(img_nii)
        mask_nii = fix_slice_axis(mask_nii)

        num_slices_mask = mask_nii.shape[2]
        num_slices_img = img_nii.shape[2]
        z_ratio = num_slices_img / num_slices_mask if num_slices_mask > 0 else 1.0
        max_img_idx = int(max_mask_idx * z_ratio)
        max_img_idx = min(max_img_idx, num_slices_img - 1)

        img_slice = img_nii[:, :, max_img_idx]
        mask_slice = mask_nii[:, :, max_mask_idx]

        img_slice = np.clip(img_slice, -1000, 400)
        img_slice = (img_slice + 1000) / 1400

        img_tensor = torch.from_numpy(img_slice).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_slice).float().unsqueeze(0)

        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False
        ).squeeze(0)
        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0), size=self.target_size, mode='nearest'
        ).squeeze(0)

        return img_tensor, mask_tensor, pair['name'], tumor_size


def build_valid_pairs(images_dir, labels_dir):
    all_images = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
    valid_pairs = []
    for f in all_images:
        img_path = os.path.join(images_dir, f)
        label_path = os.path.join(labels_dir, f)
        if os.path.exists(label_path):
            valid_pairs.append({'image': img_path, 'label': label_path, 'name': f})
    return valid_pairs


def compute_metric_dict(preds, masks):
    # バッチ内の「1件ごと」のスコアをリストで返すように修正
    metrics = {'DICE': [], 'IoU': [], 'ACC': [], 'REC': [], 'PRE': []}

    for p, t in zip(preds, masks):
        p = p.cpu().numpy().flatten()
        t = t.numpy().flatten()

        p_score = precision_score(t, p, zero_division=0)
        r_score = recall_score(t, p, zero_division=0)
        i_score = jaccard_score(t, p, zero_division=0)
        a_score = accuracy_score(t, p)

        if np.sum(t) == 0 and np.sum(p) == 0:
            d_score = 1.0
            i_score = 1.0
        else:
            d_score = (2 * i_score) / (i_score + 1) if i_score > 0 else 0.0

        metrics['DICE'].append(d_score)
        metrics['IoU'].append(i_score)
        metrics['ACC'].append(a_score)
        metrics['REC'].append(r_score)
        metrics['PRE'].append(p_score)

    return metrics  # Meanをとらず、各サンプルのリストをそのまま返す


def evaluate_model(model, loader, device):
    model.eval()
    sizes = []
    metrics = {m: [] for m in ['DICE', 'IoU', 'ACC', 'REC', 'PRE']}

    with torch.no_grad():
        for images, masks, _, tumor_sizes in loader:
            images = images.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            batch_metrics = compute_metric_dict(preds, masks)
            for key in metrics:
                # extend を使ってサンプル単位のスコアを平坦に追加（計211要素になる）
                metrics[key].extend(batch_metrics[key])

            sizes.extend([int(s) for s in tumor_sizes])

    return np.array(sizes), {k: np.array(v) for k, v in metrics.items()}


if __name__ == '__main__':
    images_dir = '/workspace/NSCLC_NIfTI2/imagesTr'
    labels_dir = '/workspace/NSCLC_NIfTI2/labelsTr/Neoplasm_Primary'

    valid_pairs = build_valid_pairs(images_dir, labels_dir)
    test_dataset = NSCLCNiftiDataset(valid_pairs, is_train=False)
    test_sizes = np.array([test_dataset[i][3] for i in range(len(test_dataset))])

    # large model の閾値で抽出
    threshold = np.median(test_sizes)
    test_large_pairs = [p for p in test_dataset.extended_pairs if p['tumor_size'] >= threshold]
    test_large_ds = NSCLCNiftiDataset([], is_train=False)
    test_large_ds.extended_pairs = test_large_pairs
    test_loader_large = DataLoader(test_large_ds, batch_size=8, shuffle=False)

    fold_weight_configs = {
        'Fold 1': '/workspace/checkpoints_large_fold1/model_epoch_250.pth',
        'Fold 2': '/workspace/checkpoints_large_fold2/model_epoch_250.pth',
        'Fold 3': '/workspace/checkpoints_large_fold3/model_epoch_250.pth',
        'Fold 4': '/workspace/checkpoints_large_fold4/model_epoch_250.pth',
        'Fold 5': '/workspace/checkpoints_large_fold5/model_epoch_250.pth',
    }

    output_dir = '/workspace/output/fold_large_comparison'
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet_Attention_Transformer_Multiscale(n_channels=1, n_classes=1).to(device)

    metrics_list = ['DICE', 'IoU', 'ACC', 'REC', 'PRE']
    fold_results = {}

    for fold_name, weight_path in fold_weight_configs.items():
        if not os.path.exists(weight_path):
            print(f'[Warning] {fold_name} weight not found: {weight_path}')
            continue

        model.load_state_dict(torch.load(weight_path, map_location=device))
        sizes, scores = evaluate_model(model, test_loader_large, device)
        fold_results[fold_name] = {'sizes': sizes, 'scores': scores}

        print(f'\n{fold_name}')
        for metric_name in metrics_list:
            print(f'  {metric_name}: {np.mean(scores[metric_name]):.4f}')

    if not fold_results:
        raise FileNotFoundError('No valid large-model fold weights were found.')

    all_sizes = np.concatenate([v['sizes'] for v in fold_results.values()])
    bin_edges = np.linspace(all_sizes.min(), all_sizes.max(), 31)

    for metric_name in metrics_list:
        plt.figure(figsize=(12, 7))

        for fold_name in fold_results:
            sizes = fold_results[fold_name]['sizes']
            scores = fold_results[fold_name]['scores'][metric_name]

            mean_vals = []
            for j in range(len(bin_edges) - 1):
                low, high = bin_edges[j], bin_edges[j + 1]
                if j == len(bin_edges) - 2:
                    mask = (sizes >= low) & (sizes <= high)
                else:
                    mask = (sizes >= low) & (sizes < high)

                if np.any(mask):
                    mean_vals.append(np.mean(scores[mask]))
                else:
                    mean_vals.append(np.nan)

            x = np.arange(len(mean_vals))
            plt.plot(x, mean_vals, marker='o', linewidth=2, label=fold_name)

        tick_labels = [f'{int(bin_edges[i])}-{int(bin_edges[i + 1])}' for i in range(len(bin_edges) - 1)]
        plt.xticks(np.arange(len(tick_labels)), tick_labels, rotation=45)
        plt.xlabel('Tumor Size (Pixels on Max Slice)')
        plt.ylabel(f'Mean {metric_name} Score')
        plt.title(f'Large Model Fold Comparison: {metric_name}')
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'large_fold_{metric_name}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f'Saved: {save_path}')

    print('\nDone.')