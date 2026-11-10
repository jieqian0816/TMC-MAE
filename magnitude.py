import os
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data normalization
# ---------------------------------------------------------------------------
def min_max_normalize(data):
    """
    Apply channel-wise min-max normalization to an HSI cube of shape (H, W, B).
    """
    data_normalized = np.zeros_like(data, dtype=np.float32)
    H, W, B = data.shape
    for b in range(B):
        channel = data[:, :, b]
        min_val = channel.min()
        max_val = channel.max()
        if max_val > min_val:
            data_normalized[:, :, b] = (channel - min_val) / (max_val - min_val)
        else:
            data_normalized[:, :, b] = 0.0
    return data_normalized


def normalize_2d(data):
    """
    Min-max normalize a 2-D array to [0, 1].
    """
    min_val = data.min()
    max_val = data.max()
    if max_val > min_val:
        return (data - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(data)


# ---------------------------------------------------------------------------
# Magnitude map computation
# ---------------------------------------------------------------------------
def compute_local_weight_map(hsi_cube, window_size=7, alpha=1.0, beta=1.0):
    """
    Compute the Magnitude (TMCP) weight map for an HSI cube.

    For each pixel, a local window is extracted. Spectral and spatial distances
    are fused into a mixed distance matrix, from which a locally linear weighting
    vector is solved. The magnitude value of the central pixel is the weight
    assigned to itself.

    Args:
        hsi_cube: HSI cube of shape (H, W, B).
        window_size: size of the local square window.
        alpha: spectral distance weight.
        beta: spatial distance weight.
    Returns:
        2-D magnitude map of shape (H, W) normalized to [0, 1].
    """
    H, W, B = hsi_cube.shape
    weight_map = np.full((H, W), np.nan, dtype=np.float32)
    half_win = window_size // 2

    for i in tqdm(range(H), desc=f"Computing Magnitude map (alpha={alpha:.1f}, beta={beta:.1f})", leave=False):
        for j in range(W):
            i_start = max(0, i - half_win)
            i_end = min(H, i + half_win + 1)
            j_start = max(0, j - half_win)
            j_end = min(W, j + half_win + 1)

            actual_window_h = i_end - i_start
            actual_window_w = j_end - j_start

            if actual_window_h < 3 or actual_window_w < 3:
                weight_map[i, j] = 0.0
                continue

            window_features = hsi_cube[i_start:i_end, j_start:j_end]
            window_features_flat = window_features.reshape(-1, B)
            n_pixels = window_features_flat.shape[0]

            center_i_in_window = i - i_start
            center_j_in_window = j - j_start
            center_idx = center_i_in_window * actual_window_w + center_j_in_window

            # Spectral distance (cosine-based)
            norms = np.linalg.norm(window_features_flat, axis=1, keepdims=True)
            valid_norms = norms > 1e-8
            window_features_norm = np.where(valid_norms, window_features_flat / norms, 0.0)
            spectral_dist = 1.0 - np.dot(window_features_norm, window_features_norm.T)
            spectral_dist_norm = spectral_dist / 2.0

            # Spatial distance (sum of Euclidean distances to center)
            spatial_dist = np.zeros((n_pixels, n_pixels), dtype=np.float32)
            for p in range(n_pixels):
                row1, col1 = p // actual_window_w, p % actual_window_w
                for q in range(p + 1, n_pixels):
                    row2, col2 = q // actual_window_w, q % actual_window_w
                    dist_val = (
                        ((row1 - center_i_in_window) ** 2 + (col1 - center_j_in_window) ** 2) ** 0.5 +
                        ((row2 - center_i_in_window) ** 2 + (col2 - center_j_in_window) ** 2) ** 0.5
                    )
                    spatial_dist[p, q] = dist_val
                    spatial_dist[q, p] = dist_val

            max_spat_dist = 2 * ((half_win ** 2 + half_win ** 2) ** 0.5)
            if max_spat_dist > 0:
                spatial_dist_norm = spatial_dist / max_spat_dist
            else:
                spatial_dist_norm = np.zeros_like(spatial_dist)

            # Mixed distance and similarity matrix
            mixed_dist = alpha * spectral_dist_norm + beta * spatial_dist_norm
            Z = np.exp(-mixed_dist)

            ones = np.ones(n_pixels)
            try:
                w = np.linalg.lstsq(Z, ones, rcond=1e-6)[0]
            except Exception:
                w = ones / n_pixels

             # Use the absolute center weight before image-wide min-max normalization.
            weight_map[i, j] = np.abs(w[center_idx])

    weight_map = np.nan_to_num(weight_map, nan=0.0)
    return normalize_2d(weight_map)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def get_custom_gt_cmap():
    """
    Colormap consistent with the RGB colors used in get_cls_map.py.
    """
    color_rgb_dict = {
        0: [0, 0, 0],
        1: [147, 67, 46], 2: [0, 0, 255], 3: [255, 100, 0], 4: [0, 255, 123],
        5: [164, 75, 155], 6: [101, 174, 255], 7: [118, 254, 172], 8: [60, 91, 112],
        9: [255, 255, 0], 10: [255, 255, 125], 11: [255, 0, 255], 12: [100, 0, 255],
        13: [0, 172, 254], 14: [0, 255, 0], 15: [171, 175, 80], 16: [101, 193, 60],
        17: [255, 102, 102], 18: [255, 158, 204], 19: [204, 204, 255], 20: [255, 204, 153],
        21: [153, 255, 153], 22: [0, 204, 204], 23: [204, 153, 255], 24: [153, 76, 0]
    }
    max_class = max(color_rgb_dict.keys())
    colors = np.zeros((max_class + 1, 3))
    for k, v in color_rgb_dict.items():
        colors[k] = np.array(v) / 255.0
    return ListedColormap(colors), max_class


def plot_magnitude_grid(results, gt_map, dataset_name, save_path):
    """
    Plot a grid of magnitude maps for different alpha/beta ratios alongside
    the ground-truth label map.
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 6, figsize=(28, 10))
    axes = axes.flatten()

    for idx, (alpha, beta, w_map) in enumerate(results):
        ax = axes[idx]
        ax.imshow(w_map, cmap='coolwarm', vmin=0, vmax=1)
        ax.set_title(f'Spatial: {beta:.1f} | Spectral: {alpha:.1f}', fontweight='bold', fontsize=20, pad=10)
        ax.axis('off')

    ax_gt = axes[11]
    custom_cmap, max_class = get_custom_gt_cmap()
    ax_gt.imshow(gt_map, cmap=custom_cmap, vmin=0, vmax=max_class, interpolation='nearest')
    ax_gt.set_title(f'GT ({dataset_name})', fontweight='bold', fontsize=20, pad=15)
    ax_gt.axis('off')

    fig.tight_layout(h_pad=3.0, w_pad=1.0)
    plt.savefig(save_path, format='png', dpi=400, bbox_inches='tight')
    plt.close()
    print(f"[{dataset_name}] Magnitude grid visualization saved to: {save_path}")


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Compute magnitude maps for supported HSI datasets.')
    parser.add_argument('--dataset', type=str, default='IP', choices=['IP', 'LK', 'HC', 'HH'])
    parser.add_argument('--data_root', type=str,
                        default=r'C:\PyCharm\Project\HSI_Project\HGCN-MHF-main\datasets',
                        help='Root directory that contains the dataset folders.')
    parser.add_argument('--window_size', type=int, default=7)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=0.3)
    parser.add_argument('--save_dir', type=str, default='./magnitude_maps')
    return parser.parse_args()


def build_dataset_paths(data_root, dataset_name):
    """
    Return the data and ground-truth paths for each supported dataset.
    """
    if dataset_name == 'IP':
        data_path = os.path.join(data_root, 'Indian_pines', 'Indian_pines_corrected.mat')
        data_key = 'indian_pines_corrected'
        gt_path = os.path.join(data_root, 'Indian_pines', 'Indian_pines_gt.mat')
        gt_key = 'indian_pines_gt'
    elif dataset_name == 'LK':
        data_path = os.path.join(data_root, 'WHU-Hi', 'WHU-Hi-LongKou', 'WHU_Hi_LongKou.mat')
        data_key = 'WHU_Hi_LongKou'
        gt_path = os.path.join(data_root, 'WHU-Hi', 'WHU-Hi-LongKou', 'WHU_Hi_LongKou_gt.mat')
        gt_key = 'WHU_Hi_LongKou_gt'
    elif dataset_name == 'HC':
        data_path = os.path.join(data_root, 'WHU-Hi', 'WHU-Hi-HanChuan', 'WHU_Hi_HanChuan.mat')
        data_key = 'WHU_Hi_HanChuan'
        gt_path = os.path.join(data_root, 'WHU-Hi', 'WHU-Hi-HanChuan', 'WHU_Hi_HanChuan_gt.mat')
        gt_key = 'WHU_Hi_HanChuan_gt'
    elif dataset_name == 'HH':
        data_path = os.path.join(data_root, 'WHU-Hi', 'WHU-Hi-HongHu', 'WHU_Hi_HongHu.mat')
        data_key = 'WHU_Hi_HongHu'
        gt_path = os.path.join(data_root, 'WHU-Hi', 'WHU-Hi-HongHu', 'WHU_Hi_HongHu_gt.mat')
        gt_key = 'WHU_Hi_HongHu_gt'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return data_path, data_key, gt_path, gt_key


if __name__ == "__main__":
    args = parse_args()
    window_size = args.window_size
    os.makedirs(args.save_dir, exist_ok=True)

    dataset_save_dir = os.path.join(args.save_dir, args.dataset)
    os.makedirs(dataset_save_dir, exist_ok=True)

    data_path, data_key, gt_path, gt_key = build_dataset_paths(args.data_root, args.dataset)

    data_mat = sio.loadmat(data_path)
    data = data_mat[data_key]
    gt_mat = sio.loadmat(gt_path)
    gt = gt_mat[gt_key]

    normalized_data = min_max_normalize(data)

    # Generate 11 ratio pairs from (0.0, 1.0) to (1.0, 0.0)
    # ratio_pairs = [(np.round(a, 1), np.round(1.0 - a, 1)) for a in np.linspace(0, 1, 11)]
    ratio_pairs = [(1.0, 1.0)]
    pipeline_results = []

    for alpha, beta in ratio_pairs:
        w_map = compute_local_weight_map(normalized_data, window_size=window_size, alpha=alpha, beta=beta)
        pipeline_results.append((alpha, beta, w_map))

        npy_filename = f'magnitude_{args.dataset}_a{alpha:.1f}_b{beta:.1f}.npy'
        npy_path = os.path.join(dataset_save_dir, npy_filename)
        np.save(npy_path, w_map)

    plot_path = os.path.join(dataset_save_dir,
                             f'Magnitude_Grid_{args.dataset}_{window_size}x{window_size}.png')
    plot_magnitude_grid(pipeline_results, gt, args.dataset, plot_path)
    print(f"All magnitude maps for {args.dataset} saved under {dataset_save_dir}")
