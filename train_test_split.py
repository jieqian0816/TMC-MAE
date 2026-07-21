import os
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch


def loadData(dataset_name='IP', mag_path=None):
    """
    Load hyperspectral data and the corresponding magnitude map.

    Args:
        dataset_name: one of 'IP', 'LK', 'HC', 'HH'.
        mag_path: optional path to a pre-computed .npy magnitude map.
    Returns:
        data: HSI cube (H, W, B).
        labels: ground-truth label map (H, W).
        magnitude_map: 2-D magnitude map (H, W).
    """
    if dataset_name == 'IP':
        data_mat = sio.loadmat(
            r'C:\PyCharm\Project\HSI_Project\HGCN-MHF-main\datasets\Indian_pines\Indian_pines_corrected.mat')
        data = data_mat['indian_pines_corrected']
        gt_mat = sio.loadmat(
            r'C:\PyCharm\Project\HSI_Project\HGCN-MHF-main\datasets\Indian_pines\Indian_pines_gt.mat')
        labels = gt_mat['indian_pines_gt']
    elif dataset_name == 'LK':
        data_mat = sio.loadmat(
            r'C:\Download\Pycharm\Project\HSI_Project\HGCN-MHF-main\datasets\datasets\WHU-Hi\WHU-Hi-LongKou\WHU_Hi_LongKou.mat')
        data = data_mat['WHU_Hi_LongKou']
        gt_mat = sio.loadmat(
            r'C:\Download\Pycharm\Project\HSI_Project\HGCN-MHF-main\datasets\datasets\WHU-Hi\WHU-Hi-LongKou\WHU_Hi_LongKou_gt.mat')
        labels = gt_mat['WHU_Hi_LongKou_gt']
    elif dataset_name == 'HC':
        data_mat = sio.loadmat(
            r'C:\Download\Pycharm\Project\HSI_Project\HGCN-MHF-main\datasets\datasets\WHU-Hi\WHU-Hi-HanChuan\WHU_Hi_HanChuan.mat')
        data = data_mat['WHU_Hi_HanChuan']
        gt_mat = sio.loadmat(
            r'C:\Download\Pycharm\Project\HSI_Project\HGCN-MHF-main\datasets\datasets\WHU-Hi\WHU-Hi-HanChuan\WHU_Hi_HanChuan_gt.mat')
        labels = gt_mat['WHU_Hi_HanChuan_gt']
    elif dataset_name == 'HH':
        data_mat = sio.loadmat(
            r'C:\Download\Pycharm\Project\HSI_Project\HGCN-MHF-main\datasets\datasets\WHU-Hi\WHU-Hi-HongHu\WHU_Hi_HongHu.mat')
        data = data_mat['WHU_Hi_HongHu']
        gt_mat = sio.loadmat(
            r'C:\Download\Pycharm\Project\HSI_Project\HGCN-MHF-main\datasets\datasets\WHU-Hi\WHU-Hi-HongHu\WHU_Hi_HongHu_gt.mat')
        labels = gt_mat['WHU_Hi_HongHu_gt']
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if mag_path is not None and os.path.exists(mag_path):
        magnitude_map = np.load(mag_path)
        print(f"Loaded magnitude map from: {mag_path}")
    else:
        print(f"Warning: magnitude map not found at {mag_path}. Using random initialization.")
        magnitude_map = np.random.rand(data.shape[0], data.shape[1]).astype(np.float32)

    return data, labels, magnitude_map


def ss_loadData(feature, labels, x0, x1, args):
    """
    Reshape flat extracted features and labels back to 2-D image maps.
    """
    x0 = int(x0)
    x1 = int(x1)
    feat_dim = feature.shape[-1] if feature.ndim == 2 else feature.shape[-1]
    data = feature.reshape(x0, x1, feat_dim)
    labels = labels.reshape(x0, x1)
    return data, labels


def supervised_test(device, net, test_loader):
    """
    Evaluate a trained classifier on the test set.
    """
    net.eval()
    y_pred_test = []
    y_test_list = []
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        y_pred_test.append(outputs)
        y_test_list.append(labels.numpy())

    return np.concatenate(y_pred_test), np.concatenate(y_test_list)


def applyPCA(X, numComponents):
    """Apply PCA whitening to the spectral dimension."""
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX).astype(np.float32)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def padWithZeros(X, margin=2):
    """Pad the HSI image with zeros to allow patch extraction at borders."""
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]), dtype=np.float32)
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def get_coordinates(y, removeZeroLabels=True):
    """
    Extract valid pixel coordinates instead of creating huge image cubes in memory.
    """
    rows, cols = y.shape
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    coords_all = np.stack([r.flatten(), c.flatten()], axis=-1)
    labels_all = y.flatten()

    if removeZeroLabels:
        valid_mask = labels_all > 0
        coords_valid = coords_all[valid_mask]
        labels_valid = labels_all[valid_mask] - 1
        return coords_valid, labels_valid, coords_all, labels_all

    return coords_all, labels_all, coords_all, labels_all


def splitTrainTestSet(coords, y, testRatio, randomState=345):
    """
    Split coordinate indices into train and test sets using stratified sampling.

    This ensures:
    - Training and test sets are strictly disjoint (no overlap).
    - Each class is proportionally represented in both sets.
    - The split is reproducible via randomState.
    """
    coords_train, coords_test, y_train, y_test = train_test_split(
        coords, y, test_size=testRatio, random_state=randomState, stratify=y
    )
    unique_classes = np.unique(y).astype(int)
    train_class_counts = {cls: np.sum(y_train == cls) for cls in unique_classes}
    test_class_counts = {cls: np.sum(y_test == cls) for cls in unique_classes}

    print(f"Training set size: {len(coords_train)}")
    print(f"Testing set size:  {len(coords_test)}")
    summary_df = pd.DataFrame({
        'Class No.': unique_classes,
        'Train': [train_class_counts[cls] for cls in unique_classes],
        'Test':  [test_class_counts.get(cls, 0) for cls in unique_classes]
    })
    print(summary_df.to_string(index=False))

    # Verify no overlap between train and test
    train_set = set(map(tuple, coords_train))
    test_set = set(map(tuple, coords_test))
    overlap = train_set & test_set
    assert len(overlap) == 0, f"ERROR: {len(overlap)} pixels overlap between train and test sets!"
    print(f"[Check] Train/Test disjointness verified: 0 overlap (OK)")

    return coords_train, coords_test, y_train, y_test


class HSIDynamicDataset(torch.utils.data.Dataset):
    """
    Dynamic patch extraction dataset: avoids loading the entire cube of patches
    into memory by cropping patches on the fly from the padded image.
    """
    def __init__(self, padded_image, coords, labels, patch_size):
        self.padded_image = padded_image
        self.coords = coords
        self.labels = labels
        self.patch_size = patch_size
        self.margin = patch_size // 2

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        r, c = self.coords[idx]
        r_pad = r + self.margin
        c_pad = c + self.margin

        patch = self.padded_image[
            r_pad - self.margin: r_pad + self.margin + 1,
            c_pad - self.margin: c_pad + self.margin + 1, :
        ]

        # Rearrange from (H, W, C) to (1, C, H, W) and add a batch dimension
        patch = np.transpose(patch, (2, 0, 1))
        patch = np.expand_dims(patch, axis=0)

        return torch.FloatTensor(np.array(patch)), torch.LongTensor([self.labels[idx]])[0]


def ss_create_data_loader(feature, labels, x0, x1, args):
    """
    Create STRICTLY separated train/test data loaders for downstream classification.

    Key design decisions:
    - Train and test sets are strictly disjoint (no pixel overlap).
    - Uses stratified split to ensure each class appears in both sets.
    - test_loader contains ONLY the held-out test pixels (never the training pixels).
    - all_data_loader is kept for generating full-classification maps (visualization only).
    - A val_loader is added for optional validation during training.

    Args:
        feature: Extracted node features from the GNN (N, D).
        labels: Ground-truth labels (N,) or (H, W).
        x0, x1: Spatial dimensions (rows, cols) for reshaping.
        args: Argument namespace containing hyperparameters.

    Returns:
        train_loader: DataLoader for training (strictly disjoint from test).
        test_loader: DataLoader for final evaluation (held-out, never used for training).
        val_loader:  DataLoader for validation (subset of training set).
        y_all:       All valid labels (for reference).
        index:       Non-zero label indices.
        all_data_loader: DataLoader over all labeled pixels (for visualization).
        all_data_loader_whole: DataLoader over the entire image (including background).
        y_out:       Full label map.
    """
    X, y = ss_loadData(feature, labels, x0, x1, args)
    y_out = y
    index = np.nonzero(y.reshape(y.shape[0] * y.shape[1]))[0]

    test_ratio = args.test_ratio
    val_ratio = getattr(args, 'val_ratio', 0.1)  # 10% of training data for validation

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA transformation ... ...')
    X_pca = applyPCA(X, numComponents=args.pca_components)
    print('Data shape after PCA: ', X_pca.shape)

    print('\n... ... Extracting pixel coordinates (memory efficient) ... ...')
    margin = args.patch_size // 2
    X_pca_padded = padWithZeros(X_pca, margin=margin)

    # Get all labeled pixels
    coords_valid, labels_valid, coords_all, labels_all = get_coordinates(y, removeZeroLabels=True)
    print(f'Labeled pixels after removing background: {len(coords_valid)}')

    # ------------------------------------------------------------------
    # Strict Train / Test split (stratified, reproducible)
    # ------------------------------------------------------------------
    print('\n... ... Creating STRICTLY separated train & test data ... ...')
    coords_train, coords_test, ytrain, ytest = splitTrainTestSet(
        coords_valid, labels_valid, test_ratio, randomState=args.random_state
    )

    # ------------------------------------------------------------------
    # Further split train into train / val (stratified)
    # ------------------------------------------------------------------
    if val_ratio > 0:
        coords_train, coords_val, ytrain, yval = train_test_split(
            coords_train, ytrain, test_size=val_ratio,
            random_state=args.random_state, stratify=ytrain
        )
        print(f'Validation set size: {len(coords_val)} (from training set)')
    else:
        coords_val = np.array([], dtype=np.int64).reshape(0, 2)
        yval = np.array([], dtype=np.int64)

    # ------------------------------------------------------------------
    # Sanity checks: verify strict disjointness
    # ------------------------------------------------------------------
    train_set = set(map(tuple, coords_train))
    test_set = set(map(tuple, coords_test))
    val_set = set(map(tuple, coords_val))

    assert len(train_set & test_set) == 0, "Train/Test overlap detected!"
    assert len(train_set & val_set) == 0, "Train/Val overlap detected!"
    assert len(val_set & test_set) == 0, "Val/Test overlap detected!"
    print(f'[Check] Train/Val/Test disjointness verified (OK)')
    print(f'  -> Train: {len(coords_train)} | Val: {len(coords_val)} | Test: {len(coords_test)}')

    # ------------------------------------------------------------------
    # Build datasets and loaders
    # ------------------------------------------------------------------
    trainset = HSIDynamicDataset(X_pca_padded, coords_train, ytrain, args.patch_size)
    valset   = HSIDynamicDataset(X_pca_padded, coords_val,   yval,   args.patch_size) if len(coords_val) > 0 else None
    testset  = HSIDynamicDataset(X_pca_padded, coords_test,  ytest,  args.patch_size)

    # all_data_loader: all labeled pixels -> for full classification map visualization
    all_dataset = HSIDynamicDataset(X_pca_padded, coords_valid, labels_valid, args.patch_size)
    # all_data_loader_whole: entire image (including background) -> for full map
    all_dataset_whole = HSIDynamicDataset(X_pca_padded, coords_all, labels_all, args.patch_size)

    train_loader = DataLoader(
        dataset=trainset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        dataset=testset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False
    )
    if valset is not None and len(valset) > 0:
        val_loader = DataLoader(
            dataset=valset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False
        )
    else:
        val_loader = None

    all_data_loader = DataLoader(
        dataset=all_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False
    )
    all_data_loader_whole = DataLoader(
        dataset=all_dataset_whole, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False
    )

    print(f'\nDataLoader summary:')
    print(f'  Train loader: {len(trainset)} samples, {len(train_loader)} batches')
    print(f'  Val loader:   {len(valset) if valset else 0} samples')
    print(f'  Test loader:  {len(testset)} samples, {len(test_loader)} batches')

    return train_loader, test_loader, val_loader, labels_valid, index, all_data_loader, all_data_loader_whole, y_out


def create_data_loader(args):
    """
    Create STRICTLY separated data loaders directly from the raw HSI cube.

    Same strict separation logic as ss_create_data_loader, but works with
    raw HSI data instead of pre-extracted features.
    """
    mag_path = args.mag_path if hasattr(args, 'mag_path') and args.mag_path else None
    X, y, _ = loadData(args.dataset, mag_path)

    y_out = y
    index = np.nonzero(y.reshape(y.shape[0] * y.shape[1]))[0]

    test_ratio = args.test_ratio
    val_ratio = getattr(args, 'val_ratio', 0.1)

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA transformation ... ...')
    X_pca = applyPCA(X, numComponents=args.pca_components)
    print('Data shape after PCA: ', X_pca.shape)

    print('\n... ... Extracting pixel coordinates (memory efficient) ... ...')
    margin = args.patch_size // 2
    X_pca_padded = padWithZeros(X_pca, margin=margin)

    coords_valid, labels_valid, coords_all, labels_all = get_coordinates(y, removeZeroLabels=True)
    print(f'Labeled pixels after removing background: {len(coords_valid)}')

    # ------------------------------------------------------------------
    # Strict Train / Test split
    # ------------------------------------------------------------------
    print('\n... ... Creating STRICTLY separated train & test data ... ...')
    coords_train, coords_test, ytrain, ytest = splitTrainTestSet(
        coords_valid, labels_valid, test_ratio, randomState=args.random_state
    )

    # ------------------------------------------------------------------
    # Train / Val split
    # ------------------------------------------------------------------
    if val_ratio > 0:
        coords_train, coords_val, ytrain, yval = train_test_split(
            coords_train, ytrain, test_size=val_ratio,
            random_state=args.random_state, stratify=ytrain
        )
        print(f'Validation set size: {len(coords_val)} (from training set)')
    else:
        coords_val = np.array([], dtype=np.int64).reshape(0, 2)
        yval = np.array([], dtype=np.int64)

    # Sanity checks
    train_set = set(map(tuple, coords_train))
    test_set = set(map(tuple, coords_test))
    val_set = set(map(tuple, coords_val))
    assert len(train_set & test_set) == 0, "Train/Test overlap detected!"
    assert len(train_set & val_set) == 0, "Train/Val overlap detected!"
    assert len(val_set & test_set) == 0, "Val/Test overlap detected!"
    print(f'[Check] Train/Val/Test disjointness verified (OK)')

    # Build datasets
    trainset = HSIDynamicDataset(X_pca_padded, coords_train, ytrain, args.patch_size)
    valset   = HSIDynamicDataset(X_pca_padded, coords_val,   yval,   args.patch_size) if len(coords_val) > 0 else None
    testset  = HSIDynamicDataset(X_pca_padded, coords_test,  ytest,  args.patch_size)
    all_dataset = HSIDynamicDataset(X_pca_padded, coords_valid, labels_valid, args.patch_size)
    all_dataset_whole = HSIDynamicDataset(X_pca_padded, coords_all, labels_all, args.patch_size)

    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    val_loader = DataLoader(dataset=valset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False) if valset is not None else None
    all_data_loader = DataLoader(dataset=all_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    all_data_loader_whole = DataLoader(dataset=all_dataset_whole, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    print(f'\nDataLoader summary:')
    print(f'  Train loader: {len(trainset)} samples, {len(train_loader)} batches')
    print(f'  Val loader:   {len(valset) if valset else 0} samples')
    print(f'  Test loader:  {len(testset)} samples, {len(test_loader)} batches')

    return train_loader, test_loader, val_loader, labels_valid, index, all_data_loader, all_data_loader_whole, y_out
