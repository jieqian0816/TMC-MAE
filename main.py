import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, add_self_loops
import warnings

# torch_sparse is not available on this Windows environment. Use a dense fallback
# for the adjacency matrix by building a torch.sparse_coo_tensor instead.
try:
    from torch_sparse import SparseTensor
except Exception:
    class SparseTensor:
        """Pure-Python fallback that wraps a torch.sparse.Tensor."""
        def __init__(self, row, col, value=None, sparse_sizes=None, is_sorted=False, trust_data=False):
            self.row = row
            self.col = col
            self.value = value
            self.sparse_sizes = sparse_sizes
            self._build(row, col, value, sparse_sizes)

        def _build(self, row, col, value, sparse_sizes):
            if value is None:
                values = torch.ones(row.size(0), dtype=torch.float32, device=row.device)
            else:
                values = value
            if sparse_sizes is None:
                sparse_sizes = (int(row.max()) + 1, int(col.max()) + 1)
            self._mat = torch.sparse_coo_tensor(
                torch.stack([row, col], dim=0), values, size=sparse_sizes, device=row.device
            ).coalesce()

        @classmethod
        def from_edge_index(cls, edge_index, edge_attr=None, sparse_sizes=None, is_sorted=False, trust_data=False):
            return cls(edge_index[0], edge_index[1], edge_attr, sparse_sizes, is_sorted, trust_data)

        def t(self):
            self._mat = self._mat.t().coalesce()
            return self

        def matmul(self, other):
            return torch.sparse.mm(self._mat, other)

        def __mul__(self, other):
            return torch.sparse.mm(self._mat, other)

        def __repr__(self):
            return f"SparseTensor(fallback, shape={self._mat.shape})"

        def to_torch_sparse_tensor(self):
            """Return the wrapped torch.sparse.Tensor (convert COO to CSR for PyG)."""
            return self._mat.to_sparse_csr()

        def __getattr__(self, name):
            # PyG may look for SparseTensor attributes; delegate to the torch sparse tensor.
            return getattr(self._mat, name)
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from operator import truediv

from model import GCN, MagEncoder, FeatureFusion, FeatPredictor, MagnitudePredictor
from model import CNN2D, SimpleMLP, CNN3D, SSFTTnet, SVMWrapper
from train_test_split import loadData, ss_create_data_loader, supervised_test
from get_cls_map import get_cls_map


def AA_andEachClassAccuracy(confusion_matrix):
    """Compute per-class accuracy and average accuracy (AA)."""
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def acc_reports(y_test, y_pred_test):
    """Compute OA, confusion matrix, per-class accuracy, AA and Kappa."""
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)
    return oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100


def construct_edges_and_features(HSI_data):
    """
    Build a 4-neighborhood graph and flatten node features from the HSI cube.
    """
    features = []
    edges = []
    H, W, B = HSI_data.shape
    for i in range(H):
        for j in range(W):
            node_index = i * W + j
            features.append(HSI_data[i, j, :])
            if i > 0:
                edges.append((node_index, node_index - W))
            if i < H - 1:
                edges.append((node_index, node_index + W))
            if j > 0:
                edges.append((node_index, node_index - 1))
            if j < W - 1:
                edges.append((node_index, node_index + 1))
    return features, edges


def self_supervised_train(model, mag_encoder, fusion_dnn, feat_predictor, mag_predictor,
                          x, node_mag, edge_index, optimizer, args, mask_token, mag_mask_token):
    """
    One training step of the dual-branch masked autoencoder.

    Randomly masks nodes according to the magnitude-guided curriculum, then
    reconstructs both the masked spectral features and the masked magnitude values.
    """
    model.train()
    mag_encoder.train()
    fusion_dnn.train()
    feat_predictor.train()
    mag_predictor.train()
    optimizer.zero_grad()

    total_nodes = x.size(0)
    num_node_mask = int(total_nodes * args.mask_ratio)

    # Magnitude-guided node masking
    if args.use_magnitude_mask:
        mask_weights = node_mag.squeeze().clone()
        weight_min = mask_weights.min()
        weight_max = mask_weights.max()
        if weight_max > weight_min:
            mask_weights = (mask_weights - weight_min) / (weight_max - weight_min)
        mix_ratio = getattr(args, 'mask_mix_ratio', 0.5)
        uniform_weights = torch.ones_like(mask_weights)
        final_weights = mix_ratio * mask_weights + (1.0 - mix_ratio) * uniform_weights
        masked_nodes = torch.multinomial(final_weights, num_node_mask, replacement=False)
    else:
        perm_node = torch.randperm(total_nodes, device=x.device)
        masked_nodes = perm_node[:num_node_mask]

    # Mask spectral input features
    x_masked = x.clone()
    if len(masked_nodes) > 0:
        x_masked[masked_nodes] = mask_token.expand(len(masked_nodes), -1)

    # Full graph adjacency
    full_edges_self, _ = add_self_loops(edge_index, num_nodes=x.size(0))
    adj_node_task = SparseTensor.from_edge_index(full_edges_self, sparse_sizes=(x.size(0), x.size(0))).t().to_torch_sparse_coo_tensor()

    # Spectral encoding
    h_node = model(x_masked, adj_node_task)
    h_spectral = torch.cat(h_node, dim=-1)

    # Dual-branch fusion
    if args.use_dual_branch:
        h_mag_full = mag_encoder(node_mag)
        h_mag_masked = h_mag_full.clone()
        if len(masked_nodes) > 0:
            h_mag_masked[masked_nodes] = mag_mask_token.expand(len(masked_nodes), -1)
        h_fused = fusion_dnn(h_spectral, h_mag_masked)
    else:
        h_fused = h_spectral

    # Reconstruction losses
    loss = torch.tensor(0.0, device=x.device)
    loss_feat_val, loss_mag_val = 0.0, 0.0

    if len(masked_nodes) > 0:
        feat_pred = feat_predictor(h_fused[masked_nodes])
        loss_feat = F.mse_loss(feat_pred, x[masked_nodes])
        loss_feat_val = loss_feat.item()
        loss = loss_feat

        if args.use_dual_branch:
            mag_pred_masked = mag_predictor(h_fused[masked_nodes])
            loss_mag = F.smooth_l1_loss(mag_pred_masked, node_mag[masked_nodes], beta=0.1)
            loss_mag = args.weight_loss_mag * loss_mag
            loss_mag_val = loss_mag.item()
            loss = loss_feat + loss_mag

    if loss.requires_grad:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if args.use_dual_branch:
            torch.nn.utils.clip_grad_norm_(mag_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(fusion_dnn.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(mag_predictor.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(feat_predictor.parameters(), 1.0)
        optimizer.step()

    return loss.item(), loss_feat_val, loss_mag_val


def supervised_train(train_loader, val_loader, num_classes, args, device, in_chans):
    """
    Train a downstream patch classifier with early stopping on the validation set.

    Args:
        train_loader: DataLoader for training (strictly disjoint from test).
        val_loader:   DataLoader for validation (subset of training set).
        num_classes:  Number of target classes.
        args:         Argument namespace.
        device:       Torch device.
        in_chans:     Number of input channels.

    Returns:
        net: Trained model (best weights restored from validation).
        device: Torch device.
    """
    if args.classifier == 'SVM':
        net = SVMWrapper(num_classes=num_classes)
        net.train_from_loader(train_loader)
        return net, device

    if args.classifier == 'CNN2D':
        net = CNN2D(in_chans=in_chans, num_classes=num_classes).to(device)
    elif args.classifier == 'MLP':
        net = SimpleMLP(in_chans=in_chans, patch_size=args.patch_size, num_classes=num_classes).to(device)
    elif args.classifier == 'CNN3D':
        net = CNN3D(in_chans=in_chans, num_classes=num_classes).to(device)
    elif args.classifier == 'Transformer':
        net = SSFTTnet(in_chans=in_chans, patch_size=args.patch_size, num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unsupported classifier: {args.classifier}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # Early stopping parameters
    patience = getattr(args, 'early_stopping_patience', 20)
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    best_state_dict = None

    for epoch in range(args.classifier_epochs):
        # ---- Training phase ----
        net.train()
        train_losses = []
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            outputs = net(data)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses) if train_losses else 0.0

        # ---- Validation phase ----
        if val_loader is not None and len(val_loader) > 0:
            net.eval()
            val_losses = []
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    outputs = net(data)
                    loss = criterion(outputs, target)
                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses) if val_losses else 0.0

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                best_state_dict = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                epochs_no_improve += 1

            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f'  Epoch [{epoch+1:03d}/{args.classifier_epochs}] | '
                      f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | '
                      f'Best: {best_val_loss:.4f} (Epoch {best_epoch+1})')

            if epochs_no_improve >= patience:
                print(f'  [Early Stopping] No improvement for {patience} epochs. '
                      f'Best val loss: {best_val_loss:.4f} at epoch {best_epoch+1}.')
                break
        else:
            # No validation set — just print training loss
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f'  Epoch [{epoch+1:03d}/{args.classifier_epochs}] | Train Loss: {avg_train_loss:.4f}')

    # Restore best model weights
    if best_state_dict is not None:
        net.load_state_dict(best_state_dict)
        print(f'  [Restored] Best model from epoch {best_epoch+1} (Val Loss: {best_val_loss:.4f})')

    return net, device


def extract_embeddings(model, mag_encoder, fusion_dnn, mag_predictor, x, node_mag, edge_index, args):
    """Extract fused node embeddings after pre-training."""
    model.eval()
    if args.use_dual_branch:
        mag_encoder.eval()
        fusion_dnn.eval()
        mag_predictor.eval()

    with torch.no_grad():
        full_edges_self, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        full_adj_t = SparseTensor.from_edge_index(full_edges_self, sparse_sizes=(x.size(0), x.size(0))).t().to_torch_sparse_coo_tensor()

        h_full_spec = model(x, full_adj_t)
        h_spec_repr = torch.cat(h_full_spec, dim=-1)

        if args.use_dual_branch:
            h_mag_repr = mag_encoder(node_mag, full_adj_t)
            fused_tensor = fusion_dnn(h_spec_repr, h_mag_repr)
        else:
            fused_tensor = h_spec_repr

    return fused_tensor.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='TMC-MAE: Topological Magnitude Connectivity Masked Autoencoder for HSI')

    # Dataset
    parser.add_argument('--dataset', type=str, default='IP', choices=['IP', 'LK', 'HC', 'HH'])
    parser.add_argument('--data_root', type=str,
                        default=r'C:\PyCharm\Project\HSI_Project\HGCN-MHF-main\datasets',
                        help='Root directory of the HSI datasets.')
    parser.add_argument('--mag_dir', type=str, default='./magnitude_maps',
                        help='Directory containing pre-computed magnitude maps.')
    parser.add_argument('--mag_path', type=str, default=None,
                        help='Explicit path to a magnitude map. If provided, overrides mag_dir.')

    # Data splits — strictly separated
    parser.add_argument('--test_ratio', type=float, default=0.9,
                        help='Fraction of labeled pixels held out as the test set (strictly excluded from training).')
    parser.add_argument('--val_ratio', type=float, default=0.00,
                        help='Fraction of training pixels used for validation (early stopping).')
    parser.add_argument('--random_state', type=int, default=345,
                        help='Random seed for reproducible train/val/test splits.')
    parser.add_argument('--early_stopping_patience', type=int, default=50,
                        help='Number of epochs with no val-loss improvement before stopping.')

    # PCA / patch
    parser.add_argument('--pca_components', type=int, default=30)
    parser.add_argument('--patch_size', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=64)

    # Magnitude hyper-parameters
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--window_size', type=int, default=7)

    # Device and runs
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_run', type=int, default=3)

    # GCN encoder
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--s2gae_dropout', type=float, default=0.5)
    parser.add_argument('--s2gae_lr', type=float, default=0.0005)
    parser.add_argument('--s2gae_epochs', type=int, default=100)

    # Masking and reconstruction
    parser.add_argument('--mask_ratio', type=float, default=0.3)
    parser.add_argument('--weight_loss_mag', type=float, default=1.0)
    parser.add_argument('--use_dual_branch', action='store_true', default=True)
    parser.add_argument('--use_magnitude_mask', action='store_true', default=True)
    parser.add_argument('--mask_mix_ratio', type=float, default=0.5)
    parser.add_argument('--fusion_strategy', type=str, default='SPADE', choices=['SPADE', 'Concat', 'Add'])

    # Downstream classifier
    parser.add_argument('--classifier', type=str, default='CNN2D',
                        choices=['MLP', 'CNN2D', 'CNN3D', 'Transformer', 'SVM'])
    parser.add_argument('--classifier_epochs', type=int, default=100)

    args = parser.parse_args()

    # Determine magnitude map path
    if args.mag_path is None:
        mag_filename = f'magnitude_{args.dataset}_a{args.alpha:.1f}_b{args.beta:.1f}.npy'
        args.mag_path = os.path.join(args.mag_dir, args.dataset, mag_filename)

    if not os.path.exists(args.mag_path):
        raise FileNotFoundError(
            f"Magnitude map not found at {args.mag_path}. "
            "Please run magnitude.py first to generate it."
        )

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print(f"Dataset: {args.dataset} | Classifier: {args.classifier} | Mode: TMC-MAE")
    print(f"Magnitude -> Alpha: {args.alpha:.1f}, Beta: {args.beta:.1f}")
    print(f"Train/Test split: test_ratio={args.test_ratio}, val_ratio={args.val_ratio}")
    print(f"Device: {device}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data and preprocess
    # ------------------------------------------------------------------
    HSI_data, HSI_labels, magnitude_map = loadData(args.dataset, args.mag_path)
    n_rows, n_cols, n_bands = HSI_data.shape

    node_mag_np = magnitude_map.flatten().astype(np.float32)
    node_mag = torch.tensor(node_mag_np, dtype=torch.float32).unsqueeze(-1).to(device)

    HSI_data = HSI_data.reshape((n_rows * n_cols, n_bands))
    pca = PCA(n_components=args.pca_components)
    HSI_data = pca.fit_transform(HSI_data)

    scaler = MinMaxScaler()
    HSI_data = scaler.fit_transform(HSI_data)
    HSI_data = HSI_data.reshape((n_rows, n_cols, args.pca_components))

    # ------------------------------------------------------------------
    # 2. Build 4-neighborhood graph
    # ------------------------------------------------------------------
    features, edges = construct_edges_and_features(HSI_data)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)

    x = torch.tensor(np.array(features), dtype=torch.float).to(device)
    y_np = HSI_labels.reshape(-1)
    edge_index = edge_index.to(device)

    # ------------------------------------------------------------------
    # 3. Initialize model components
    # ------------------------------------------------------------------
    model = GCN(x.shape[1], args.hidden_channels, args.hidden_channels,
                args.num_layers, args.s2gae_dropout).to(device)
    actual_gcn_layers = len(model.convs)
    D_spec = args.hidden_channels * actual_gcn_layers

    mag_encoder = MagEncoder(in_channels=1, out_channels=D_spec,
                             n_rows=n_rows, n_cols=n_cols).to(device)
    fusion_dnn = FeatureFusion(d_model=D_spec, strategy=args.fusion_strategy, dropout=0.1).to(device)
    feat_predictor = FeatPredictor(in_channels=D_spec, out_channels=x.shape[1]).to(device)
    mag_predictor = MagnitudePredictor(in_channels=D_spec, out_channels=1).to(device)

    # Learnable mask tokens
    mask_token = nn.Parameter(torch.zeros(1, x.shape[1], device=device))
    nn.init.xavier_uniform_(mask_token)
    mag_mask_token = nn.Parameter(torch.zeros(1, D_spec, device=device))
    nn.init.xavier_uniform_(mag_mask_token)

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': mag_encoder.parameters()},
        {'params': fusion_dnn.parameters()},
        {'params': feat_predictor.parameters()},
        {'params': mag_predictor.parameters()},
        {'params': mask_token},
        {'params': mag_mask_token}
    ], lr=args.s2gae_lr)

    # ------------------------------------------------------------------
    # 4. Self-supervised pre-training
    # ------------------------------------------------------------------
    print("\n[Start pre-training TMC-MAE ...]")
    t1 = time.time()

    best_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    save_path = f"best_autoencoder_{args.dataset}.pth"

    for epoch in range(1, 1 + args.s2gae_epochs):
        loss_total, loss_feat, loss_mag = self_supervised_train(
            model, mag_encoder, fusion_dnn, feat_predictor, mag_predictor,
            x, node_mag, edge_index, optimizer, args, mask_token, mag_mask_token
        )

        if loss_total < best_loss:
            best_loss = loss_total
            best_epoch = epoch
            best_model_state = {
                'epoch': epoch,
                'total_loss': loss_total,
                'feature_loss': loss_feat,
                'magnitude_loss': loss_mag,
                'model_state_dict': model.state_dict(),
                'mag_encoder_state_dict': mag_encoder.state_dict(),
                'fusion_dnn_state_dict': fusion_dnn.state_dict(),
                'feat_predictor_state_dict': feat_predictor.state_dict(),
                'mag_predictor_state_dict': mag_predictor.state_dict(),
                'mask_token': mask_token.detach().clone(),
                'mag_mask_token': mag_mask_token.detach().clone(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args)
            }
            torch.save(best_model_state, save_path)
            print(f"[Save] Best model updated at Epoch {epoch:03d} | Total Loss: {loss_total:.4f}")

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch: {epoch:03d} | Total: {loss_total:.4f} | "
                  f"Spec: {loss_feat:.4f} | Mag: {loss_mag:.4f} | "
                  f"Best: {best_loss:.4f} (Epoch {best_epoch})")

    print(f"Pre-training finished in {time.time() - t1:.2f} seconds.")
    print(f"Best model saved at Epoch {best_epoch} with loss {best_loss:.4f}")

    # ------------------------------------------------------------------
    # 5. Load best model and extract embeddings
    # ------------------------------------------------------------------
    print(f"\n[Loading best model from {save_path} ...]")
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    mag_encoder.load_state_dict(checkpoint['mag_encoder_state_dict'])
    fusion_dnn.load_state_dict(checkpoint['fusion_dnn_state_dict'])
    feat_predictor.load_state_dict(checkpoint['feat_predictor_state_dict'])
    mag_predictor.load_state_dict(checkpoint['mag_predictor_state_dict'])
    mask_token.data = checkpoint['mask_token'].to(device)
    mag_mask_token.data = checkpoint['mag_mask_token'].to(device)
    print(f"Loaded best model from Epoch {checkpoint['epoch']} with loss {checkpoint['total_loss']:.4f}")

    features_extracted = extract_embeddings(
        model, mag_encoder, fusion_dnn, mag_predictor, x, node_mag, edge_index, args
    )
    print(f"Extracted embeddings shape: {features_extracted.shape}")

    # ------------------------------------------------------------------
    # 6. Downstream classification (strict train/test separation)
    # ------------------------------------------------------------------
    print(f"\n[Start downstream evaluation using {args.classifier} ...]")
    print(f"  -> Training on train_loader, validating on val_loader, "
          f"evaluating on HELD-OUT test_loader")
    oa_list, aa_list, kappa_list = [], [], []
    best_oa = 0.0
    best_net = None
    best_all_loader = None
    best_all_loader_whole = None
    best_y = None

    for num in range(args.num_run):
        # Unpack 8 values: now includes val_loader
        (train_loader, test_loader, val_loader, y_all, index,
         all_data_loader, all_data_loader_whole, y) = \
            ss_create_data_loader(features_extracted, y_np, n_rows, n_cols, args)

        num_classes = int(y.max().item() + 1)

        print(f"\n--- Run [{num + 1}/{args.num_run}] ---")
        print(f"  Train: {len(train_loader.dataset)} | "
              f"Val: {len(val_loader.dataset) if val_loader else 0} | "
              f"Test (held-out): {len(test_loader.dataset)}")

        tic1 = time.time()
        net, device = supervised_train(
            train_loader, val_loader, num_classes, args, device,
            in_chans=args.pca_components
        )
        toc1 = time.time()

        # Evaluate ONLY on the strictly held-out test set
        tic2 = time.time()
        y_pred_test, y_test = supervised_test(device, net, test_loader)
        toc2 = time.time()

        each_oa, confusion, each_acc, each_aa, each_kappa = acc_reports(y_test, y_pred_test)

        oa_list.append(each_oa)
        aa_list.append(each_aa)
        kappa_list.append(each_kappa)

        if each_oa > best_oa:
            best_oa = each_oa
            best_net = net
            best_all_loader = all_data_loader
            best_all_loader_whole = all_data_loader_whole
            best_y = y

        print(f"  Run [{num + 1}/{args.num_run}] | "
              f"OA: {each_oa:.2f}% | AA: {each_aa:.2f}% | Kappa: {each_kappa:.2f}% | "
              f"Train: {toc1-tic1:.1f}s | Test: {toc2-tic2:.1f}s")

    mean_oa, std_oa = np.mean(oa_list), np.std(oa_list)
    mean_aa, std_aa = np.mean(aa_list), np.std(aa_list)
    mean_kappa, std_kappa = np.mean(kappa_list), np.std(kappa_list)

    print("\n" + "=" * 60)
    print(f"Final Results on HELD-OUT Test Set ({args.num_run} Runs):")
    print(f"  OA:    {mean_oa:.2f} +- {std_oa:.2f}%")
    print(f"  AA:    {mean_aa:.2f} +- {std_aa:.2f}%")
    print(f"  Kappa: {mean_kappa:.2f} +- {std_kappa:.2f}%")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 7. Logging and classification maps
    # ------------------------------------------------------------------
    log_file = f"Training_Logs_{args.dataset}.txt"
    with open(log_file, "a") as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        f.write(f"STRICT train/test split: test_ratio={args.test_ratio}, "
                f"val_ratio={args.val_ratio}, random_state={args.random_state}\n")
        f.write(f"Arguments: {vars(args)}\n")
        f.write("-" * 50 + "\n")
        for idx, (oa, aa, ka) in enumerate(zip(oa_list, aa_list, kappa_list)):
            f.write(f"Run {idx + 1} -> OA: {oa:.4f}%, AA: {aa:.4f}%, Kappa: {ka:.4f}%\n")
        f.write("-" * 50 + "\n")
        f.write(f"Average -> OA: {mean_oa:.4f} +- {std_oa:.4f}%, "
                f"AA: {mean_aa:.4f} +- {std_aa:.4f}%, "
                f"Kappa: {mean_kappa:.4f} +- {std_kappa:.4f}%\n")
        f.write("=" * 50 + "\n")
    print(f"[Log] Metrics and configuration saved to: {log_file}")

    print("\n[Visual] Generating classification maps for the best run ...")
    get_cls_map(best_net, device, best_all_loader, best_all_loader_whole, best_y, best_oa, args.dataset)


if __name__ == '__main__':
    main()
