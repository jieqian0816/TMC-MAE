# TMC-MAE

> Topological Magnitude Connectivity Masked Autoencoder for Hyperspectral Image Classification

This repository provides a PyTorch implementation of **TMC-MAE**, a self-supervised learning framework for hyperspectral image (HSI) classification. It combines a graph neural network (GCN) encoder with a magnitude-guided masked autoencoder to learn spatial-spectral representations, and then fine-tunes multiple downstream classifiers on strictly separated train/test splits.

---

## 📌 Highlights

- **Dual-branch masked autoencoder**: jointly reconstructs masked spectral node features and magnitude values.
- **Magnitude-guided masking & fusion**: uses a TMCP (Topological Magnitude Connectivity Prior) map to guide node masking and SPADE-style feature fusion.
- **Strict train/test separation**: all experiments use stratified, reproducible, and disjoint train/validation/test splits.
- **Multiple downstream classifiers**: CNN2D, CNN3D, MLP, Transformer (SSFTT), and SVM.
- **Windows-friendly fallback**: includes a pure-Python `SparseTensor` fallback when `torch_sparse` is unavailable.

---

## 🗂️ Repository Structure

```text
TMC-MAE/
├── main.py                 # Entry point: pre-training + downstream evaluation
├── model.py                # GCN, MagEncoder, fusion modules, predictors & classifiers
├── magnitude.py            # Compute TMCP magnitude maps from HSI cubes
├── train_test_split.py     # Data loading, PCA, patch extraction, train/test splits
├── get_cls_map.py          # Generate classification maps and ground-truth visuals
└── README.md               # This file
```

---

## 🛠️ Requirements

Core dependencies:

```text
torch
torch-geometric
einops
numpy
scipy
scikit-learn
matplotlib
pandas
tqdm
```

Optional (recommended on Linux):

```text
torch_sparse
```

On Windows, the code automatically falls back to `torch.sparse_coo_tensor` when `torch_sparse` is not installed.

---

## 📊 Supported Datasets

| Dataset | Key | Description |
|---------|-----|-------------|
| `IP`  | Indian Pines | Classic AVIRIS benchmark |
| `LK`  | WHU-Hi-LongKou | WHU-Hi dataset |
| `HC`  | WHU-Hi-HanChuan | WHU-Hi dataset |
| `HH`  | WHU-Hi-HongHu | WHU-Hi dataset |

> Update the hard-coded paths in `train_test_split.py` and `magnitude.py` to point to your local dataset folders.

---

## 🚀 Quick Start

### 1. Generate the magnitude map

Before running the main pipeline, compute the TMCP magnitude map for your target dataset:

```bash
python magnitude.py --dataset IP --alpha 1.0 --beta 1.0 --window_size 7
```

The map will be saved to:

```text
./magnitude_maps/IP/magnitude_IP_a1.0_b1.0.npy
```

### 2. Run the full pipeline

```bash
python main.py \
  --dataset IP \
  --data_root "C:\PyCharm\Project\HSI_Project\HGCN-MHF-main\datasets" \
  --mag_path "./magnitude_maps/IP/magnitude_IP_a1.0_b1.0.npy" \
  --classifier CNN2D \
  --num_run 3
```

> Make sure `--data_root` matches the location of your HSI `.mat` files.

---

## ⚙️ Key Arguments

### Data & splits

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `IP` | Dataset name: `IP`, `LK`, `HC`, `HH` |
| `--data_root` | `C:\PyCharm\Project\HSI_Project\HGCN-MHF-main\datasets` | Root folder containing `.mat` datasets |
| `--mag_path` | `None` | Explicit path to a pre-computed magnitude map |
| `--test_ratio` | `0.9` | Fraction of labeled pixels held out for testing |
| `--val_ratio` | `0.0` | Fraction of training pixels used for validation |
| `--random_state` | `345` | Random seed for reproducible splits |

### Pre-training

| Argument | Default | Description |
|----------|---------|-------------|
| `--pca_components` | `30` | Number of PCA components kept |
| `--patch_size` | `25` | Spatial patch size for downstream classifiers |
| `--num_layers` | `3` | Number of GCN layers |
| `--hidden_channels` | `128` | Hidden dimension of GCN |
| `--s2gae_epochs` | `100` | Self-supervised pre-training epochs |
| `--s2gae_lr` | `0.0005` | Learning rate for pre-training |
| `--mask_ratio` | `0.3` | Ratio of masked nodes per iteration |
| `--use_dual_branch` | `True` | Enable dual-branch spectral + magnitude MAE |
| `--use_magnitude_mask` | `True` | Use magnitude-guided node masking |
| `--fusion_strategy` | `SPADE` | Fusion strategy: `SPADE`, `Concat`, `Add` |

### Downstream classifier

| Argument | Default | Description |
|----------|---------|-------------|
| `--classifier` | `CNN2D` | `MLP`, `CNN2D`, `CNN3D`, `Transformer`, `SVM` |
| `--classifier_epochs` | `100` | Training epochs for the classifier |
| `--early_stopping_patience` | `50` | Early stopping patience |
| `--num_run` | `3` | Number of independent runs |

---

## 📈 Outputs

After running `main.py`, you will get:

- `best_autoencoder_<dataset>.pth` — best pre-trained MAE checkpoint.
- `Training_Logs_<dataset>.txt` — per-run and averaged OA/AA/Kappa metrics.
- `classification_map/<dataset>/` — predicted classification maps and ground-truth visualization.

---

## 📖 Method Overview

1. **Magnitude Map (`magnitude.py`)**  
   Computes a per-pixel TMCP weight map from local spectral and spatial distances.

2. **Graph Construction (`main.py`)**  
   Builds a 4-neighborhood graph over the HSI pixels after PCA and min-max normalization.

3. **Self-Supervised Pre-training (`main.py`)**  
   Masks nodes (guided by magnitude), runs them through a GCN + MagEncoder, and reconstructs both spectral features and magnitude values.

4. **Feature Extraction (`main.py`)**  
   Extracts fused node embeddings from the best checkpoint.

5. **Downstream Classification (`main.py`)**  
   Trains a classifier on memory-efficient patches extracted from the embeddings, with strict train/validation/test separation.

6. **Visualization (`get_cls_map.py`)**  
   Generates classification maps and ground-truth images for qualitative analysis.

---

## 🔗 Citation

If you find this work useful, please consider citing it:

```bibtex
@article{tmcmae2025,
  title={TMC-MAE: Topological Magnitude Connectivity Masked Autoencoder for Hyperspectral Image Classification},
  author={Your Name},
  journal={},
  year={2025}
}
```

*(Please fill in the final citation once the paper is published.)*

---

## 📜 License

This project is released for academic research purposes. Please contact the authors for commercial licensing.

---

## 🙋 Acknowledgements

This implementation is built on top of [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/). The SSFTT Transformer backbone follows the architecture from the SSFTTNet paper.
