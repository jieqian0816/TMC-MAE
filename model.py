import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from einops import rearrange


# ---------------------------------------------------------------------------
# Magnitude-guided feature fusion (SPADE-style)
# ---------------------------------------------------------------------------
class MagGuidedSPADEFusion(nn.Module):
    """
    Magnitude-guided adaptive feature fusion.
    Boundary (heterogeneous-region) priors are used as a strong spatial
    constraint to modulate spectral features at every node, preventing
    boundary pixels from being smoothed into the wrong class.
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Normalization without learnable affine parameters;
        # affine parameters are generated from the magnitude branch.
        self.norm_spec = nn.LayerNorm(d_model, elementwise_affine=False)

        # Shared MLP that processes the magnitude/boundary branch
        self.shared_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU()
        )

        # Projections to obtain SPADE modulation coefficients
        self.gamma_proj = nn.Linear(d_model, d_model)
        self.beta_proj = nn.Linear(d_model, d_model)

        # Feed-forward network followed by a residual connection
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, h_spec, h_mag):
        """
        Args:
            h_spec: [N, D] spectral features from the main GCN.
            h_mag:  [N, D] magnitude/boundary features.
        Returns:
            [N, D] fused node representations.
        """
        h_spec_norm = self.norm_spec(h_spec)

        mag_features = self.shared_mlp(h_mag)
        gamma = self.gamma_proj(mag_features)
        beta = self.beta_proj(mag_features)

        # SPADE modulation: boundaries are amplified, homogeneous regions are kept smooth
        h_fused = h_spec_norm * (1.0 + gamma) + beta

        ffn_out = self.ffn(h_fused)
        final_out = self.norm2(h_fused + ffn_out)
        return final_out


class FeatureFusion(nn.Module):
    """
    Unified fusion dispatcher supporting SPADE, Concat, and Add strategies.
    """
    def __init__(self, d_model, strategy='SPADE', dropout=0.1):
        super().__init__()
        self.strategy = strategy

        if self.strategy == 'SPADE':
            self.spade = MagGuidedSPADEFusion(d_model, dropout)
        elif self.strategy == 'Concat':
            self.concat_proj = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        elif self.strategy != 'Add':
            raise ValueError(f"Unsupported fusion strategy: {strategy}")

    def forward(self, h_spec, h_mag):
        if self.strategy == 'SPADE':
            return self.spade(h_spec, h_mag)
        elif self.strategy == 'Concat':
            return self.concat_proj(torch.cat([h_spec, h_mag], dim=-1))
        else:  # Add
            return h_spec + h_mag


# ---------------------------------------------------------------------------
# Magnitude branch encoder
# ---------------------------------------------------------------------------
class SimAM(nn.Module):
    """SimAM-style 3D attention module."""
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (
            4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
        ) + 0.5
        return x * self.activation(y)


class MagDiffModule(nn.Module):
    """
    Edge-enhancement module: it highlights heterogeneous regions by
    subtracting a local average-pooled feature map.
    """
    def __init__(self, in_channel):
        super().__init__()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sigmoid = nn.Sigmoid()
        self.simam = SimAM()

    def forward(self, x):
        x = self.simam(x)
        edge = x - self.avg_pool(x)
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        out = weight * x + x
        out = self.simam(out)
        return out


class MagEncoder(nn.Module):
    """
    Bridge module: converts a 1-D graph magnitude signal into a 2-D image,
    applies MagDiffModule for spatial feature extraction, and converts back.
    """
    def __init__(self, in_channels, out_channels, n_rows, n_cols):
        super().__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols

        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )
        self.diff_extractor = MagDiffModule(in_channel=out_channels)

    def forward(self, x, adj_t=None):
        N = x.size(0)
        h = self.proj(x)
        h_img = h.view(1, self.n_rows, self.n_cols, -1).permute(0, 3, 1, 2)
        out_img = self.diff_extractor(h_img)
        out_graph = out_img.permute(0, 2, 3, 1).reshape(N, -1)
        return out_graph


# ---------------------------------------------------------------------------
# Decoders / predictors
# ---------------------------------------------------------------------------
class FeatPredictor(nn.Module):
    """Node feature reconstruction head."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, out_channels)
        )

    def forward(self, x):
        return self.net(x)


class MagnitudePredictor(nn.Module):
    """Magnitude reconstruction / prediction head."""
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# GCN encoder
# ---------------------------------------------------------------------------
class GCN(nn.Module):
    """
    Multi-layer GCN encoder. Returns the activations of every intermediate
    layer so that they can be concatenated into a rich node representation.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=False, add_self_loops=False)
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False, add_self_loops=False)
            )
        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=False, add_self_loops=False)
        )
        self.dropout = dropout

    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(F.relu(x))
        return xx


# ---------------------------------------------------------------------------
# Downstream classifiers (optional)
# ---------------------------------------------------------------------------
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, dropout=0, norm=nn.BatchNorm2d,
                 act_func=nn.ReLU):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride), padding=(padding, padding), dilation=(dilation, dilation),
            groups=groups, bias=bias
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class Stem(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        self.conv1 = ConvLayer(in_chans, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False, act_func=None)
        )
        self.conv3 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv3(x)
        x = rearrange(x, 'b d h w -> b h w d')
        return x


class CNN2D(nn.Module):
    """2-D CNN patch classifier."""
    def __init__(self, in_chans=30, num_classes=16, depths=[1], dims=64, drop_rate=0.0,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.patch_embed = Stem(in_chans=in_chans, embed_dim=self.embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = torch.flatten(x, 1, 2)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x, return_features=False):
        x = x.squeeze(1)
        feat = self.forward_features(x)
        if return_features:
            return feat
        x = self.head(feat)
        return x


class SimpleMLP(nn.Module):
    def __init__(self, in_chans=30, patch_size=15, num_classes=16):
        super().__init__()
        self.flatten = nn.Flatten()
        in_features = in_chans * patch_size * patch_size
        self.net = nn.Sequential(
            nn.Linear(in_features, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, return_features=False):
        x = self.flatten(x)
        for i in range(len(self.net) - 1):
            x = self.net[i](x)
        feat = x
        out = self.net[-1](feat)
        if return_features:
            return feat
        return out


class CNN3D(nn.Module):
    def __init__(self, in_chans=30, num_classes=16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, (3, 3, 3), padding='same'), nn.ReLU(), nn.MaxPool3d((2, 2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, (3, 3, 3), padding='same'), nn.ReLU(), nn.MaxPool3d((2, 2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, (3, 3, 3), padding='same'), nn.ReLU()
        )
        self.adaptive_pool = nn.AdaptiveAvgPool3d((3, 1, 1))
        self.linear = nn.Linear(128 * 3 * 1 * 1, num_classes)

    def forward(self, X, return_features=False):
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)
        X = self.adaptive_pool(X)
        feat = rearrange(X, 'b c h w d -> b (c h w d)')
        out = self.linear(feat)
        if return_features:
            return feat
        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)
            x = mlp(x)
        return x


class SSFTTnet(nn.Module):
    def __init__(self, in_chans=30, patch_size=15, num_classes=16, num_tokens=4,
                 dim=64, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.L = num_tokens
        self.cT = dim
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(1, 8, (3, 3, 3)), nn.BatchNorm3d(8), nn.ReLU()
        )
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(8 * (in_chans - 2), 64, (3, 3)), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.token_wA = nn.Parameter(torch.empty(1, self.L, 64))
        nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT))
        nn.init.xavier_normal_(self.token_wV)
        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        nn.init.normal_(self.pos_embedding, std=.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()
        self.nn1 = nn.Linear(dim, num_classes)
        nn.init.xavier_uniform_(self.nn1.weight)

    def forward(self, x, mask=None, return_features=False):
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        wa = rearrange(self.token_wA, 'b h w -> b w h')
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h').softmax(dim=-1)
        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask)
        feat = self.to_cls_token(x[:, 0])
        out = self.nn1(feat)
        if return_features:
            return feat
        return out


class SVMWrapper:
    """Scikit-learn SVM wrapper for quick baseline evaluation."""
    def __init__(self, num_classes=16, C=100.0):
        from sklearn.svm import SVC
        self.clf = SVC(kernel='rbf', C=C, gamma='scale')
        self.num_classes = num_classes

    def train_from_loader(self, train_loader):
        import numpy as np
        X_tr, y_tr = [], []
        for data, target in train_loader:
            X_tr.append(data.view(data.size(0), -1).cpu().numpy())
            y_tr.append(target.cpu().numpy())
        self.clf.fit(np.concatenate(X_tr), np.concatenate(y_tr))

    def eval(self):
        pass

    def to(self, device):
        return self

    def __call__(self, x, return_features=False):
        x_np = x.view(x.size(0), -1).cpu().numpy()
        if return_features:
            return x.view(x.size(0), -1)
        preds = self.clf.predict(x_np)
        out = torch.zeros(x.size(0), self.num_classes, device=x.device)
        out[torch.arange(x.size(0)), torch.tensor(preds, device=x.device)] = 1.0
        return out
