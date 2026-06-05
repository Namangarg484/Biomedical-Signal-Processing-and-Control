"""
PINNACLE — Model architecture matching draft.tex §3.3.

Dual-stream CNN: 1D spectral branch + 2D scalogram branch
→ SeparationCross fusion (sigmoid gating + cross-attention)
→ Classifier (FC→ReLU→Dropout→FC→Softmax)

Total trainable parameters: ~1.62M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from pinnacle.utils import logger, count_parameters


# ================================================================
# 1D SPECTRAL BRANCH (draft.tex §3.3.1)
# ================================================================

class SpectralBranch(nn.Module):
    """
    1D ResNet-style CNN for Raman spectra.

    Architecture (4 layers + 1 residual block):
        Conv1d(1→64, k=7, stride=2) → BN → ReLU          [stem, halves L]
        Conv1d(64→128, k=5) → BN                           [residual input]
        Conv1d(128→128, k=3) → BN → ReLU(+residual skip)  [res block]
        Conv1d(128→embed_dim, k=3) → BN → ReLU            [projection]
        AdaptiveAvgPool1d(1) → embed_dim embedding

    The residual connection runs from the output of the first 128-ch conv
    (before ReLU) to after the second 128-ch conv, so both branches are
    128-dim and the add is dimension-compatible.
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        # Stem: stride-2 to compress the 1000-pt sequence early
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(64)

        # Residual block: 64 → 128 (main path), skip = main path before ReLU
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False)
        self.bn2   = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm1d(128)

        # Projection to final embedding dimension
        self.conv4 = nn.Conv1d(128, embed_dim, kernel_size=3, padding=1, bias=False)
        self.bn4   = nn.BatchNorm1d(embed_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return full feature map (B, embed_dim, L') for cross-attention."""
        if x.dim() == 2:
            x = x.unsqueeze(1)                      # (B, 1, L)
        x = F.relu(self.bn1(self.conv1(x)))          # (B, 64, L/2)

        # Residual block
        res = self.bn2(self.conv2(x))                # (B, 128, L/2)  — skip
        x   = F.relu(res)                            # (B, 128, L/2)
        x   = self.bn3(self.conv3(x))                # (B, 128, L/2)
        x   = F.relu(x + res)                        # (B, 128, L/2)  — residual add

        x = F.relu(self.bn4(self.conv4(x)))          # (B, embed_dim, L/2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len) or (B, 1, seq_len)
        Returns:
            z_s: (B, embed_dim)
        """
        h = self.forward_features(x)                 # (B, embed_dim, L/2)
        return F.adaptive_avg_pool1d(h, 1).squeeze(-1)  # (B, embed_dim)


# ================================================================
# 2D SCALOGRAM BRANCH (draft.tex §3.3.2)
# ================================================================

class ScalogramBranch(nn.Module):
    """
    2D CNN for CWT scalogram images.
    Conv2D(3→32, k=3) → BN → ReLU → MaxPool2D →
    Conv2D(32→64, k=3) → BN → ReLU → MaxPool2D →
    Conv2D(64→128, k=3) → BN → ReLU → MaxPool2D →
    AdaptiveAvgPool2d(1) → 128-dim embedding
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) scalogram image
        Returns:
            z_w: (B, embed_dim)
        """
        h = self.features(x)    # (B, embed_dim, H', W')
        z = self.pool(h)        # (B, embed_dim, 1, 1)
        return z.view(z.size(0), -1)  # (B, embed_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return full spatial feature map (for cross-attention)."""
        return self.features(x)  # (B, embed_dim, H', W')


# ================================================================
# SEPARATIONCROSS FUSION (draft.tex §3.3.3)
# ================================================================

class SeparationCross(nn.Module):
    """
    SeparationCross fusion module with spatial cross-attention and
    residual gating.

    Step 1 — Separation gating (on pooled embeddings):
        α = σ(W_α · z_s + b_α)   (spectral gate)
        β = σ(W_β · z_w + b_β)   (scalogram gate)

    Step 2 — Spatial cross-attention (on unpooled feature maps):
        Spectral features h_s (B, D, L) are pooled to S positions.
        Scalogram features h_w (B, D, H', W') give H'W' positions.
        Q = W_q · gated_spectral_seq    (B, S, D)
        K = W_k · gated_scalogram_seq   (B, H'W', D)
        V = W_v · gated_scalogram_seq   (B, H'W', D)
        Attn = softmax(QK^T / √d) V     (B, S, D) — real attention

    Step 3 — Residual-gated fusion:
        γ = learnable scalar (init=0)
        z_fused = [z_β + γ · pool(z_attn) ; z_β]  ∈ ℝ^{256}
    """

    def __init__(self, embed_dim: int = 128, spectral_pool_len: int = 25):
        super().__init__()
        self.embed_dim = embed_dim

        # Separation gates (Eqs. 4-5 in paper)
        self.gate_spectral = nn.Linear(embed_dim, embed_dim)
        self.gate_scalogram = nn.Linear(embed_dim, embed_dim)

        # Pool spectral sequence to manageable length for attention
        self.spectral_pool = nn.AdaptiveAvgPool1d(spectral_pool_len)

        # Cross-attention projections
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        self.scale = embed_dim ** 0.5

        # Residual gate: γ starts at 0 → pure scalogram at init
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        h_s: torch.Tensor,
        h_w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_s: (B, D, L) unpooled spectral feature map
            h_w: (B, D, H', W') unpooled scalogram feature map

        Returns:
            z_fused: (B, 2*D) fused representation
            alpha: (B, D) spectral gate values
            beta: (B, D) scalogram gate values
        """
        B, D, L = h_s.shape
        _, _, Hp, Wp = h_w.shape

        # Pool for gating
        z_s = h_s.mean(dim=-1)                  # (B, D)
        z_w = h_w.mean(dim=(-2, -1))            # (B, D)

        # Step 1: Separation gating
        alpha = torch.sigmoid(self.gate_spectral(z_s))   # (B, D)
        beta = torch.sigmoid(self.gate_scalogram(z_w))   # (B, D)

        # Apply gates to spatial features (broadcast across positions)
        h_s_gated = alpha.unsqueeze(-1) * h_s             # (B, D, L)
        h_w_gated = beta.unsqueeze(-1).unsqueeze(-1) * h_w  # (B, D, H', W')

        # Prepare spatial sequences for attention
        seq_s = self.spectral_pool(h_s_gated)             # (B, D, S)
        seq_s = seq_s.transpose(1, 2)                     # (B, S, D)

        seq_w = h_w_gated.view(B, D, Hp * Wp)             # (B, D, H'W')
        seq_w = seq_w.transpose(1, 2)                     # (B, H'W', D)

        # Step 2: Cross-attention (spectral queries scalogram)
        Q = self.W_q(seq_s)                               # (B, S, D)
        K = self.W_k(seq_w)                               # (B, H'W', D)
        V = self.W_v(seq_w)                               # (B, H'W', D)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, S, H'W')
        attn_weights = F.softmax(attn_weights, dim=-1)
        z_attn = torch.matmul(attn_weights, V)            # (B, S, D)
        z_attn = z_attn.mean(dim=1)                       # (B, D) pool over S

        # Step 3: Residual-gated fusion
        z_beta = beta * z_w                               # (B, D)
        z_combined = z_beta + self.gamma * z_attn          # (B, D)
        z_fused = torch.cat([z_combined, z_beta], dim=1)   # (B, 2D)

        return z_fused, alpha, beta


# ================================================================
# COMPLETE PINNACLE MODEL (draft.tex §3.3)
# ================================================================

class PINNACLE(nn.Module):
    """
    Complete PINNACLE model for bacterial classification.

    Architecture:
        SpectralBranch(1D) → 128-dim
        ScalogramBranch(2D) → 128-dim
        SeparationCross → 256-dim
        Classifier(256→128→C)

    Total params: ~1.62M
    """

    def __init__(
        self,
        num_classes: int = 5,
        embed_dim: int = 128,
        dropout: float = 0.3,
        use_fusion: bool = True,
        mode: str = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Resolve mode: 'fusion', 'spectral_only', 'scalogram_only'
        # Backward-compatible with use_fusion bool
        if mode is not None:
            self.mode = mode
        else:
            self.mode = "fusion" if use_fusion else "spectral_only"
        self.use_fusion = (self.mode == "fusion")

        # Encoding branches
        self.spectral_branch = SpectralBranch(in_channels=1, embed_dim=embed_dim)
        self.scalogram_branch = ScalogramBranch(in_channels=3, embed_dim=embed_dim)

        # Fusion / classifier dim
        if self.mode == "fusion":
            self.fusion = SeparationCross(embed_dim=embed_dim)
            classifier_dim = 2 * embed_dim  # 256
        else:
            self.fusion = None
            classifier_dim = embed_dim  # 128 (single branch)

        # Classifier head (draft.tex §3.3.4)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        logger.info(
            f"PINNACLE initialised: classes={num_classes}, embed={embed_dim}, "
            f"mode={self.mode}, params={count_parameters(self):,}"
        )

    def forward(
        self,
        raman: torch.Tensor,
        scalogram: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            raman: (B, seq_len) preprocessed Raman spectrum
            scalogram: (B, 3, H, W) CWT scalogram image

        Returns:
            logits: (B, num_classes)
            alpha: gate values (or None if no fusion)
            beta: gate values (or None if no fusion)
        """
        if self.mode == "fusion" and scalogram is not None:
            h_s = self.spectral_branch.forward_features(raman)   # (B, D, L)
            h_w = self.scalogram_branch.forward_features(scalogram)  # (B, D, H', W')
            z_fused, alpha, beta = self.fusion(h_s, h_w)
            logits = self.classifier(z_fused)
            return logits, alpha, beta

        elif self.mode == "scalogram_only" and scalogram is not None:
            z_w = self.scalogram_branch(scalogram)
            logits = self.classifier(z_w)
            return logits, None, None

        else:
            # spectral_only (default fallback)
            z_s = self.spectral_branch(raman)
            logits = self.classifier(z_s)
            return logits, None, None

    def forward_with_fusion(
        self, raman: torch.Tensor, scalogram: torch.Tensor
    ):
        """Explicit fusion forward (always uses both branches)."""
        z_s = self.spectral_branch(raman)       # pooled (B, D) for return
        z_w = self.scalogram_branch(scalogram)   # pooled (B, D) for return

        if self.fusion is not None:
            h_s = self.spectral_branch.forward_features(raman)      # (B, D, L)
            h_w = self.scalogram_branch.forward_features(scalogram) # (B, D, H', W')
            z_fused, alpha, beta = self.fusion(h_s, h_w)
        else:
            z_fused = z_s
            alpha = beta = None

        logits = self.classifier(z_fused)
        return logits, alpha, beta, z_s, z_w
