"""JetFormer (arXiv:2601.17215), the frame-dependent control teacher on [pt, eta, phi].

A port of ``JetFormerNet`` of the sibling repository, jet_tagging_gdl
(src/algorithms/jetformer.py), at its layer norms: the submodule names are the
sibling's, so its checkpoints load with ``strict=True``. It reads the columns in the
datamodule's normalisation, as the sibling's JetTaggingModule fed them.

Bias-free Q/K/V projections, a pre-activation feed-forward sub-layer (norm, activation,
linear) with bias-free linears and a x2 expansion, dropout on the residual stream after
each block, no padding mask, and a learned CLS token read out through one norm and one
linear layer. The padded slots are attended to like any other token, so the logits
depend on the padding.
"""

import torch
from torch import nn

from src.algorithms.components import FeatureNet


class SeqNorm(nn.Module):
    """A layer norm over the channel axis, under the sibling's wrapper, whose ``norm``
    attribute names the checkpoint keys."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class JetFormerAttention(nn.Module):
    """Pre-norm multi-head self-attention over all the tokens, residual."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = SeqNorm(dim)
        self.q, self.k, self.v = (nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.out = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over the tokens of a (jets, tokens, channels) batch and add the
        result."""
        b, n, _ = x.shape
        h = self.norm(x)
        q, k, v = (
            proj(h).view(b, n, self.num_heads, self.head_dim)
            for proj in (self.q, self.k, self.v)
        )
        scores = torch.einsum("bqhc,bkhc->bhqk", q, k) / (self.head_dim**0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bkhc->bqhc", attn, v).reshape(b, n, -1)
        return x + self.out(out)


class JetFormerBlock(nn.Module):
    """Self-attention, then a x2-expansion pre-activation feed-forward sub-layer."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = JetFormerAttention(dim, num_heads)
        self.norm_0 = SeqNorm(dim)
        self.activ_0 = nn.ReLU()
        self.linear_0 = nn.Linear(dim, dim * 2, bias=False)
        self.norm_1 = SeqNorm(dim * 2)
        self.activ_1 = nn.ReLU()
        self.linear_1 = nn.Linear(dim * 2, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        h = self.linear_0(self.activ_0(self.norm_0(x)))
        h = self.linear_1(self.activ_1(self.norm_1(h)))
        return self.dropout(x + h)


class JetFormerNet(FeatureNet):
    """Embed the constituents, prepend a CLS token, run the blocks, read token 0.

    The hint is what the classifier reads: the normed CLS token, width ``embed_dim``.

    :param num_heads: attention heads per block, which must divide ``embed_dim``.
    :param num_transformers: number of blocks.
    """

    required_features = ["pt", "eta", "phi"]

    def __init__(
        self,
        in_dim: int = 3,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_transformers: int = 4,
        output_dim: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hint_dim = embed_dim
        self.embedding = nn.Linear(in_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [
                JetFormerBlock(embed_dim, num_heads, dropout)
                for _ in range(num_transformers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_hint(x)[0]

    def forward_with_hint(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the logits and the normed CLS token of a (jets, constituents,
        features) batch."""
        x = self.embedding(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        # (jets, 1 + constituents, embed_dim)
        x = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            x = block(x)
        hint = self.norm(x[:, 0, :])
        return self.classifier(hint), hint
