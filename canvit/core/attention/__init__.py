"""Asymmetric cross-attention for canvas-based vision transformers.

Canvas Attention uses dense projections (Linear) on the local stream (few tokens)
and Identity on the canvas stream (many tokens), avoiding O(D²) on the large canvas.

- CanvasReadAttention (CRA): local queries canvas
- CanvasWriteAttention (CWA): canvas queries local

Full-QKVO variants (ablation) in canvit.core.attention.full.
"""

from canvit.core.attention.base import CanvasAttention
from canvit.core.attention.full import CanvasReadAttentionFull, CanvasWriteAttentionFull
from canvit.core.attention.read import CanvasReadAttention
from canvit.core.attention.write import CanvasWriteAttention


__all__ = [
    "CanvasAttention",
    "CanvasReadAttention",
    "CanvasReadAttentionFull",
    "CanvasWriteAttention",
    "CanvasWriteAttentionFull",
]
