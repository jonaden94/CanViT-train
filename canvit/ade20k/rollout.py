"""ADE20K's readout off the shared episode loop: canvas features per timestep.

The unified replacement for specialize's ``extract_canvas_features``, which was
uniform-only and reached past the task wrapper into the raw pretraining model — the root
cause of the 3-month silent breakage (unification-status §4).

Everything that is not ADE20K-specific now lives in ``harness/rollout/episode.py``: the
glimpse loop, the uniform-vs-foveated routing, and the training-matched crop size. What is
left here is the readout — the canvas spatial features the probe head consumes. Only
canvas_hidden is produced (recon_normalized dropped, D3).
"""

from torch import Tensor

from canvit.core import CanViTForSemanticSegmentation
from canvit.core.viewpoint import Viewpoint

from ..harness.rollout.episode import run_episode


def rollout_canvas_hidden(
    *,
    seg: CanViTForSemanticSegmentation,
    images: Tensor,
    viewpoints: list[Viewpoint],
    canvas_grid: int,
    glimpse_px: int | None,
) -> list[Tensor]:
    """Run the recurrent rollout, return canvas_hidden [B, G, G, D] per timestep.

    Steps ``seg.canvit`` directly (CanViT-only execution, blessed by the wrapper
    docstring); the probe head is applied by the caller so the same features can
    feed training and evaluation.
    """
    B = images.shape[0]
    return run_episode(
        model=seg, images=images, viewpoints=viewpoints,
        canvas_grid=canvas_grid, glimpse_px=glimpse_px,
        readout=lambda out: seg.canvit.get_spatial(out.state.canvas).view(B, canvas_grid, canvas_grid, -1),
    )
