"""The shared eval-time glimpse loop: one episode, one readout per timestep.

Every downstream task runs the SAME loop — init the canvas state, then for each viewpoint
either pre-crop the glimpse (uniform patcher) or hand over the full image (foveated/square),
step ``model.canvit``, carry the state forward — and differs only in what it reads off each
step. Before this module there were two copies of that loop (``ade20k/rollout.py`` and
``in1k/rollout.py``) and ``in1k`` imported the routing helpers out of ``ade20k``, a task
reaching into another task's package purely because the generalization was missing.

Design (eval-merge doc §2, taken from ``canvit_eval/episode.py``, which had it right):
**share the loop, not the metrics.** ``run_episode`` owns the routing and the recurrence;
``readout`` stays with the task, because ade20k's canvas features, in1k's CLS token and
distill's teacher targets are where each task's correctness actually lives.

``readout`` rather than "return the per-step outputs": both callers keep only a small
projection of each step, and a foveated ADE20K episode's ten full ``CanViTOutput`` states are
gigabytes. The callback keeps memory exactly where it was.

This is the EVAL loop. Training goes through ``harness/rollout/engine.py::run_rollout``,
which additionally owns BPTT chunking, branches and per-glimpse losses.
"""

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from typing import Any

import torch
from canvit_pytorch import CanViTOutput, sample_at_viewpoint
from canvit_pytorch.patcher.foveated import FoveatedPatcher
from canvit_pytorch.patcher.square import SquarePatcher
from canvit_pytorch.viewpoint import Viewpoint
from torch import Tensor

__all__ = ["consumes_full_image", "derive_glimpse_px", "run_episode"]


def consumes_full_image(model: Any) -> bool:
    """True when the patcher does its own foveation and must be handed the WHOLE image.

    Pre-cropping first would double-crop. Takes any wrapper exposing ``.canvit``
    (``CanViTForSemanticSegmentation`` / ``CanViTForImageClassification``), which is why
    both tasks could share it even before it lived here.
    """
    return isinstance(getattr(model.canvit, "patcher", None), (FoveatedPatcher, SquarePatcher))


def derive_glimpse_px(model: Any, glimpse_px: int | None) -> int:
    """Training-matched uniform glimpse crop size (canvit_eval's rule): the patch-embed
    conv must yield exactly ``glimpse_grid_size`` tokens per side."""
    canvit = model.canvit
    patch_size = canvit.backbone.patch_size_px
    stride = getattr(canvit.backbone, "patch_stride_px", patch_size)
    glimpse_grid = getattr(canvit, "glimpse_grid_size", None)
    grid = glimpse_grid if glimpse_grid is not None else 8
    if glimpse_px is None:
        glimpse_px = (grid - 1) * stride + patch_size
    assert (glimpse_px - patch_size) % stride == 0 and glimpse_px >= patch_size, (
        f"glimpse_px={glimpse_px} incompatible with patch_size_px={patch_size}, "
        f"patch_stride_px={stride} (need (glimpse_px - patch) divisible by stride)"
    )
    tokens = (glimpse_px - patch_size) // stride + 1
    if glimpse_grid is not None:
        assert tokens == glimpse_grid, (
            f"glimpse_px={glimpse_px} yields {tokens} tokens/side but the model "
            f"trained with glimpse_grid_size={glimpse_grid}"
        )
    return glimpse_px


def run_episode[R](
    *,
    model: Any,
    images: Tensor,
    viewpoints: Sequence[Viewpoint],
    canvas_grid: int,
    glimpse_px: int | None,
    readout: Callable[[CanViTOutput], R],
    no_grad: bool = False,
) -> list[R]:
    """Run one recurrent episode over ``viewpoints``; return ``readout(out)`` per timestep.

    ``no_grad`` runs the backbone under ``torch.no_grad`` — in1k's frozen mode, where only
    the caller's head carries gradient. Left False for the paths that either need the graph
    or are already inside an outer no-grad context.
    """
    B = images.shape[0]
    full_image = consumes_full_image(model)
    px = None if full_image else derive_glimpse_px(model, glimpse_px)

    state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid)
    out_per_step: list[R] = []
    ctx = torch.no_grad() if no_grad else nullcontext()
    with ctx:
        for vp in viewpoints:
            model_input = images if full_image else sample_at_viewpoint(
                spatial=images, viewpoint=vp, glimpse_size_px=px
            )
            out = model.canvit(image=model_input, state=state, viewpoint=vp)
            state = out.state
            out_per_step.append(readout(out))
    return out_per_step
