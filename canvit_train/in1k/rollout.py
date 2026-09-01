"""IN1k's readout off the shared episode loop: the CLS token per timestep.

The classification counterpart of ``ade20k/rollout.py``. Both used to carry their own copy
of the glimpse loop, and this module imported the routing helpers out of ``ade20k.rollout``
— one task reaching into another task's package, which was the clearest evidence that the
shared abstraction was missing rather than unwanted. It now lives in
``harness/rollout/episode.py`` and only the readout stays here.

Frozen mode runs the backbone under no_grad and trains only LN+head; finetune runs the whole
graph. The head is applied by the caller (``clf.head(clf.norm(cls))``) so the same CLS stream
feeds training and evaluation.
"""

import torch
from canvit_pytorch import CanViTForImageClassification
from canvit_pytorch.policies import coarse_to_fine_viewpoints, repeated_full_scene
from canvit_pytorch.viewpoint import Viewpoint
from torch import Tensor

from ..harness.config import FoveatedScaleConfig
from ..harness.rollout.episode import run_episode
from ..harness.rollout.eval_viewpoints import make_random_viewpoints

__all__ = ["eval_viewpoints", "rollout_cls_tokens"]


def eval_viewpoints(
    policy: str, batch_size: int, device: torch.device, n: int, *,
    is_foveated: bool, foveated_scale: FoveatedScaleConfig,
) -> list[Viewpoint]:
    """Deploy-time viewpoint sequence. ``coarse_to_fine`` (canvit_eval default):
    quadtree full->quadrants->…; ``full``: repeated full scene; ``random``: the
    training random law (patcher-aware). Foveated/square honor ``fix_size=scale*H``,
    so C2F's varying scales are only in-distribution for a per-glimpse-scale model;
    fixed-scale foveated models should deploy with a scale-pinned policy.

    Superseded by ``harness/rollout/eval_viewpoints.py::open_loop_viewpoints``, which every
    caller in the package now uses; only this module's own test still reaches for it. Left
    in place rather than deleted here — the policy surface is Stage 3's subject, not this
    stage's (eval-merge doc §5).
    """
    if policy == "coarse_to_fine":
        return coarse_to_fine_viewpoints(batch_size, device, n)
    if policy == "full":
        return repeated_full_scene(batch_size, device, n)
    if policy == "random":
        return make_random_viewpoints(
            batch_size, device, n, min_scale=0.05, max_scale=1.0, start_with_full_scene=True,
            is_foveated=is_foveated, foveated_scale=foveated_scale,
        )
    raise ValueError(f"unknown eval policy: {policy}")


def rollout_cls_tokens(
    *,
    clf: CanViTForImageClassification,
    images: Tensor,
    viewpoints: list[Viewpoint],
    canvas_grid: int,
    glimpse_px: int | None,
    freeze_backbone: bool,
) -> list[Tensor]:
    """Run the rollout; return the CLS token [B, D] after each timestep. In frozen
    mode the backbone runs under no_grad (only the caller's head carries grad)."""
    return run_episode(
        model=clf, images=images, viewpoints=viewpoints,
        canvas_grid=canvas_grid, glimpse_px=glimpse_px, no_grad=freeze_backbone,
        readout=lambda out: out.state.recurrent_cls[:, 0].float(),
    )
