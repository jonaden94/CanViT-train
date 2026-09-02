"""The DINOv3 teacher's own ADE20K score: one passive forward, mIoU at t0.

The reference line every CanViT ADE20K number is read against — what the teacher this model
distilled from achieves on the same val set with the same probe protocol, given the whole
image at once instead of a glimpse sequence. Ported from
``canvit_eval/tasks/ade20k_seg.py::run_dinov3``.

No CanViT and no checkpoint are involved, which is why it is a separate entry point rather
than a policy of the main one: there is no episode, no canvas and no timestep axis. It
shares what it should — ``make_ade20k_val_loader`` (same images, same transforms) and
``eval_probe_on_batch`` (same upsample-then-argmax reduction) — so the baseline and the
model it bounds are measured identically.

``eval_resolution`` has NO default on purpose: the probe was trained at one resolution and
feeding the teacher a different one degrades mIoU silently, with no error to show for it.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from canvit_pytorch.metrics import mIoUAccumulator
from canvit_pytorch.model_source import load_segmentation_probe
from canvit_pytorch.teacher import load_teacher

from canvit_train.ade20k.config import Ade20kConfig
from canvit_train.ade20k.data import IGNORE_LABEL, NUM_CLASSES, make_ade20k_val_loader
from canvit_train.ade20k.metrics import eval_probe_on_batch

log = logging.getLogger(__name__)

DINOV3_PATCH_SIZE = 16


@dataclass
class DinoV3BaselineOpts:
    """The teacher/probe pair and the resolution they were trained at."""

    probe_repo: str
    """The published ADE20K segmentation probe to read the teacher's features with."""
    eval_resolution: int
    """Resolution the probe was trained at (e.g. 512). The teacher runs at THIS, not at
    ``cfg.scene_size`` — a mismatch degrades mIoU with no error, so there is no default."""
    teacher_repo: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    device: str = "cuda"


@torch.no_grad()
def evaluate_dinov3(cfg: Ade20kConfig, opts: DinoV3BaselineOpts) -> dict[str, Any]:
    """mIoU of ``teacher_repo`` + ``probe_repo`` on ADE20K val, one forward per image."""
    device = torch.device(opts.device if torch.cuda.is_available() else "cpu")
    teacher = load_teacher(opts.teacher_repo, device)
    probe = load_segmentation_probe(opts.probe_repo).to(device).eval()
    grid = opts.eval_resolution // DINOV3_PATCH_SIZE
    loader = make_ade20k_val_loader(cfg)

    acc = mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device)
    amp = (torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda"
           else torch.autocast("cpu", enabled=False))
    n_images, t0 = 0, time.perf_counter()
    for i, (images, masks) in enumerate(loader):
        if cfg.limit_val_batches is not None and i >= cfg.limit_val_batches:
            break
        images, masks = images.to(device), masks.to(device)
        n_images += images.shape[0]
        with amp:
            resized = F.interpolate(images, size=(opts.eval_resolution, opts.eval_resolution),
                                    mode="bilinear", align_corners=False)
            feats = teacher.forward_norm_features(resized).patches
        # The probe runs OUTSIDE autocast on float32, as in the CanViT path — same reduction,
        # so the baseline and the model it bounds differ only in where the features came from.
        eval_probe_on_batch(probe, feats.view(images.shape[0], grid, grid, -1), masks, acc)

    return {
        "task": "ade20k_dinov3",
        "metrics": {"miou_t0": acc.compute()},
        "protocol": {
            "teacher_repo": opts.teacher_repo, "probe_repo": opts.probe_repo,
            "eval_resolution": opts.eval_resolution, "scene_size": cfg.scene_size,
            "resize_mode": cfg.resize_mode, "n_images": n_images,
            "limit_val_batches": cfg.limit_val_batches,
        },
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }
