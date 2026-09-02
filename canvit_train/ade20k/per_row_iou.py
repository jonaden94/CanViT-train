"""Per-(image, class, timestep) IoU counts for ADE20K val.

A dataset mIoU answers "how good is this model"; these rows answer "on WHICH images and
WHICH classes, and how does that change as glimpses accumulate". That is the question a
glimpse policy is actually about, and it cannot be recovered from an aggregate.

Ported from ``canvit_eval/tasks/ade20k_obj/iou.py`` — the per-row METRIC only. Its
three-stage cached-feature pipeline is deliberately not ported (eval-merge doc §3): it is a
second notion of "run with cached intermediates" alongside the one `slurm/runs/` already
has. Here the rows are an optional output of the ade20k eval that is running anyway, so
they cost one scatter-add per batch and no extra forward.

Raw COUNTS are stored, never a per-row ratio. `inter/union` is undefined for a class absent
from an image (union 0) and averaging per-image ratios is not the dataset mIoU; keeping
counts lets any aggregation be done downstream, and lets these rows be checked against the
accumulator (they are, in `test_per_row_iou.py`).

Parquet is written through **pyarrow**, which this repo already depends on. canvit_eval used
pandas, which sits in its *dev* dependency group while this module imports it at module
level — so that task only ran in a dev install.
"""

import logging
from pathlib import Path

import torch
from torch import Tensor

from canvit_train.ade20k.data import IGNORE_LABEL

log = logging.getLogger(__name__)


def batch_confusion(preds: Tensor, masks: Tensor, n_classes: int) -> tuple[Tensor, Tensor, Tensor]:
    """Per-image ``(inter, union, gt_area)`` for a whole batch in one scatter_add.

    ``preds``/``masks``: ``[B, H, W]`` int64, mask values in ``[0, n_classes)`` plus
    ``IGNORE_LABEL``. Returns three ``[B, n_classes]`` float32 tensors. Integer-exact:
    counts are bounded by 512² = 262144, well under float32's 2^24 exact-integer limit.
    """
    B, H, W = preds.shape
    assert masks.shape == preds.shape, (preds.shape, masks.shape)
    assert preds.dtype == torch.int64 and masks.dtype == torch.int64, (preds.dtype, masks.dtype)
    C2 = n_classes * n_classes

    pair = preds * n_classes + masks
    keep = (masks != IGNORE_LABEL) & (pair >= 0) & (pair < C2)
    img_idx = torch.arange(B, device=preds.device).view(B, 1, 1).expand_as(preds)
    flat = (img_idx * C2 + pair)[keep]

    cm = torch.zeros(B * C2, dtype=torch.float32, device=preds.device)
    cm.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    cm = cm.view(B, n_classes, n_classes)

    inter = cm.diagonal(dim1=1, dim2=2)
    row_sum, col_sum = cm.sum(dim=2), cm.sum(dim=1)
    return inter, row_sum + col_sum - inter, col_sum


def miou_from_rows(inter: Tensor, union: Tensor) -> float:
    """Dataset mIoU from per-row counts: sum over images per class, mean over the classes
    that appear. The aggregation ``mIoUAccumulator`` performs, done from the rows — which is
    what makes the two checkable against each other."""
    # float64: the entries are integer COUNTS, so summing 2000 images per class in float32
    # loses the low bits and the result drifts from the accumulator by ~1e-9. Exact here.
    total_inter, total_union = inter.double().sum(0), union.double().sum(0)
    valid = total_union > 0
    return (total_inter[valid] / total_union[valid]).mean().item()


def write_rows(path: Path, inter: Tensor, union: Tensor, gt_area: Tensor, *,
               mask_resolution_px: int, resize_mode: str, extra: dict | None = None) -> Path:
    """Write ``[T, N, C]`` counts as one parquet row per (timestep, image, class).

    Class indices are 1-based, as ADE20K labels them.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    T, N, C = inter.shape
    cols = {
        "timestep": torch.arange(T).repeat_interleave(N * C).numpy(),
        "image_idx": torch.arange(N).repeat_interleave(C).repeat(T).numpy(),
        "class_idx": torch.arange(1, C + 1).repeat(T * N).numpy(),
        "inter_px": inter.flatten().to(torch.int64).numpy(),
        "union_px": union.flatten().to(torch.int64).numpy(),
        "gt_area_px": gt_area.flatten().to(torch.int64).numpy(),
    }
    table = pa.table(cols).replace_schema_metadata({
        "mask_resolution_px": str(mask_resolution_px), "resize_mode": str(resize_mode),
        **{k: str(v) for k, v in (extra or {}).items()},
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    log.info("wrote %d per-row IoU rows to %s", table.num_rows, path)
    return path
