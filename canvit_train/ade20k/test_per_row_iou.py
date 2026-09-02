"""Per-(image, class, timestep) IoU counts.

`batch_confusion` computes a 150x150 confusion matrix per image with one scatter_add into a
float32 buffer, which is fast and non-obvious enough to be worth pinning against a plain
`np.bincount` reference — the check `canvit_eval/tests/test_iou_equivalence.py` existed for
(eval-merge doc §5, Stage 5), ported here alongside the metric it covers.

The end-to-end property is checked on the cluster and recorded in the plan: on exp34's probe
the rows aggregate to the accumulator's `miou_t{t}` at all ten timesteps to 1.1e-16, i.e.
summation order only.
"""

import numpy as np
import pytest
import torch

from canvit_train.ade20k.data import IGNORE_LABEL
from canvit_train.ade20k.metrics import mIoUAccumulator
from canvit_train.ade20k.per_row_iou import batch_confusion, miou_from_rows


def _reference(preds: np.ndarray, masks: np.ndarray, n_classes: int):
    """np.bincount, one image at a time — the obvious implementation."""
    inter, union, gt = (np.zeros((preds.shape[0], n_classes)) for _ in range(3))
    for i in range(preds.shape[0]):
        p, m = preds[i].ravel(), masks[i].ravel()
        keep = m != IGNORE_LABEL
        p, m = p[keep], m[keep]
        cm = np.bincount(p * n_classes + m, minlength=n_classes**2).reshape(n_classes, n_classes)
        d = np.diag(cm)
        inter[i], union[i], gt[i] = d, cm.sum(1) + cm.sum(0) - d, cm.sum(0)
    return inter, union, gt


@pytest.mark.parametrize("n_classes", [4, 150])
def test_scatter_add_matches_a_bincount_reference(n_classes):
    g = torch.Generator().manual_seed(0)
    preds = torch.randint(0, n_classes, (3, 16, 16), generator=g, dtype=torch.int64)
    masks = torch.randint(0, n_classes, (3, 16, 16), generator=g, dtype=torch.int64)
    masks[0, :4, :4] = IGNORE_LABEL          # ignore pixels must not be counted anywhere
    got = batch_confusion(preds, masks, n_classes)
    want = _reference(preds.numpy(), masks.numpy(), n_classes)
    for a, b, name in zip(got, want, ("inter", "union", "gt_area")):
        assert np.allclose(a.numpy(), b), name


def test_ignore_pixels_are_excluded_entirely():
    """A fully-ignored image contributes nothing — not even to gt_area."""
    preds = torch.zeros(1, 8, 8, dtype=torch.int64)
    masks = torch.full((1, 8, 8), IGNORE_LABEL, dtype=torch.int64)
    inter, union, gt = batch_confusion(preds, masks, 4)
    assert inter.sum() == 0 and union.sum() == 0 and gt.sum() == 0


def test_rows_aggregate_to_the_accumulators_miou():
    """The gate for this feature: the rows must reproduce the number the existing path
    reports, or they describe a different quantity than the mIoU they sit beside."""
    n_classes = 6
    g = torch.Generator().manual_seed(1)
    preds = torch.randint(0, n_classes, (5, 12, 12), generator=g, dtype=torch.int64)
    masks = torch.randint(0, n_classes, (5, 12, 12), generator=g, dtype=torch.int64)
    masks[2, :3] = IGNORE_LABEL

    acc = mIoUAccumulator(n_classes, IGNORE_LABEL, torch.device("cpu"))
    acc.update(preds, masks)
    inter, union, _ = batch_confusion(preds, masks, n_classes)
    assert miou_from_rows(inter, union) == pytest.approx(acc.compute(), abs=1e-12)


def test_per_row_output_is_off_by_default():
    """It must cost nothing and change nothing unless asked for."""
    from canvit_train.ade20k.config import Ade20kConfig

    assert Ade20kConfig().per_row_iou_out is None


def test_write_rows_round_trips(tmp_path):
    import pyarrow.parquet as pq

    from canvit_train.ade20k.per_row_iou import write_rows

    T, N, C = 2, 3, 4
    x = torch.arange(T * N * C, dtype=torch.float32).view(T, N, C)
    out = write_rows(tmp_path / "rows.parquet", x, x, x,
                     mask_resolution_px=512, resize_mode="squish", extra={"step": 7})
    t = pq.read_table(out)
    assert t.num_rows == T * N * C
    assert t.column("class_idx").to_pylist()[:C] == [1, 2, 3, 4]   # ADE20K is 1-indexed
    md = {k.decode(): v.decode() for k, v in t.schema.metadata.items()}
    assert md["resize_mode"] == "squish" and md["step"] == "7"
