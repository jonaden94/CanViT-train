"""Convert a CanViT-train ``.pt`` checkpoint into the local HF Hub layout that
``CanViTForPretrainingHFHub.from_pretrained(<dir>)`` reads.

Ported from the (now-archived) ``CanViT-specialize/scripts/pretrain_ckpt_to_hf_format.py``
into the unified repo. Adds one thing the old converter lacked: an explicit
``metadata.pretrain_view_scale`` recording the foveated/square view-scale the model
was pretrained with. That scale is a *training-time viewpoint* parameter
(``FoveatedScaleConfig``) — it is NOT part of ``model_config``, so without recording it
here a downstream eval has no way to know the in-distribution glimpse scale and silently
evaluates OOD (the footgun that broke run 15025338). CanViT-eval reads this field to
auto-set its view-scale. It is derived from the checkpoint's ``training_config_history``
(the authoritative record of the training config), and is ``None`` for uniform models
(view-scale is not their OOD axis — glimpse crop pixels are).

Output dir gets:
    config.json        — backbone_name, model_config, canvas_patch_grid_sizes, glimpse_grid_size, metadata
    model.safetensors  — model state_dict

Usage:
    python -m canvit.checkpoint.to_hf --pt-path /path/to/step-NNNNNN.pt --out-dir /path/to/out

IN1k CLASSIFIER checkpoints (``metadata.task == "in1k"``) are dispatched to a second
path that emits the ``CanViTForImageClassification.from_pretrained`` layout instead —
what ``CanViT-eval/tasks/in1k_clf.py`` loads. The standalone ``in1k/train.py`` used to
write that directory itself (``clf.save_pretrained(run_dir/"best-hf")``); when it was
deleted in the harness consolidation, the harness had no HF export at all, so an in1k
finetune could not be handed to canvit_eval. Same CLI, auto-detected from the payload.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import tyro
from canvit_pytorch.checkpoint_schema import (
    SCALE_SENSITIVE_PATCHERS,
    extract_pretrain_view_scale,
    normalize_schema,
)
from canvit_pytorch.checkpoint_schema import (
    migrate_standardizers_in_place as _migrate_standardizers_in_place,
)
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

@dataclass
class Args:
    pt_path: Path
    out_dir: Path


def build_config(raw: dict, pt_path: Path) -> dict[str, Any]:
    """Assemble the config.json dict for the HF layout (pure; no I/O)."""
    config: dict[str, Any] = {
        "backbone_name": raw["backbone_name"],
        "model_config": raw["model_config"],
        "canvas_patch_grid_sizes": raw["canvas_patch_grid_sizes"],
        # Glimpse token-grid side (tokens per glimpse edge). Lets the HF-loaded
        # model / eval reconstruct the trained pixel glimpse size as
        # ``glimpse_grid_size * patch_size_px`` for any patch size. May be absent
        # in pre-this-field checkpoints (older runs) -> consumers fall back to 8.
        "glimpse_grid_size": raw.get("glimpse_grid_size"),
        "metadata": {
            "source_pt": str(pt_path),
            "step": raw.get("step"),
            "teacher_name": raw.get("teacher_name"),
            "dataset": raw.get("dataset"),
            "timestamp": raw.get("timestamp"),
            "git_commit": raw.get("git_commit"),
            # Explicit pretraining view-scale (None for uniform / unknown). See
            # extract_pretrain_view_scale + CanViT-eval config.resolve_view_scale.
            "pretrain_view_scale": extract_pretrain_view_scale(raw),
        },
    }

    # Overlapping-patch models (stride < patch_size) trained with a non-default
    # patch_stride, which lives OUTSIDE model_config (top-level training field)
    # and is needed to rebuild the patch-embed conv. Persist it ONLY when set,
    # so non-overlapping checkpoints produce a byte-for-byte identical config.
    patch_stride = raw.get("patch_stride")
    if patch_stride is not None:
        config["patch_stride"] = patch_stride
        log.info("Persisted patch_stride=%s (overlapping patches)", patch_stride)

    return config


def is_classifier_checkpoint(raw: dict) -> bool:
    """True for an IN1k classifier checkpoint, which needs the classification layout
    rather than the pretraining one. ``model_config`` is the harness in1k task's
    (``tasks/in1k/task.py::model_config``); a distill payload has no ``task`` key there."""
    return (raw.get("model_config") or {}).get("task") == "in1k"


def _as_view_scale_dict(value: Any, patcher_name: str | None) -> dict[str, Any] | None:
    """Coerce a recorded ``pretrain_view_scale`` to the canonical dict form.

    Two forms exist in the wild: the dict a distill HF export carries, and the bare FLOAT
    every ade20k checkpoint written before 2026-09-01 recorded. A reader that handles only
    the dict is a silent no-op on the float, which is how a foveated model ends up evaluated
    at the policy's own scales. Returns ``None`` for a uniform model or an unusable value.
    """
    if patcher_name not in SCALE_SENSITIVE_PATCHERS:
        return None
    if isinstance(value, dict):
        return value if value.get("mode") is not None else None
    if isinstance(value, int | float):
        # A bare float only ever meant `foveated_scale.fixed_scale`.
        return {"patcher_name": patcher_name, "mode": "fixed", "distribution": None,
                "fixed_scale": float(value), "min_scale": None, "max_scale": None}
    return None


def read_pretraining_provenance(
    raw: dict, *, source: Any = "<checkpoint>",
) -> tuple[dict[str, Any] | None, str | None]:
    """``(pretrain_view_scale, teacher_name)`` for a checkpoint, or ``(None, None)``.

    A checkpoint knows what scale it was pretrained at and which teacher supervised it.
    Making the user retype either when evaluating is how they end up mistyped — so the
    standalone evaluator reads them from here (``harness/evaluate.py``), and ``to_hf``
    writes them into the published layout.

    Resolution order, because a DOWNSTREAM checkpoint records less than a distill one:

    1. the payload's own ``metadata`` — new checkpoints, and ade20k's legacy bare float;
    2. the BACKBONE repo in ``model_config["model_repo"]``, which does carry both. This is
       the only route for exp25/exp29/exp33, whose in1k payloads record the scale nowhere.

    ``None`` means UNKNOWN and must never be read as 1.0 — a uniform model has no view
    scale (its out-of-distribution axis is the glimpse crop in pixels), and an
    undeterminable one is not the same as a known 1.0.
    """
    from canvit_pytorch.model_source import read_pretrain_metadata

    mc = raw.get("model_config") or {}
    md = raw.get("metadata") or {}
    patcher = (mc.get("canvit") or {}).get("patcher_name") or md.get("patcher_name")
    repo = mc.get("model_repo") or md.get("model_repo")

    view_scale = _as_view_scale_dict(md.get("pretrain_view_scale"), patcher)
    teacher_name = md.get("teacher_name")
    if (view_scale is None or teacher_name is None) and repo:
        try:
            backbone_md = read_pretrain_metadata(str(repo))
        except Exception as exc:  # a path from another machine, or a repo since moved
            log.warning("could not read backbone metadata from %s (%s); "
                        "pretrain_view_scale/teacher_name stay unknown", repo, exc)
            backbone_md = {}
        if view_scale is None:
            view_scale = _as_view_scale_dict(backbone_md.get("pretrain_view_scale"), patcher)
            if view_scale is not None:
                log.info("pretrain_view_scale recovered from the backbone repo %s", repo)
        teacher_name = teacher_name or backbone_md.get("teacher_name")

    if patcher in SCALE_SENSITIVE_PATCHERS and view_scale is None:
        log.warning(
            "%s is a %r-patcher checkpoint but its pretraining view scale could not be "
            "determined, so it cannot be filled in for you. Pass "
            "--cfg.foveated-scale.fixed-scale explicitly, or an evaluation may run out of "
            "distribution (measured at -0.114 top1 / -0.128 mIoU) with nothing to warn on.",
            source, patcher)
    return view_scale, teacher_name


def classifier_metadata(raw: dict, pt_path: Path) -> dict[str, Any]:
    """The ``metadata`` block for the classifier HF layout.

    ``save_pretrained`` writes only the ``__init__`` kwargs, so before this the published
    classifier carried NO metadata and CanViT-eval's ``resolve_view_scale`` /
    ``teacher_probe_for_model`` were both inert on it — a foveated finetune would be
    evaluated at the policy's own scales, measured at -0.114 top1 (eval-merge doc §5, F5/F6).

    ``pretrain_view_scale`` is resolved in two steps because in1k checkpoints written before
    2026-09-01 record it NOWHERE — not in ``metadata``, not in ``training_config_history``:

    1. the payload's own record (new checkpoints, and ade20k's float form), then
    2. the BACKBONE repo the run was built on (``model_config["model_repo"]``), which does
       carry it. That rescues exp25/exp29/exp33, so this is not only a fix going forward.

    ``teacher_name`` has no step 1 at all — a downstream checkpoint never recorded it — so it
    always comes from the backbone. Both fall back to ``None``, never to a guess: the
    contract is that ``None`` means "unknown", and a consumer must not read it as 1.0.
    """

    mc = raw.get("model_config") or {}
    md = raw.get("metadata") or {}
    repo = mc.get("model_repo") or md.get("model_repo")
    view_scale, teacher_name = read_pretraining_provenance(raw, source=pt_path)

    return {
        "source_pt": str(pt_path),
        "step": raw.get("step"),
        "task": md.get("task"),
        "mode": md.get("mode"),
        "model_repo": str(repo) if repo else None,
        "teacher_name": teacher_name,
        "pretrain_view_scale": view_scale,
    }


def classifier_to_hf(raw: dict, out_dir: Path, pt_path: Path) -> None:
    """Write the ``CanViTForImageClassification.from_pretrained`` layout.

    Rebuilds the module and reuses the class's OWN ``save_pretrained`` (from
    ``PyTorchModelHubMixin``) rather than hand-assembling config.json + safetensors —
    the layout then cannot drift from what ``from_pretrained`` expects, which is the
    whole failure mode a second writer would introduce.

    The architecture is reconstructed with ``from_pretrained_with_new_head``, whose fresh
    random head is immediately overwritten by the checkpoint's own weights. That is
    correct for BOTH modes: ``finetune`` trained from a head built by
    ``from_pretrained_with_probe``, but the *architecture* either constructor produces is
    identical (LN(D) -> Linear(D, n_classes)) — only the init differs, and we load over it.
    """
    from canvit_pytorch import CanViTForImageClassification

    mc = raw["model_config"]
    repo, n_classes = mc["model_repo"], mc["n_classes"]
    log.info("in1k classifier (mode=%s, n_classes=%d) over %s", mc.get("mode"), n_classes, repo)
    clf = CanViTForImageClassification.from_pretrained_with_new_head(
        pretrained_repo=repo, n_classes=n_classes)
    # strict: a silently-partial load would publish a half-trained classifier.
    clf.load_state_dict(raw["model_state"], strict=True)
    assert clf.head.out_features == n_classes, (
        f"head has {clf.head.out_features} classes, checkpoint says {n_classes}")
    out_dir.mkdir(parents=True, exist_ok=True)
    clf.save_pretrained(out_dir)
    # `save_pretrained` records ONLY the __init__ kwargs, so the provenance block has to be
    # merged in afterwards. Verified safe: PyTorchModelHubMixin filters config.json to the
    # __init__ signature on the way back, so an extra `metadata` key does not reach
    # __init__ — the same arrangement the pretraining layout has always used.
    cfg_path = out_dir / "config.json"
    published = json.loads(cfg_path.read_text())
    published["metadata"] = classifier_metadata(raw, pt_path)
    cfg_path.write_text(json.dumps(published, indent=2, default=str))
    log.info("Wrote %s (step=%s)", out_dir, raw.get("step"))
    log.info("pretrain_view_scale: %s", published["metadata"]["pretrain_view_scale"])
    log.info("Load with: CanViTForImageClassification.from_pretrained(%r)", str(out_dir))


def main(args: Args) -> None:
    log.info("Loading %s ...", args.pt_path)
    raw = torch.load(args.pt_path, map_location="cpu", weights_only=False)
    if is_classifier_checkpoint(raw):
        classifier_to_hf(raw, args.out_dir, args.pt_path)
        return
    raw = normalize_schema(raw)
    _migrate_standardizers_in_place(raw)

    config = build_config(raw, args.pt_path)
    vs = config["metadata"]["pretrain_view_scale"]
    log.info("pretrain_view_scale: %s", vs if vs is not None else "None (uniform / not recorded)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = args.out_dir / "config.json"
    sd_path = args.out_dir / "model.safetensors"
    cfg_path.write_text(json.dumps(config, indent=2, default=str))
    save_file(raw["state_dict"], sd_path)

    log.info("Wrote %s (%d params)", sd_path, len(raw["state_dict"]))
    log.info("Wrote %s", cfg_path)
    log.info("Load with: CanViTForPretrainingHFHub.from_pretrained(%r)", str(args.out_dir))


if __name__ == "__main__":
    main(tyro.cli(Args))
