"""CPU tests for the checkpoint → HF-format converter's pure logic."""

import json
from pathlib import Path

import pytest

from .to_hf import build_config, extract_pretrain_view_scale, normalize_schema


def _raw(patcher: str, history: dict | None) -> dict:
    return {
        "backbone_name": "vitb16",
        "model_config": {"patcher_name": patcher, "teacher_dim": 768},
        "canvas_patch_grid_sizes": [32],
        "glimpse_grid_size": 8,
        "step": 200_000,
        "teacher_name": "dinov3_vitb16",
        "dataset": "in21k",
        "timestamp": "2026-02-02T00:00:00+00:00",
        "git_commit": "abc123",
        "state_dict": {},
        **({"training_config_history": history} if history is not None else {}),
    }


def _history(mode: str, fixed_scale: float) -> dict:
    return {
        "2026-02-01T00:00:00+00:00": {
            "foveated_scale.mode": mode,
            "foveated_scale.distribution": "loguniform",
            "foveated_scale.fixed_scale": fixed_scale,
            "foveated_scale.min_scale": 0.5,
            "foveated_scale.max_scale": 2.0,
        }
    }


def test_extract_fixed_foveated_scale():
    vs = extract_pretrain_view_scale(_raw("foveated", _history("fixed", 2.0)))
    assert vs == {
        "patcher_name": "foveated",
        "mode": "fixed",
        "distribution": "loguniform",
        "fixed_scale": 2.0,
        "min_scale": 0.5,
        "max_scale": 2.0,
    }


def test_extract_square_multiscale():
    vs = extract_pretrain_view_scale(_raw("square", _history("per_rollout", 1.0)))
    assert vs is not None and vs["patcher_name"] == "square" and vs["mode"] == "per_rollout"


def test_extract_uniform_is_none():
    # Uniform's OOD axis is glimpse crop pixels, not view-scale → not recorded.
    assert extract_pretrain_view_scale(_raw("uniform", _history("fixed", 2.0))) is None


def test_extract_no_history_is_none():
    # Older checkpoints predating training_config_history: unknown, not "1.0".
    assert extract_pretrain_view_scale(_raw("foveated", None)) is None


def test_extract_latest_history_entry_wins():
    hist = {
        "2026-01-01T00:00:00+00:00": {"foveated_scale.mode": "fixed", "foveated_scale.fixed_scale": 1.0},
        "2026-03-01T00:00:00+00:00": {"foveated_scale.mode": "fixed", "foveated_scale.fixed_scale": 2.0},
    }
    vs = extract_pretrain_view_scale(_raw("foveated", hist))
    assert vs is not None and vs["fixed_scale"] == 2.0


def test_build_config_embeds_view_scale():
    cfg = build_config(_raw("foveated", _history("fixed", 2.0)), Path("/x/step-200000.pt"))
    assert cfg["metadata"]["pretrain_view_scale"]["fixed_scale"] == 2.0
    assert cfg["backbone_name"] == "vitb16"
    assert cfg["metadata"]["teacher_name"] == "dinov3_vitb16"
    # Uniform → explicit None so eval treats it as "not applicable", not "unknown scare".
    cfg_u = build_config(_raw("uniform", None), Path("/x/step.pt"))
    assert cfg_u["metadata"]["pretrain_view_scale"] is None


# --- harness-schema checkpoints (nested metadata) --------------------------
def _harness_raw(patcher: str, *, patch_stride: int | None = None) -> dict:
    """A checkpoint shaped exactly as the harness writes it.

    Built by calling the REAL `DistillRunTask.model_config()` / `checkpoint_metadata()`
    rather than hand-writing the dicts — a hand-written fixture silently encoded the
    wrong `model_config` shape (flat instead of the harness's `{"canvit": {...}}`) and
    hid a live footgun that only showed up converting an actual checkpoint.
    """
    from types import SimpleNamespace

    from canvit.distill.config import Config
    from canvit.distill.task import DistillRunTask
    from canvit.harness.config import FoveatedScaleConfig

    cfg = Config(webdataset_dir=Path("/nonexistent"), patch_stride=patch_stride)
    cfg.model.patcher_name = patcher
    cfg.foveated_scale = FoveatedScaleConfig(mode="fixed", fixed_scale=2.0,
                                             distribution="uniform", min_scale=0.5, max_scale=1.0)
    task = DistillRunTask(cfg)
    stub = SimpleNamespace(cfg=cfg.model, canvas_patch_grid_sizes=[32])
    task_meta = task.checkpoint_metadata(stub)
    return {
        "step": 200_000,
        "model_state": {},
        "model_config": task.model_config(stub),
        "metadata": {**task_meta,
                     "training_config_history": {"2026-02-01T00:00:00+00:00": task_meta}},
    }


def test_normalize_leaves_legacy_untouched():
    legacy = _raw("foveated", _history("fixed", 2.0))
    assert normalize_schema(legacy) is legacy


def test_harness_checkpoint_converts_with_view_scale():
    """The footgun regression: before the shim this silently produced
    pretrain_view_scale=None because training_config_history was not top-level."""
    cfg = build_config(normalize_schema(_harness_raw("foveated")), Path("/x/step-200000.pt"))
    assert cfg["metadata"]["pretrain_view_scale"] == {
        "patcher_name": "foveated", "mode": "fixed", "distribution": "uniform",
        "fixed_scale": 2.0, "min_scale": 0.5, "max_scale": 1.0,
    }
    assert cfg["backbone_name"] == "vitb16"
    assert cfg["canvas_patch_grid_sizes"] == [32]
    assert cfg["glimpse_grid_size"] == 8


def test_harness_checkpoint_preserves_patch_stride():
    """Overlapping-patch models (exp21) are unrebuildable without patch_stride."""
    cfg = build_config(normalize_schema(_harness_raw("uniform", patch_stride=8)), Path("/x/s.pt"))
    assert cfg["patch_stride"] == 8
    # non-overlapping stays absent, so the config is byte-identical to before
    cfg_n = build_config(normalize_schema(_harness_raw("uniform")), Path("/x/s.pt"))
    assert "patch_stride" not in cfg_n


def test_harness_non_distill_checkpoint_rejected():
    """An ade20k checkpoint is not a pretraining model — fail loudly, not silently.

    in1k no longer lands here: it dispatches to the classifier layout instead (below)."""
    bad = {"model_state": {}, "model_config": {}, "metadata": {"task": "ade20k"}}
    with pytest.raises(KeyError, match="backbone_name"):
        normalize_schema(bad)


# --- in1k classifier dispatch ------------------------------------------------
# The standalone in1k trainer used to write the classifier HF layout itself
# (`clf.save_pretrained(run_dir/"best-hf")`); deleting it in the consolidation left the
# harness with no HF export at all, so an in1k finetune could not be handed to
# CanViT-eval's in1k_clf task. These pin the dispatch that replaced it. The conversion
# itself needs a real backbone repo, so only the routing is unit-tested here.
def test_in1k_checkpoint_routes_to_the_classifier_layout():
    from canvit.checkpoint.to_hf import is_classifier_checkpoint

    in1k = {"model_state": {}, "step": 401_408,
            "model_config": {"task": "in1k", "n_classes": 1000, "canvas_grid": 32,
                             "model_repo": "/some/hf/dir", "mode": "finetune"},
            "metadata": {"task": "in1k"}}
    assert is_classifier_checkpoint(in1k)


def test_distill_and_ade20k_do_not_route_to_the_classifier_layout():
    """Misrouting a distill checkpoint would try to build a 1000-class classifier from it."""
    from canvit.checkpoint.to_hf import is_classifier_checkpoint

    assert not is_classifier_checkpoint(_harness_raw("uniform"))
    assert not is_classifier_checkpoint(_raw("uniform", _history("fixed", 1.0)))
    assert not is_classifier_checkpoint(
        {"model_state": {}, "model_config": {"task": "ade20k"}, "metadata": {}})


# --- the classifier layout's metadata block ----------------------------------
# `save_pretrained` records only the __init__ kwargs, so before 2026-09-01 a published
# classifier carried NO metadata and CanViT-eval's resolve_view_scale /
# teacher_probe_for_model were both inert on it (eval-merge doc §5, F6b). The note above
# said "the conversion itself needs a real backbone repo, so only the routing is
# unit-tested" — which is exactly how the artifact stayed broken. `classifier_metadata`
# needs no model, only a config.json, so it gets tested properly.

def _backbone_dir(tmp_path, *, view_scale, teacher_name="dinov3_vitb16"):
    """A minimal HF-layout dir: `read_pretrain_metadata` reads config.json, nothing else."""
    d = tmp_path / "backbone"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(
        {"backbone_name": "vitb16", "model_config": {},
         "metadata": {"teacher_name": teacher_name, "pretrain_view_scale": view_scale}}))
    return d


def _in1k_raw(*, patcher, repo, recorded=None):
    return {"model_state": {}, "step": 400_000,
            "model_config": {"task": "in1k", "n_classes": 1000, "canvas_grid": 32,
                             "model_repo": str(repo), "mode": "finetune",
                             "canvit": {"patcher_name": patcher}},
            "metadata": {"task": "in1k", "mode": "finetune", "pretrain_view_scale": recorded}}


def test_ade20k_legacy_float_view_scale_is_coerced_to_the_dict_form():
    """Every ade20k checkpoint before 2026-09-01 recorded a BARE FLOAT. A reader written
    against the published dict form is a silent no-op on it, which is how a foveated model
    gets evaluated at the policy's own scales."""
    from canvit.checkpoint.to_hf import _as_view_scale_dict

    got = _as_view_scale_dict(2.0, "foveated")
    assert got is not None and got["mode"] == "fixed" and got["fixed_scale"] == 2.0
    assert got["patcher_name"] == "foveated"


def test_uniform_never_reports_a_view_scale():
    """A uniform model's OOD axis is the glimpse crop in pixels, not the view scale, and
    ade20k used to record 1.0 for it unconditionally. `None` means unknown and must never
    be read as 1.0, so emitting 1.0 here would poison exactly that contract."""
    from canvit.checkpoint.to_hf import _as_view_scale_dict

    assert _as_view_scale_dict(1.0, "uniform") is None
    assert _as_view_scale_dict({"mode": "fixed", "fixed_scale": 1.0}, "uniform") is None


def test_in1k_view_scale_is_recovered_from_the_backbone_repo(tmp_path):
    """exp25/exp29/exp33 record the view scale NOWHERE — not in metadata, not in
    training_config_history. The backbone they were built on does, so old checkpoints are
    recoverable and this is not only a fix going forward."""
    from canvit.checkpoint.to_hf import classifier_metadata

    repo = _backbone_dir(tmp_path, view_scale={
        "patcher_name": "foveated", "mode": "fixed", "fixed_scale": 2.0})
    md = classifier_metadata(_in1k_raw(patcher="foveated", repo=repo), tmp_path / "best.pt")
    assert md["pretrain_view_scale"]["fixed_scale"] == 2.0
    # never recorded on a downstream checkpoint, so it always comes from the backbone
    assert md["teacher_name"] == "dinov3_vitb16"
    assert md["task"] == "in1k" and md["step"] == 400_000


def test_unknown_view_scale_stays_none_and_warns(tmp_path, caplog):
    """A backbone path from another machine must not silently become a guessed scale."""
    from canvit.checkpoint.to_hf import classifier_metadata

    with caplog.at_level("WARNING"):
        md = classifier_metadata(
            _in1k_raw(patcher="foveated", repo=tmp_path / "gone"), tmp_path / "best.pt")
    assert md["pretrain_view_scale"] is None and md["teacher_name"] is None
    assert "out of distribution" in caplog.text


def test_uniform_classifier_needs_no_view_scale_and_does_not_warn(tmp_path, caplog):
    from canvit.checkpoint.to_hf import classifier_metadata

    repo = _backbone_dir(tmp_path, view_scale=None)
    with caplog.at_level("WARNING"):
        md = classifier_metadata(_in1k_raw(patcher="uniform", repo=repo), tmp_path / "best.pt")
    assert md["pretrain_view_scale"] is None
    assert "out of distribution" not in caplog.text
