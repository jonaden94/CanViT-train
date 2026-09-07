"""``--spec.*`` overrides: every ``TrainSpec`` combination reachable from the CLI.

Before this, ``resolve_spec(task, preset, lr, wd)`` was the only builder and ``TrainSpec``
was not tyro-exposed, so the five named presets were the whole surface. On ade20k/in1k that
left ``[head, policy]``, ``[backbone, policy]`` and ``[backbone]`` alone unreachable, plus
both grad-routing flags — even though ``TrainSpec`` expresses them and ``check_spec`` was
written for arbitrary combinations ("give all options, trust the user, warn on the
degenerate").

The two properties that matter here and are easy to get wrong:

* **a no-op override changes nothing** — the pinning digests and the phase-2 numeric gate
  are pinned to the preset specs, so ``SpecOverrides()`` must be transparent;
* **``bptt`` follows ``train_backbone``, but only when the override CHANGES it** — bptt is
  derived, so it goes stale on a flip; re-deriving it unconditionally would replace
  distill's stochastic ``chunked``/``continue_prob`` regime on a no-op.
"""

import pytest

from canvit.harness.cli import Ade20kCmd, DistillCmd, In1kCmd, SpecOverrides, resolve_spec
from canvit.harness.spec import check_spec

_PRESETS = ("default", "probe", "finetune", "policy_only", "joint")


@pytest.fixture(scope="module")
def tasks():
    return {"ade20k": Ade20kCmd().build()[0], "in1k": In1kCmd().build()[0],
            "distill": DistillCmd().build()[0]}


# --- transparency: the gate the digests depend on --------------------------------


@pytest.mark.parametrize("task_name", ["ade20k", "in1k", "distill"])
@pytest.mark.parametrize("preset", _PRESETS)
def test_no_overrides_is_a_no_op(tasks, task_name, preset):
    """``SpecOverrides()`` and ``None`` must both leave the preset's spec untouched."""
    task = tasks[task_name]
    base = resolve_spec(task, preset, 3e-4, 1e-3)
    assert resolve_spec(task, preset, 3e-4, 1e-3, None) == base
    assert resolve_spec(task, preset, 3e-4, 1e-3, SpecOverrides()) == base


def test_as_dict_only_reports_fields_that_were_set():
    assert SpecOverrides().as_dict() == {}
    assert SpecOverrides(train_policy=True).as_dict() == {"train_policy": True}
    # False is a real value, not "unset" — the None sentinel is what means unset.
    assert SpecOverrides(train_head=False).as_dict() == {"train_head": False}


# --- reachability: the three combinations presets could not express ---------------


def test_head_plus_policy_is_now_reachable(tasks):
    """The combination this whole change exists for: train the head AND the policy with a
    frozen backbone (the RL repo's ``unfreeze="probe"`` shape, structurally)."""
    spec = resolve_spec(tasks["ade20k"], "probe", 3e-4, 1e-3,
                        SpecOverrides(train_policy=True, policy_weight=1.0))
    assert spec.trainable_modules() == ("head", "policy")
    assert not spec.train_backbone
    assert check_spec(spec, tasks["ade20k"].caps()).ok
    # a newly-trainable module must arrive WITH an optimizer group, not without one
    assert {"head", "policy"} <= set(spec.optim)
    assert spec.optim["policy"].lr > 0


def test_backbone_plus_policy_is_now_reachable(tasks):
    spec = resolve_spec(tasks["ade20k"], "joint", 3e-4, 1e-3,
                        SpecOverrides(train_head=False))
    assert spec.trainable_modules() == ("backbone", "policy")
    assert check_spec(spec, tasks["ade20k"].caps()).ok


def test_every_boolean_combination_is_reachable(tasks):
    """All 7 non-empty subsets of {backbone, head, policy}, on a task that has all three."""
    task = tasks["ade20k"]
    seen = set()
    for bb in (False, True):
        for hd in (False, True):
            for pol in (False, True):
                if not (bb or hd or pol):
                    continue  # the empty set is rejected by design, not unreachable
                spec = resolve_spec(task, "joint", 3e-4, 1e-3, SpecOverrides(
                    train_backbone=bb, train_head=hd, train_policy=pol,
                    policy_weight=1.0 if pol else 0.0,
                    task_grad_to_backbone=bb, policy_grad_to_backbone=False))
                assert check_spec(spec, task.caps()).ok, spec
                seen.add(spec.trainable_modules())
    assert len(seen) == 7


# --- bptt: derived, and only re-derived on an actual change ----------------------


def test_bptt_is_rederived_when_train_backbone_flips_on(tasks):
    """`--preset probe --spec.train-backbone True` must not leave bptt='none' behind: that
    would train the backbone with no cross-timestep credit at all."""
    probe = resolve_spec(tasks["ade20k"], "probe", 3e-4, 1e-3)
    assert probe.bptt.mode == "none"
    spec = resolve_spec(tasks["ade20k"], "probe", 3e-4, 1e-3,
                        SpecOverrides(train_backbone=True, task_grad_to_backbone=True))
    assert spec.bptt.mode == "full", "bptt went stale when the override flipped the backbone on"


def test_bptt_is_rederived_when_train_backbone_flips_off(tasks):
    spec = resolve_spec(tasks["ade20k"], "finetune", 3e-4, 1e-3,
                        SpecOverrides(train_backbone=False, task_grad_to_backbone=False))
    assert spec.bptt.mode == "none"


def test_a_noop_backbone_override_preserves_distills_stochastic_bptt(tasks):
    """distill's default bptt is 'chunked' with a continue_prob that ``fixed_horizon_bptt``
    cannot express. Re-deriving bptt on an override that does NOT change train_backbone
    would silently replace distill's training regime, so the guard is on the VALUE."""
    base = resolve_spec(tasks["distill"], "default", 3e-4, 1e-3)
    assert base.train_backbone and base.bptt.mode == "chunked"
    spec = resolve_spec(tasks["distill"], "default", 3e-4, 1e-3,
                        SpecOverrides(train_backbone=True, policy_weight=0.0))
    assert spec.bptt == base.bptt, "a no-op train_backbone override re-derived bptt"


# --- the guards still fire on override-induced nonsense --------------------------


def test_incoherent_override_is_rejected(tasks):
    """train_policy without a policy weight trains nothing — a hard error, not a warning."""
    spec = resolve_spec(tasks["ade20k"], "probe", 3e-4, 1e-3, SpecOverrides(train_policy=True))
    report = check_spec(spec, tasks["ade20k"].caps())
    assert not report.ok
    assert any("policy_weight == 0" in e for e in report.errors)


def test_override_cannot_route_a_loss_into_a_frozen_backbone(tasks):
    spec = resolve_spec(tasks["ade20k"], "probe", 3e-4, 1e-3,
                        SpecOverrides(task_grad_to_backbone=True))
    report = check_spec(spec, tasks["ade20k"].caps())
    assert not report.ok
    assert any("frozen backbone" in e for e in report.errors)


def test_override_cannot_add_a_head_to_a_headless_task(tasks):
    spec = resolve_spec(tasks["distill"], "default", 3e-4, 1e-3, SpecOverrides(train_head=True))
    report = check_spec(spec, tasks["distill"].caps())
    assert not report.ok
    assert any("no head" in e for e in report.errors)


def test_trainable_backbone_with_bptt_none_warns(tasks):
    """The mirror of the frozen-backbone bptt warning, and the case an override can reach
    when bptt is pinned deliberately. Previously silent."""
    from dataclasses import replace

    spec = resolve_spec(tasks["ade20k"], "finetune", 3e-4, 1e-3)
    pinned = replace(spec, bptt=replace(spec.bptt, mode="none"))
    report = check_spec(pinned, tasks["ade20k"].caps())
    assert any("bptt.mode='none'" in w for w in report.warnings)


def test_no_preset_trips_the_new_warning(tasks):
    """Every shipped preset must be free of the new warning, or it is noise."""
    for name, task in tasks.items():
        for preset in _PRESETS:
            spec = resolve_spec(task, preset, 3e-4, 1e-3)
            report = check_spec(spec, task.caps())
            assert not any("bptt.mode='none'" in w for w in report.warnings), (name, preset)
