"""``harness.evaluate``: the standalone entry point.

The load-bearing claim is proven on the cluster, not here — standalone reproduces
training-time validation BIT-IDENTICALLY on all three tasks (eval-merge doc §5, Stage 3,
`stage0_baseline/gate3b_*.json`). What unit tests can pin is the surface: that a missing
policy is refused rather than guessed, that every flag actually reaches its config, and that
the record says which protocol produced the number.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest
import tyro

from canvit_train.harness.evaluate import Command, EvalOpts, _resolved_protocol, evaluate


@dataclass
class _Dummy:
    """Stands in for a task: the policy assertion fires before anything is built."""

    def build_model(self, device, prior_model_config=None):  # pragma: no cover
        raise AssertionError("must not be reached")


def test_auto_is_refused_and_the_message_names_the_hazard():
    """`auto` is the TRAINING-time table, kept for exp22-exp36 comparability. A standalone
    measurement has no history to protect, and the wrong guess here is SILENT — the metric
    falls as glimpses accumulate instead of raising."""
    from canvit_train.ade20k.config import Ade20kConfig

    cfg = Ade20kConfig()
    assert cfg.eval_policy == "auto", "the task default must stay 'auto' for training"
    with pytest.raises(AssertionError) as e:
        evaluate(_Dummy(), cfg, EvalOpts(ckpt=Path("nope.pt")), task_name="ade20k")
    msg = str(e.value)
    assert "--cfg.eval-policy is required" in msg
    assert "OUT OF DISTRIBUTION" in msg and "FALLS" in msg


@pytest.mark.parametrize("sub,extra", [
    ("ade20k", ["--cfg.resize-mode", "squish"]),
    ("in1k", ["--cfg.mode", "finetune"]),
    ("distill", []),
])
def test_every_flag_reaches_its_config(sub, extra):
    """tyro over each task's OWN dataclass, the same idiom as harness.run. If a subcommand
    stopped exposing `--cfg.*`, the flags would be silently unavailable rather than error."""
    cmd = tyro.cli(Command, args=[
        sub, "--opts.ckpt", "/tmp/x.pt", "--cfg.eval-policy", "fixation_grid", *extra,
    ])
    assert cmd.opts.ckpt == Path("/tmp/x.pt")
    assert cmd.cfg.eval_policy == "fixation_grid"
    task, name = cmd.build()
    assert name == sub and task.cfg is cmd.cfg


def test_override_scale_is_reachable_on_all_three():
    """F4: it used to be ade20k-only, and in1k is the task whose historical default is the
    off-scale one."""
    for sub in ("ade20k", "in1k", "distill"):
        cmd = tyro.cli(Command, args=[
            sub, "--opts.ckpt", "/tmp/x.pt", "--cfg.eval-policy", "full",
            "--cfg.eval-override-scale", "2.0",
        ])
        assert cmd.cfg.eval_override_scale == 2.0, sub


def test_the_record_says_which_protocol_produced_the_number():
    """Provenance, not restriction, is what protects comparability: any executable
    combination is allowed, so the artifact must name the one that ran."""
    from canvit_train.ade20k.config import Ade20kConfig

    cfg = Ade20kConfig(eval_policy="full", eval_override_scale=2.0)
    cfg.foveated_scale.fixed_scale = 2.0
    p = _resolved_protocol(cfg, "ade20k", is_foveated=True)
    assert p["eval_policy"] == "full" and p["eval_policy_requested"] == "full"
    assert p["override_scale"] == 2.0 and p["is_foveated"] is True
    assert p["foveated_scale"]["fixed_scale"] == 2.0
    assert p["n_timesteps"] == cfg.n_timesteps and p["scene_size"] == cfg.scene_size


def test_auto_is_recorded_as_what_it_resolved_to():
    """A record saying 'auto' would be useless a year later; it must say the trajectory."""
    from canvit_train.in1k.config import In1kConfig

    p = _resolved_protocol(In1kConfig(), "in1k", is_foveated=True)
    assert p["eval_policy"] == "coarse_to_fine"      # HISTORICAL_DEFAULTS["in1k"], foveated
    assert p["eval_policy_requested"] == "auto"
