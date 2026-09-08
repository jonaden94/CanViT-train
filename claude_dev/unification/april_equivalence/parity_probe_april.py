"""April-side mirror of claude_dev/unification/parity_probe.py.

Same tiny CPU model, same pinned RNG, same 25 synthetic steps -- but driving
canvit_pretrain/train/step.py::training_step at 1a36eec instead of today's run_rollout.
Re-derives the digest that parity_probe.py's docstring says can no longer be re-derived
"because that implementation is gone". It is not gone; it is in git history.
"""
import hashlib, json, random, sys
from contextlib import nullcontext

import torch
from canvit_pretrain import CanViTForPretraining, CanViTForPretrainingConfig
from canvit_pretrain.train.step import training_step
from canvit_pretrain.train.viewpoint import ViewpointType  # noqa: F401  (parity of import side effects)
from canvit_pytorch import create_backbone

_B, _G, _D = 2, 8, 384
_N_STEPS = 25
_DEVICE = torch.device("cpu")


def _build_model():
    torch.manual_seed(1234)
    backbone = create_backbone("vits16").to(_DEVICE)
    cfg = CanViTForPretrainingConfig(teacher_dim=_D)
    return CanViTForPretraining(
        backbone=backbone, cfg=cfg, backbone_name="vits16", canvas_patch_grid_sizes=[_G],
    ).to(_DEVICE)


def main() -> None:
    torch.use_deterministic_algorithms(True)
    model = _build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    random.seed(4321)
    torch.manual_seed(5678)
    losses: list[str] = []
    for _ in range(_N_STEPS):
        images = torch.randn(_B, 3, 224, 224, device=_DEVICE)
        scene_target = torch.randn(_B, _G * _G, _D, device=_DEVICE)
        cls_target = torch.randn(_B, _D, device=_DEVICE)
        raw_scene_target = torch.randn(_B, _G * _G, _D, device=_DEVICE)
        raw_cls_target = torch.randn(_B, _D, device=_DEVICE)

        opt.zero_grad()
        m = training_step(
            model=model, images=images,
            scene_target=scene_target, cls_target=cls_target,
            raw_scene_target=raw_scene_target, raw_cls_target=raw_cls_target,
            scene_denorm=lambda x: x, cls_denorm=lambda x: x,
            enable_scene_patches_loss=True, enable_scene_cls_loss=True,
            glimpse_size_px=128, canvas_grid_size=_G,
            n_full_start_branches=1, n_random_start_branches=1,
            chunk_size=2, continue_prob=0.5, min_viewpoint_scale=0.1,
            amp_ctx=nullcontext(),
        )
        opt.step()
        losses.append(m.total_loss.item().hex())

    digest = hashlib.sha256("".join(losses).encode()).hexdigest()[:16]
    json.dump({"source": "1a36eec canvit_pretrain/train/step.py", "n_steps": _N_STEPS,
               "torch": torch.__version__, "losses_hex": losses, "digest": digest},
              open(sys.argv[1], "w"), indent=1)
    print(f"APRIL loss-stream sha256[:16] = {digest}")


if __name__ == "__main__":
    main()
