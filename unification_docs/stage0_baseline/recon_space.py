"""Do canvit_eval's and distill's `scene_cos_raw` mean the same thing? Measure it."""
import torch, torch.nn.functional as F
from pathlib import Path
from canvit_train.distill.config import Config
from canvit_train.distill.data import create_imagefolder_val_loader
from canvit_train.distill.task import DistillRunTask
from canvit_train.harness.rollout.eval_viewpoints import open_loop_viewpoints

CKPT = Path("logs/jon_exp32_pretrain_lrdrop/exp32-fovi/checkpoints/step-1916928.pt")
dev = torch.device("cuda")
payload = torch.load(CKPT, weights_only=False, map_location="cpu")
md = payload["metadata"]; fsm = md["foveated_scale"]

cfg = Config(val_dir=Path("/mnt/vast-nhr/projects/nib00021/jonathan/datasets/imagenet1k-val"),
             val_index_dir=Path("/mnt/vast-nhr/projects/nib00021/jonathan/repos/_data_cache"),
             webdataset_dir=Path("/nonexistent"), tracker="none")
for k in ("mode", "distribution", "fixed_scale", "min_scale", "max_scale"):
    setattr(cfg.foveated_scale, k, fsm[k])
task = DistillRunTask(cfg)
model, _ = task.build_model(dev, prior_model_config=payload["model_config"])
model.load_state_dict(payload["model_state"]); model.eval()
teacher = task._teacher_for_forward(dev)
sz = task._scene_size_px()

images, _ = next(iter(create_imagefolder_val_loader(cfg)._loader))
images = images.to(dev)
B = images.shape[0]
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    feats = teacher.forward_norm_features(
        F.interpolate(images, size=(sz, sz), mode="bilinear", align_corners=False)
        if images.shape[-1] != sz else images)
    raw_p, raw_c = feats.patches.float(), feats.cls.float()
    norm_p = task.scene_norm(raw_p)
    vps = open_loop_viewpoints("fixation_grid", batch_size=B, device=dev, n=10,
                               is_foveated=True, foveated_scale=cfg.foveated_scale,
                               foveated_eval_scale=cfg.foveated_scale.fixed_scale)
    acc, _ = model.forward_reduce(
        image=images, viewpoints=vps, canvas_grid_size=cfg.canvas_patch_grid_size,
        init_fn=lambda st: [], step_fn=lambda a, out, vp: a + [out.state.canvas])
    canvas = acc[-1]                                   # t9
    pred = model.predict_teacher_scene(canvas)         # normalized space
    pred_raw = task.scene_norm.destandardize(pred)     # -> raw space

    # WHICH SPACE does predict_teacher_scene output? Decide from the moments, not from
    # reading the training code: normalized space => per-channel mean~0, std~1.
    def moments(x, name):
        f = x.reshape(-1, x.shape[-1]).float()
        print(f"  {name:34} per-channel mean={f.mean(0).mean():+.4f}  "
              f"std={f.std(0).mean():.4f}  |vec|={f.norm(dim=-1).mean():.2f}")

    print("\nMOMENTS (per-channel, averaged over channels):")
    moments(raw_p,    "raw teacher patches")
    moments(norm_p,   "normalized teacher (scene_norm)")
    moments(pred,     "predict_teacher_scene output")
    moments(task.scene_norm.destandardize(pred), "destandardize(pred)")

    print("\nCOSINES:")
    for lbl, a, b in [
        ("pred            vs norm_teacher", pred, norm_p),
        ("destandardize(pred) vs raw_teacher", task.scene_norm.destandardize(pred), raw_p),
        ("pred            vs raw_teacher  (canvit_eval)", pred, raw_p),
        ("destandardize(pred) vs norm_teacher (also mismatched)",
         task.scene_norm.destandardize(pred), norm_p),
    ]:
        print(f"  {lbl:56} {F.cosine_similarity(a, b, dim=-1).mean().item():.6f}")

