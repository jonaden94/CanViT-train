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

    ce  = F.cosine_similarity(pred,     raw_p,  dim=-1).mean().item()   # canvit_eval
    tr  = F.cosine_similarity(pred_raw, raw_p,  dim=-1).mean().item()   # canvit_train
    nrm = F.cosine_similarity(pred,     norm_p, dim=-1).mean().item()   # both agree here
print(f"batch of {B}, t9, exp32-fovi step-1916928")
print(f"  scene_cos_raw  canvit_eval (no destandardize) = {ce:.6f}")
print(f"  scene_cos_raw  canvit_train (destandardized)  = {tr:.6f}")
print(f"  difference                                    = {tr - ce:+.6f}")
print(f"  scene_cos_norm (identical in both)            = {nrm:.6f}")
