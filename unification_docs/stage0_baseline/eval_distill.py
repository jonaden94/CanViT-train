"""STAGE-0 THROWAWAY driver: evaluate a distill harness checkpoint. Not repo code.

Rebuilds the model from the checkpoint's OWN `model_config` (build_model's
`prior_model_config` path, i.e. what resume does), so no launcher EXTRA_ARGS are needed.
Builds ONLY the ImageFolder val loader -- `task.build_loaders` would also construct the
WebDataset train loader and check the resume schedule invariants, neither of which eval
uses. The normalizer stats ride in `model_state`, so they arrive from the checkpoint.
"""
import argparse, json, logging
from pathlib import Path
import torch

from canvit_train.distill.config import Config
from canvit_train.distill.data import _create_imagefolder_val_loader
from canvit_train.distill.task import DistillRunTask

p = argparse.ArgumentParser()
p.add_argument("--ckpt", type=Path, required=True)
p.add_argument("--val-dir", type=Path, required=True)
p.add_argument("--val-index-dir", type=Path, default=None)
p.add_argument("--eval-policy", default="auto")
p.add_argument("--fixed-scale", type=float, default=None)
p.add_argument("--out", type=Path, default=None)
a = p.parse_args()

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

payload = torch.load(a.ckpt, weights_only=False, map_location="cpu")
md = payload.get("metadata") or {}

kw = dict(val_dir=a.val_dir, eval_policy=a.eval_policy, tracker="none",
          webdataset_dir=Path("/nonexistent"))
if a.val_index_dir is not None:
    kw["val_index_dir"] = a.val_index_dir
cfg = Config(**kw)
fs = md.get("foveated_scale") or {}
if a.fixed_scale is not None:
    cfg.foveated_scale.mode = "fixed"; cfg.foveated_scale.fixed_scale = a.fixed_scale
elif fs:
    cfg.foveated_scale.mode = fs["mode"]; cfg.foveated_scale.fixed_scale = fs["fixed_scale"]
    cfg.foveated_scale.min_scale = fs["min_scale"]; cfg.foveated_scale.max_scale = fs["max_scale"]
    cfg.foveated_scale.distribution = fs["distribution"]
if md.get("teacher_name"):
    cfg.teacher_name = md["teacher_name"]

task = DistillRunTask(cfg)
model, _ = task.build_model(device, prior_model_config=payload["model_config"])
model.load_state_dict(payload["model_state"])
model.eval()
assert task.scene_norm.initialized, "normalizer stats did not arrive from the checkpoint"

val_loader = _create_imagefolder_val_loader(cfg)
step = int(payload.get("step", 0))
metrics = task.evaluate(model=model, head=None, val_loader=val_loader, device=device, step=step)
rec = {"ckpt": str(a.ckpt), "step": step, "eval_policy": a.eval_policy,
       "foveated_scale": {"mode": cfg.foveated_scale.mode, "fixed_scale": cfg.foveated_scale.fixed_scale},
       "metrics": {k: float(v) for k, v in metrics.items()}}
print(json.dumps(rec, indent=2))
if a.out:
    a.out.write_text(json.dumps(rec, indent=2))
