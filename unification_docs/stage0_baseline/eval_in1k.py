"""STAGE-0 THROWAWAY driver: evaluate an in1k harness checkpoint. Not repo code.

Mirrors scripts/eval_ade20k_checkpoint.py: rebuild the model through the in1k task,
load the checkpoint's weights strictly, build ONLY the val loader (build_loaders would
also spin up the WebDataset train loader, which eval does not need), call task.evaluate.
"""
import argparse, json, logging
from pathlib import Path
import torch

from canvit_train.in1k.config import In1kConfig
from canvit_train.in1k.data import make_val_loader
from canvit_train.in1k.task import In1kRunTask

p = argparse.ArgumentParser()
p.add_argument("--ckpt", type=Path, required=True)
p.add_argument("--model-repo", required=True)
p.add_argument("--probe-repo", default=None)
p.add_argument("--mode", default="finetune")
p.add_argument("--eval-policy", default="random")
p.add_argument("--n-timesteps", type=int, default=4)
p.add_argument("--fixed-scale", type=float, default=None)
p.add_argument("--min-vp-scale", type=float, default=0.05)
p.add_argument("--eval-batch-size", type=int, default=64)
p.add_argument("--num-workers", type=int, default=8)
p.add_argument("--limit-val-batches", type=int, default=None)
p.add_argument("--out", type=Path, default=None)
a = p.parse_args()

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = In1kConfig(
    model_repo=a.model_repo, probe_repo=a.probe_repo, mode=a.mode,
    n_timesteps=a.n_timesteps, eval_policy=a.eval_policy, min_vp_scale=a.min_vp_scale,
    eval_batch_size=a.eval_batch_size, num_workers=a.num_workers,
    limit_val_batches=a.limit_val_batches, tracker="none",
)
if a.fixed_scale is not None:
    cfg.foveated_scale.mode = "fixed"
    cfg.foveated_scale.fixed_scale = a.fixed_scale

task = In1kRunTask(cfg)
model, head = task.build_model(device)
payload = torch.load(a.ckpt, weights_only=False, map_location=device)
model.load_state_dict(payload["model_state"], strict=True)
model.eval()

val_loader = make_val_loader(cfg, world_size=1, rank=0)
metrics = task.evaluate(model=model, head=head, val_loader=val_loader, device=device,
                        step=int(payload.get("step", 0)))
rec = {"ckpt": str(a.ckpt), "step": payload.get("step"), "model_repo": a.model_repo,
       "probe_repo": a.probe_repo, "mode": a.mode, "eval_policy": a.eval_policy,
       "n_timesteps": a.n_timesteps, "fixed_scale": a.fixed_scale,
       "eval_batch_size": a.eval_batch_size,
       "metrics": {k: float(v) for k, v in metrics.items()}}
print(json.dumps(rec, indent=2))
if a.out:
    a.out.write_text(json.dumps(rec, indent=2))
