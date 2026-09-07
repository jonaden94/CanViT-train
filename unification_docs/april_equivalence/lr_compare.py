"""April's SequentialLR(LinearLR->ConstantLR) vs today's LambdaLR(_lr_lambda),
distill's production recipe: warmup_steps=100_000, peak_lr=4e-4, start_lr=1e-7."""
import torch
from torch import nn
from canvit.harness.optim.scheduler import warmup_constant_scheduler   # byte-identical to April's
from canvit.harness.optim.build import _lr_lambda
from canvit.harness.spec import ScheduleSpec

WARM, PEAK, START = 100_000, 4e-4, 1e-7
N = 120_000

p1 = nn.Parameter(torch.zeros(1)); p2 = nn.Parameter(torch.zeros(1))
o_old = torch.optim.AdamW([p1], lr=PEAK)
s_old = warmup_constant_scheduler(o_old, WARM, PEAK, start_lr=START)          # April's path
o_new = torch.optim.AdamW([p2], lr=PEAK)
s_new = torch.optim.lr_scheduler.LambdaLR(
    o_new, lr_lambda=_lr_lambda(ScheduleSpec(kind="warmup_constant", warmup_steps=WARM,
                                             start_lr=START), PEAK))          # today's path

worst = 0.0; worst_step = -1; n_diff = 0
first_diff = None
for step in range(N):
    a = o_old.param_groups[0]["lr"]; b = o_new.param_groups[0]["lr"]
    d = abs(a - b)
    if d > 0:
        n_diff += 1
        if first_diff is None: first_diff = (step, a, b)
    rel = d / max(a, 1e-30)
    if rel > worst: worst, worst_step = rel, step
    o_old.step(); o_new.step(); s_old.step(); s_new.step()

print(f"steps compared: {N}")
print(f"steps with ANY difference: {n_diff}")
print(f"worst RELATIVE difference: {worst:.3e} at step {worst_step}")
if first_diff: print(f"first difference: step {first_diff[0]}: april={first_diff[1]:.12e} today={first_diff[2]:.12e}")
for s in (0, 1, 2, 50_000, 99_998, 99_999, 100_000, 100_001, 110_000):
    pass
