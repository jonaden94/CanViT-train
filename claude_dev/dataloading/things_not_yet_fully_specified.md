# Things not yet fully specified

This file documents design choices and open questions that `plan_dataloading/dataloading.md`, `plan_dataloading/dataset_structure.md` and `plan_dataloading/ddp` do not resolve. Make adequate decisions on these points independently, informed by your knowledge about the repository where you are implementing the dataloading logic.

## 1. What the training loop actually consumes per batch

`plan_dataloading/dataset_structure.md` shows that each sample contains `jpg`, `json`, `cls.npy`, and `ptch.npy`. However, **the training step may not use all of these**. For example, if the training objective is purely feature distillation with no classification loss, `json` (which contains the class label) is not needed during training. Determine exactly which fields are needed, and only decode/load those. Do not over-engineer the decode pipeline by assuming all fields are always needed.

## 2. Shard schedule generation

`plan_dataloading/dataloading.md` shows pseudocode for building `all_shards_tiled` but does not specify:
- Whether this should be a **standalone script** run once before job 0, or generated inside the training script on first run
- Where `shard_schedule.npy` should be saved (next to the dataset? in the training run directory?)

Without knowing anything about the repository where the dataloading should be implemented, a **standalone script** seems like a good idea, which is run once before training begins and saves `shard_schedule.npy` to a well-defined location (e.g. the training run directory or alongside the dataset). The training script then loads it at startup. This keeps the schedule deterministic and auditable.

## 3. Checkpoint integration

`plan_dataloading/dataloading.md` says "track `job_index` in your checkpoint" but does not specify the existing checkpoint format. Read the existing checkpointing code to determine how to add `job_index` without breaking existing checkpoint loading.

## 4. Feature dtype conversion

Features are stored as `float16` (see `plan_dataloading/dataset_structure.md`). Whether they need to be cast to `float32` before being fed into the loss depends on the training setup (mixed precision, loss function, etc.). Check the existing training code and cast appropriately.

## Further issues
There are probably many more things to figure out along the way. This `plan_dataloading` folder is only meant to provide a concrete goal and not the concrete implementation details.
