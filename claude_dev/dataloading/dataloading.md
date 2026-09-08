# WebDataset Dataloading Design

## Overview

The dataset and dataloader should be designed to support job arrays, where training is split across multiple short SLURM jobs rather than one long run. Each job processes a fixed subset of shards, saves a checkpoint, and terminates. The next job loads the checkpoint and continues from exactly where the previous one stopped. This is necessary because wall-time limits on HPC clusters make single long-running jobs impractical.

To enable clean resumption, the dataloader must stop at shard boundaries — each job processes only whole shards, never partial ones. This means `steps_per_job` and `batch_size_per_gpu` must be chosen such that the total samples consumed per GPU is an exact multiple of `samples_per_shard`. Progress is tracked as a `job_index` in the checkpoint; each job loads `job_index` from the checkpoint, slices its shards from a precomputed schedule saved to disk, and saves `job_index + 1` at the end.

Shuffling should be handled entirely at dataset creation time via pre-shuffled tars, not during training. This avoids the need for an in-memory shuffle buffer, which would blur shard boundaries and make clean resumption impossible.

## Tar format recap

See `plan_dataloading/dataset_structure.md` for full per-sample details. In brief: each shard is a flat sequential tar of `(header, bytes)` pairs. Entries for one sample are contiguous:

```
000000000.jpg
000000000.json
000000000.cls.npy
000000000.ptch.npy
000000001.jpg
...
```

WebDataset groups consecutive entries by shared key prefix into sample dicts. There is no random-access index — reading is always sequential.

## Streaming during training

Workers stream tars entry by entry; shards are never fully loaded into memory. Each DataLoader worker is assigned a non-overlapping subset of shards (`split_by_worker`). Workers stream their assigned shards sequentially and independently.

**Do not use a shuffle buffer** (e.g. `wds.shuffle()`). A shuffle buffer blurs shard boundaries — when training stops, some samples have been read from the tar but not yet yielded, so you cannot cleanly resume from the next unprocessed shard. Pre-shuffled tars provide sufficient randomness without needing an in-memory buffer.

## Clean shard-boundary stopping (job arrays)

To stop training at a clean shard boundary and resume from the next shard in the following job:

1. Choose `steps_per_job` and `batch_size_per_gpu` such that:
   ```
   steps_per_job * batch_size_per_gpu % samples_per_shard == 0
   ```
   This gives `shards_per_gpu = (steps_per_job * batch_size_per_gpu) // samples_per_shard`.

2. Slice the shard list for each job:

   Single GPU:
   ```python
   shards_per_job = shards_per_gpu
   start = job_index * shards_per_job
   shard_subset = all_shards_tiled[start : start + shards_per_job]
   dataset = wds.WebDataset(shard_subset).split_by_worker()
   ```

   DDP (`shards_per_job = shards_per_gpu * n_gpus`):
   ```python
   shards_per_job = shards_per_gpu * n_gpus
   start = job_index * shards_per_job
   shard_subset = all_shards_tiled[start : start + shards_per_job]
   dataset = wds.WebDataset(shard_subset).split_by_node().split_by_worker()
   ```
   `split_by_node` divides `shard_subset` evenly across ranks; each rank then streams its `shards_per_gpu` shards via `split_by_worker`.

3. Track `job_index` in your checkpoint. The next job increments it and picks up from the next shard.

Each shard contains exactly `samples_per_shard` samples (except the last shard of the dataset), so the arithmetic is exact for all but the final shard. `samples_per_shard` is set at dataset creation time and stored in `info.json["images_per_shard"]` — currently 4096 for the imagenet-1k datasets, but arbitrary and should always be read from `info.json` rather than hardcoded.

**The last shard** contains fewer than `samples_per_shard` samples and breaks the clean-boundary arithmetic. Exclude it from the shard schedule — keep the file on disk but simply never include it in `all_shards_tiled`. The loss is at most `samples_per_shard - 1` samples, which is negligible for large datasets.

**Building `all_shards_tiled`:** precompute the full shard schedule upfront by tiling per-epoch permutations and save it to disk. Each job then slices by `job_index`:

```python
rng = np.random.default_rng(seed=0)
all_shards = sorted(shard_dir.glob("*.tar"))[:-1]  # exclude last shard
epochs = [rng.permutation(all_shards).tolist() for _ in range(1000)]  # 1000 epochs is enough for any practical training run
all_shards_tiled = [s for epoch in epochs for s in epoch]
np.save("shard_schedule.npy", all_shards_tiled)  # save once before job 0
```

Generate 1000 epochs upfront — the array is small (312,000 shard paths) and covers any practical training run. Jobs that finish early simply never reach the tail of the list.

## Multi-GPU (DDP)

This applies whenever more than one GPU is used, whether all GPUs are on a single node or spread across multiple nodes (e.g. 4 nodes × 8 GPUs = 32 ranks). DDP treats each GPU as one rank regardless of physical location, and the dataloader setup is identical in both cases. Note that despite its name, `split_by_node` splits by DDP rank (one per GPU process), not by physical machine.

`n_gpus` is the total number of GPU processes (ranks) across all nodes. It is determined by the SLURM script (`--nodes × --gpus-per-node`) and made available in Python as `world_size = int(os.environ["WORLD_SIZE"])` — see `plan_dataloading/ddp/minimal_distributed_training.sh` and `plan_dataloading/ddp/minimal_distributed_training.py` for a minimal reference implementation.

With `n_gpus` ranks, add `split_by_node` before `split_by_worker`:

```python
dataset = wds.WebDataset(shard_subset).split_by_node().split_by_worker()
```

`split_by_node` divides `shard_subset` evenly across ranks; `split_by_worker` subdivides each rank's shards across its local workers.

Additional constraints for DDP:

- `shards_per_job` is automatically divisible by `n_gpus` since it is defined as `shards_per_gpu * n_gpus`. The implementation must ensure `shards_per_gpu` is a whole number, which is guaranteed by the divisibility condition `steps_per_job * batch_size_per_gpu % samples_per_shard == 0`.

- `num_workers` per rank should be equal to `shards_per_gpu` (each worker gets exactly one shard). This is why there does not need to be a `num_workers` config field. The number of workers implied by the provided config should nevertheless be printed/logged so that it can be assessed whether a reasonable number of workers is used.

- All ranks must process the same number of batches per step to avoid hangs in DDP all-reduce.

## Validation loader

Both train and val loaders should use WebDataset. The val loader is simpler — no shard schedule, no `job_index` tracking, no shard exclusion needed. It always streams all val shards from start to finish within whichever job triggers validation:

```python
val_shards = sorted((val_dir / "*.tar").glob("*.tar"))  # include ALL shards, even the last partial one
dataset_val = wds.WebDataset(val_shards).split_by_node().split_by_worker().map(decode_sample).batched(batch_size)
```

Unlike training, the last shard of the val set should **not** be excluded — every val sample should be evaluated. Since validation is a single sequential pass with no checkpointing or boundary constraints, a partial final shard causes no issues.

## Example

- `batch_size_per_gpu = 256`, `steps_per_job = 48`, `n_gpus = 4`
- Samples per GPU: `steps_per_job * batch_size_per_gpu = 48 * 256 = 12288 = 3 * 4096 (samples_per_shard)` → `shards_per_gpu = 3`
- Total shards per job: `3 * 4 = 12`
- `num_workers` per rank: = 3
- Job 0 uses shards 0–11, job 1 uses shards 12–23, etc.
