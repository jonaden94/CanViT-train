# Plan: WebDataset Dataloading

## Goal

Implement a WebDataset-based dataloader for this PyTorch training repo which should be run on an HPC cluster under SLURM job arrays. Training is split across multiple short jobs; each job processes a fixed slice of shards, checkpoints, and terminates. The next job resumes from exactly where the previous one stopped.

## Files

| File | Contents |
|---|---|
| `plan_dataloading/dataset_structure.md` | On-disk layout of the train and val datasets: paths, shard count, per-sample contents (`jpg`, `json`, `cls.npy`, `ptch.npy`), and a decode snippet. **Start here.** |
| `plan_dataloading/dataloading.md` | Full loading design: how shards are sliced per job, the precomputed shard schedule (`all_shards_tiled`), `job_index` tracking, DDP setup with `split_by_node` / `split_by_worker`, val loader, and a worked example. **Core reference.** |
| `plan_dataloading/ddp/minimal_distributed_training.sh` | Minimal SLURM script showing how `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT` are set for multi-GPU / multi-node DDP on this cluster. |
| `plan_dataloading/ddp/minimal_distributed_training.py` | Minimal Python DDP reference: rank derivation from `SLURM_PROCID`, `local_rank` computation, and `init_process_group`. Uses `DistributedSampler` as a placeholder — replace with `split_by_node` / `split_by_worker` per `plan_dataloading/dataloading.md`. |
| `plan_dataloading/things_not_yet_fully_specified.md` | Open design questions not resolved by the above files: which sample fields the training loop actually needs, where to save the shard schedule, how to integrate `job_index` into the existing checkpoint format, and feature dtype handling. **Read last.** |

## Suggested reading order

1. `plan_dataloading/dataset_structure.md` — understand what data is available and how it is structured
2. `plan_dataloading/dataloading.md` — understand the full loading strategy
3. `plan_dataloading/ddp/` — understand how DDP is initialised on this cluster
4. `plan_dataloading/things_not_yet_fully_specified.md` — note the open questions to resolve in the target repo
