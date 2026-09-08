# Dataset Structure

## Paths

- **Train**: `/user/henrich1/u25995/jonathan/datasets/webdataset-imagenet-1k/train-shuffled`
- **Val**: `/user/henrich1/u25995/jonathan/datasets/webdataset-imagenet-1k/val`

Train has been pre-shuffled globally — samples are randomly distributed across shards so that any contiguous subset of shards contains a diverse mix of classes. Val is unshuffled; samples are in their original dataset order, which is fine since validation iterates over all samples exactly once.

## Directory layout

```
train-shuffled/
    info.json
    shard-000000.tar
    shard-000001.tar
    ...
    shard-000312.tar   ← last shard, has fewer than 4096 samples, exclude from training

val/
    info.json
    checkpoint.json    ← creation artifact, can be ignored
    shard-000000.tar
    ...
    shard-000012.tar   ← last shard, has fewer than 4096 samples, but do not exclude since it is for validation only
```

## info.json

Always read dataset metadata from `info.json` rather than hardcoding values. Key fields:

| Field | Train | Val |
|---|---|---|
| `n_images` | 1281167 | 50000 |
| `n_shards` | 313 | 13 |
| `images_per_shard` | 4096 | 4096 |
| `keys` | `["jpg", "json", "cls.npy", "ptch.npy"]` | same |
| `resize` / `crop` | 512 / 512 | 512 / 512 |
| `jpeg_quality` | 95 | 95 |

Whether features are present can be determined as:
```python
has_features = "cls.npy" in info["keys"]
```

If `has_features` is `False`, `cls.npy` and `ptch.npy` are absent from the tar entries and must not be loaded. In this case the teacher model should be loaded at the start of training and features computed on the fly within the training loop for each batch.

## Tar entry format

Each tar is a flat sequence of entries. Entries for one sample are contiguous and share a zero-padded 9-digit key (the global sample index):

```
000000484.cls.npy
000000484.jpg
000000484.json
000000484.ptch.npy
000001157.cls.npy
000001157.jpg
...
```

Note: entry order within a sample may vary — do not assume a fixed order of extensions.

## Per-sample contents

| File | Description | Shape / type |
|---|---|---|
| `<key>.jpg` | JPEG image, resized and center-cropped to 512×512 | bytes |
| `<key>.json` | Label metadata | `{"label": 502}` (integer class index 0–999) |
| `<key>.cls.npy` | DINOv3 ViT-B/16 CLS token | `float16`, shape `(768,)` |
| `<key>.ptch.npy` | DINOv3 ViT-B/16 patch tokens | `float16`, shape `(1024, 768)` — 1024 patches of hidden dim 768 |

Features were extracted with teacher model `facebook/dinov3-vitb16-pretrain-lvd1689m` using 5 prefix tokens and patch size 16. Normalization: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.

## Loading with WebDataset

For the full job-array-aware loading design (shard schedule, `job_index` tracking, DDP setup), see `plan_dataloading/dataloading.md`. The snippet below shows how to decode individual samples.

```python
import webdataset as wds
import numpy as np, json, io
from PIL import Image

def decode_sample(sample):
    img = Image.open(io.BytesIO(sample["jpg"])).convert("RGB")
    label = json.loads(sample["json"])["label"]
    cls = np.load(io.BytesIO(sample["cls.npy"]))    # shape (768,), float16
    ptch = np.load(io.BytesIO(sample["ptch.npy"]))  # shape (1024, 768), float16
    return img, label, cls, ptch

dataset = (
    wds.WebDataset(shard_paths)
    .split_by_node()   # one subset per DDP rank
    .split_by_worker() # one subset per DataLoader worker
    .map(decode_sample)
    .batched(batch_size)
)
```
