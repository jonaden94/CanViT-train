import json, sys, torch
WHICH = sys.argv[1]
torch.manual_seed(1234)
if WHICH == "old":
    from canvit_pretrain import CanViTForPretraining, CanViTForPretrainingConfig
    from canvit_pytorch import create_backbone
    bb = create_backbone("vits16")
    m = CanViTForPretraining(backbone=bb, cfg=CanViTForPretrainingConfig(teacher_dim=384),
                             backbone_name="vits16", canvas_patch_grid_sizes=[8])
else:
    from canvit import CanViTForPretraining, CanViTForPretrainingConfig
    from canvit.core import create_backbone
    bb = create_backbone("vits16")
    m = CanViTForPretraining(backbone=bb, cfg=CanViTForPretrainingConfig(teacher_dim=384),
                             glimpse_size_px=128, backbone_name="vits16",
                             canvas_patch_grid_sizes=[8])
sd = m.state_dict()
out = {k: list(v.shape) for k, v in sd.items()}
chk = {k: float(v.detach().double().sum()) for k, v in sd.items()}
json.dump({"shapes": out, "sums": chk, "n_params": sum(p.numel() for p in m.parameters())},
          open(sys.argv[2], "w"), indent=1)
print(WHICH, "params:", sum(p.numel() for p in m.parameters()), "tensors:", len(sd))
