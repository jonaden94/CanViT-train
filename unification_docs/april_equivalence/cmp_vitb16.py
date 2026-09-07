import json, sys, torch
WHICH=sys.argv[1]
torch.manual_seed(1234)
if WHICH=="old":
    from canvit_pretrain import CanViTForPretraining, CanViTForPretrainingConfig
    from canvit_pytorch import create_backbone
    bb=create_backbone("vitb16")
    m=CanViTForPretraining(backbone=bb, cfg=CanViTForPretrainingConfig(teacher_dim=768),
                           backbone_name="vitb16", canvas_patch_grid_sizes=[32])
else:
    from canvit import CanViTForPretraining, CanViTForPretrainingConfig
    from canvit.core import create_backbone
    bb=create_backbone("vitb16")
    m=CanViTForPretraining(backbone=bb, cfg=CanViTForPretrainingConfig(teacher_dim=768),
                           glimpse_size_px=128, backbone_name="vitb16", canvas_patch_grid_sizes=[32])
sd=m.state_dict()
json.dump({"shapes":{k:list(v.shape) for k,v in sd.items()},
           "sums":{k:float(v.detach().double().sum()) for k,v in sd.items()},
           "n":sum(p.numel() for p in m.parameters())}, open(sys.argv[2],"w"))
print(WHICH,"vitb16 params:",sum(p.numel() for p in m.parameters()),"tensors:",len(sd))
