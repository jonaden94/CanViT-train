"""CanViT: Dual-stream vision transformer with canvas cross-attention."""

from canvit.core.backbone import BackboneName, ViTBackbone, create_backbone
from canvit.core.checkpoints import CANVIT_REPO_ROOT, resolve_canvit_repo
from canvit.core.model import (
    CanViT,
    CanViTConfig,
    CanViTForImageClassification,
    CanViTForPretraining,
    CanViTForPretrainingConfig,
    CanViTForPretrainingHFHub,
    CanViTForSemanticSegmentation,
    CanViTOutput,
    RecurrentState,
    fuse_probe,
)
from canvit.core.patcher import (
    FoveatedPatcherConfig,
    Patcher,
    PatcherName,
    SquarePatcherConfig,
    UniformPatcher,
    create_patcher,
)
from canvit.core.probes import SegmentationProbe
from canvit.core.standardizers import CLSStandardizer, PatchStandardizer, PositionAwareStandardizer
from canvit.core.viewpoint import Viewpoint, sample_at_viewpoint
from canvit.core.vpe import VPEEncoder

__all__ = [
    "BackboneName",
    "CLSStandardizer",
    "CanViT",
    "CanViTConfig",
    "CanViTForImageClassification",
    "CanViTForPretraining",
    "CanViTForPretrainingConfig",
    "CanViTForPretrainingHFHub",
    "CanViTForSemanticSegmentation",
    "CANVIT_REPO_ROOT",
    "CanViTOutput",
    "FoveatedPatcherConfig",
    "Patcher",
    "PatcherName",
    "PatchStandardizer",
    "PositionAwareStandardizer",
    "RecurrentState",
    "SegmentationProbe",
    "SquarePatcherConfig",
    "UniformPatcher",
    "VPEEncoder",
    "Viewpoint",
    "ViTBackbone",
    "create_backbone",
    "create_patcher",
    "fuse_probe",
    "resolve_canvit_repo",
    "sample_at_viewpoint",
]
