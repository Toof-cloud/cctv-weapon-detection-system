"""
Isolated Model Builder for Research on Weapon Detection Improvements.
Supports customized multi-scale anchor pyramids, extreme aspect ratios for bladed weapons,
and channel-aligned RPN heads.
"""
import math
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead


def build_research_model(
    num_classes: int = 3,
    anchor_sizes: tuple = ((16,), (32,), (64,), (128,), (192,)),
    aspect_ratios: tuple = (0.5, 0.75, 1.0, 1.5, 2.0),
    pretrained_backbone: bool = True,
    freeze_early_backbone: bool = False,
    bbox_xform_clip: float = None,
    rpn_bbox_xform_clip: float = math.log(1.4),   # ~0.3365 (1.4x coarse proposal expansion)
    roi_bbox_xform_clip: float = math.log(1.3),   # ~0.2624 (1.3x fine refinement)
):
    """
    Constructs a Faster R-CNN ResNet-50 FPN V2 with Candidate V8 calibrated anchor geometry:
    - Multi-scale anchor pyramid ((16,), (32,), (64,), (128,), (192,)) covers:
        P2 (16px): Distant CCTV weapons (15-30px)
        P3 (32px): Lower-medium range (30-65px)
        P4 (64px): Mid-medium range near median (65-120px)
        P5 (128px): Close-range handguns (120-180px, e.g. robbery counter)
        P6 (192px): Close-range knives & handguns (180-250px)
    - 5 calibrated aspect ratios (0.5, 0.75, 1.0, 1.5, 2.0) cover natural handheld weapon shapes
      without degenerate vertical (0.4) or horizontal (2.5) anchors that match architectural fixtures.
    - Stage-decoupled box regression clamping:
        RPN clamp = ln(1.4) (1.4x)
        RoI clamp = ln(1.3) (1.3x)
        Compound theoretical limit: 1.4 * 1.3 = 1.82x
        Maximum reachable dimension: 271.5px * 1.82 = 494.1px (105px safety gap below 600px counter/door danger zone).
    """
    weights = "DEFAULT" if pretrained_backbone else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)

    # 1. Custom Region Proposal Network (RPN) Anchor Generator
    ratios_tuple = (aspect_ratios,) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=ratios_tuple)
    model.rpn.anchor_generator = anchor_generator

    # 2. Re-align RPN Head to match the new number of anchors per spatial location
    out_channels = model.backbone.out_channels
    num_anchors = anchor_generator.num_anchors_per_location()[0]
    model.rpn.head = RPNHead(out_channels, num_anchors)

    # 3. Class score predictor (Background, Handgun, Knife)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 4. Stage-decoupled box regression delta clamping
    clip_rpn = rpn_bbox_xform_clip if rpn_bbox_xform_clip is not None else bbox_xform_clip
    clip_roi = roi_bbox_xform_clip if roi_bbox_xform_clip is not None else bbox_xform_clip

    if clip_rpn is not None:
        model.rpn.box_coder.bbox_xform_clip = clip_rpn
    if clip_roi is not None:
        model.roi_heads.box_coder.bbox_xform_clip = clip_roi

    # 5. Optional early backbone freezing to preserve low-level edge kernels
    if freeze_early_backbone and hasattr(model.backbone, "body"):
        body = model.backbone.body
        for param in [body.conv1.parameters(), body.bn1.parameters(), body.layer1.parameters()]:
            for p in param:
                p.requires_grad = False

    return model


if __name__ == "__main__":
    m = build_research_model()
    dummy = [torch.rand(3, 640, 640)]
    m.eval()
    with torch.no_grad():
        out = m(dummy)
    print("Research model initialized successfully! Dummy pass output keys:", out[0].keys())
