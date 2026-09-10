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
    anchor_sizes: tuple = ((12,), (20,), (36,), (64,), (96,)),
    aspect_ratios: tuple = (0.5, 0.7, 1.0, 1.4, 2.0),
    pretrained_backbone: bool = True,
    freeze_early_backbone: bool = False,
    bbox_xform_clip: float = math.log(2.5),
):
    """
    Constructs a Faster R-CNN ResNet-50 FPN V2 with Candidate V3/V5 calibrated anchor geometry:
    - Bounded aspect ratios (0.5, 0.7, 1.0, 1.4, 2.0) provide realistic knife/gun shapes
      while mathematically preventing 500+ pixel furniture baseboard traps.
    - Capped scales ((12,), (20,), (36,), (64,), (96,)) strictly bounded by human hand/reach bounds.
      Maximum anchor proposal size is 96 * sqrt(2) = 135px.
    - bbox_xform_clip (default math.log(2.5) = 0.916) mathematically clamps maximum box expansion
      to 2.5x anchor dimension (max box <= 240px). 1000+ px counter proposals are rendered impossible.
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

    # 4. Box regression delta clamping (mathematically limits proposal expansion to 2.5x anchor size)
    if bbox_xform_clip is not None:
        model.rpn.box_coder.bbox_xform_clip = bbox_xform_clip
        model.roi_heads.box_coder.bbox_xform_clip = bbox_xform_clip

    # 5. Optional early backbone freezing to preserve low-level handgun edge kernels
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
