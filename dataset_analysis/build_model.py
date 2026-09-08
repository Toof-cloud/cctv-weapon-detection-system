import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


def get_model(num_classes=3, small_anchors=False, anchor_scales=None):
    model = (
        torchvision.models.detection
        .fasterrcnn_resnet50_fpn_v2(
            weights="DEFAULT"
        )
    )

    in_features = (
        model.roi_heads
        .box_predictor
        .cls_score
        .in_features
    )

    model.roi_heads.box_predictor = (
        FastRCNNPredictor(
            in_features,
            num_classes,
        )
    )

    if anchor_scales is not None:
        anchor_sizes = tuple((int(s),) for s in anchor_scales)
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        model.rpn.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
        )
    elif small_anchors:
        anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * 5
        model.rpn.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
        )

    return model



if __name__ == "__main__":
    model = get_model()

    print(model.roi_heads.box_predictor)