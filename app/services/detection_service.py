from pathlib import Path

import cv2
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)


class DetectionService:
    """Loads Faster R-CNN and performs object detection on video frames."""

    def __init__(self, confidence_threshold: float = 0.50):
        self.confidence_threshold = confidence_threshold
        self.target_classes = {"knife"}

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.categories = self.weights.meta["categories"]
        self.preprocess = self.weights.transforms()

        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=self.weights
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"Detection device: {self.device}")

        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    def detect_frame(self, frame):
        """Runs inference on one OpenCV frame."""

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        image_tensor = torch.from_numpy(rgb_frame)
        image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor.float() / 255.0

        image_tensor = self.preprocess(image_tensor)
        image_tensor = image_tensor.to(self.device)

        with torch.inference_mode():
            prediction = self.model([image_tensor])[0]

        boxes = prediction["boxes"].detach().cpu()
        labels = prediction["labels"].detach().cpu()
        scores = prediction["scores"].detach().cpu()

        detections = []

        for box, label, score in zip(boxes, labels, scores):
            confidence = float(score)

            if confidence < self.confidence_threshold:
                continue

            class_id = int(label)
            class_name = self.categories[class_id]

            if class_name not in self.target_classes: 
                continue

            x1, y1, x2, y2 = [
                int(value) for value in box.tolist()
            ]

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2],
                }
            )

        return detections


def test_first_frame(video_path: str):
    """Reads one video frame and tests Faster R-CNN inference."""

    video_file = Path(video_path)

    if not video_file.exists():
        raise FileNotFoundError(
            f"Video was not found: {video_file}"
        )

    capture = cv2.VideoCapture(str(video_file))

    if not capture.isOpened():
        raise RuntimeError(
            f"OpenCV could not open: {video_file}"
        )

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    middle_frame = total_frames // 2

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        middle_frame,
    )

    success, frame = capture.read()
    capture.release()

    print(f"Testing frame: {middle_frame}")

    if not success or frame is None:
        raise RuntimeError(
            "OpenCV could not read the selected video frame."
        )

    detector = DetectionService(
        confidence_threshold=0.50
    )

    detections = detector.detect_frame(frame)

    print(f"Frame dimensions: {frame.shape}")
    print(
        f"Detections above threshold: {len(detections)}"
    )

    for detection in detections:
        print(
            f"{detection['class_name']}: "
            f"{detection['confidence']:.2%}, "
            f"box={detection['box']}"
        )


if __name__ == "__main__":
    test_first_frame(
        "samples/test_10s_knife.mp4"
    )