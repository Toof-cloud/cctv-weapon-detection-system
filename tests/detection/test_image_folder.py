from pathlib import Path
import cv2

from app.services.detection_service import (
    DetectionService,
)

detector = DetectionService(
    confidence_threshold=0.70
)

FOLDERS = [
    "samples/handgun_test_images",
    "samples/knives_test_images",
    "samples/negative_test_images",
]

for folder in FOLDERS:

    print("\n" + "=" * 60)
    print(folder)
    print("=" * 60)

    for image_path in sorted(Path(folder).glob("*.jpg")):

        frame = cv2.imread(str(image_path))

        detections = detector.detect_frame(frame)

        print(f"\n{image_path.name}")

        if not detections:
            print("  No detections")
            continue

        for det in detections:
            print(
                f"  {det['class_name']} | "
                f"{det['confidence']:.2%} | "
                f"{det['box']}"
            )