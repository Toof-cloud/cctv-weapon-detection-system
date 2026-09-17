import cv2

from app.services.detection_service import (
    DetectionService,
    draw_detections,
)

IMAGE_PATH = "samples/test.jpg"

frame = cv2.imread(IMAGE_PATH)

if frame is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

detector = DetectionService(
    confidence_threshold=0.70
)

detections = detector.detect_frame(frame)

print("\nDETECTIONS")
print("----------")

for detection in detections:
    print(
        f"{detection['class_name']} | "
        f"{detection['confidence']:.2%} | "
        f"{detection['box']}"
    )

frame = draw_detections(
    frame,
    detections,
)

cv2.imshow(
    "Weapon Detection Result",
    frame,
)

cv2.waitKey(0)
cv2.destroyAllWindows()