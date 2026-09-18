import cv2
import numpy as np
import sys
from yolox import YoloX
from sort_tracker import Sort


# -----------------------------
# YOLOX preprocessing
# -----------------------------
def letterbox(image, target_size=(640, 640)):
    padded = np.ones(
        (target_size[0], target_size[1], 3),
        dtype=np.float32
    ) * 114.0

    ratio = min(
        target_size[0] / image.shape[0],
        target_size[1] / image.shape[1]
    )

    new_width = int(image.shape[1] * ratio)
    new_height = int(image.shape[0] * ratio)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)

    padded[:new_height, :new_width] = resized

    return padded, ratio


# -----------------------------
# Load YOLOX
# -----------------------------
model = YoloX(
    "models/object_detection_yolox_2022nov.onnx",
    confThreshold=0.4,
    nmsThreshold=0.5,
    objThreshold=0.4
)


# -----------------------------
# Load COCO labels
# -----------------------------
with open("coco.names", "r") as file:
    class_names = [line.strip() for line in file.readlines()]


# -----------------------------
# SORT tracker
# -----------------------------
tracker = Sort(
    max_age=10,
    min_hits=1,
    iou_threshold=0.3
)


# -----------------------------
# Open webcam
# -----------------------------
video_source = sys.argv[1] if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started.")
print("Press Q to quit.")


# -----------------------------
# Main loop
# -----------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert BGR -> RGB
    input_blob = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Resize and pad
    input_blob, scale = letterbox(input_blob)

    # YOLOX detection
    detections = model.infer(input_blob)

    boxes = []
    detection_info = []

    # Convert YOLOX output to XYXY
    for detection in detections:

        x, y, width, height, confidence, class_id = detection

        # Convert back to original frame size
        x = x / scale
        y = y / scale
        width = width / scale
        height = height / scale

        x1 = x
        y1 = y
        x2 = x + width
        y2 = y + height

        boxes.append([x1, y1, x2, y2])

        detection_info.append({
            "box": [x1, y1, x2, y2],
            "class_id": int(class_id),
            "confidence": float(confidence)
        })

    # SORT tracking
    tracks = tracker.update(boxes)

    # Draw tracked objects
    for track in tracks:

        x1, y1, x2, y2, track_id = track

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        track_id = int(track_id)

        best_label = "Object"
        best_confidence = 0
        best_iou = 0

        # Match tracker with detection
        for detection in detection_info:

            dx1, dy1, dx2, dy2 = detection["box"]

            ix1 = max(x1, int(dx1))
            iy1 = max(y1, int(dy1))
            ix2 = min(x2, int(dx2))
            iy2 = min(y2, int(dy2))

            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

            track_area = max(0, x2 - x1) * max(0, y2 - y1)
            detection_area = max(0, dx2 - dx1) * max(0, dy2 - dy1)

            union = track_area + detection_area - intersection

            current_iou = (
                intersection / union
                if union > 0
                else 0
            )

            if current_iou > best_iou:

                best_iou = current_iou
                best_confidence = detection["confidence"]

                class_id = detection["class_id"]

                if 0 <= class_id < len(class_names):
                    best_label = class_names[class_id]

        if best_iou < 0.1:
            continue

        label = f"{best_label} ID:{track_id} {best_confidence:.2f}"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    # Display FPS information
    cv2.putText(
        frame,
        "AI Object Detection & Tracking",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "AI Object Detection & Tracking",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("Program stopped.")