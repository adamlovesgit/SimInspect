from ultralytics import YOLO


MODEL_PATH = "runs/detect/training/runs/siminspect_yolo_v1/weights/best.pt"
TEST_IMAGE = "datasets/siminspect_v1/images/test/frame_00003.jpg"


def main():
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=TEST_IMAGE,
        conf=0.25,
        save=True,
    )

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            print({
                "class_id": class_id,
                "confidence": round(confidence, 3),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            })


if __name__ == "__main__":
    main()