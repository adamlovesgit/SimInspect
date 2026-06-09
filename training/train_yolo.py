from ultralytics import YOLO


def main():
    # Small pretrained model for a fast first experiment.
    # This will download the weights the first time if needed.
    model = YOLO("yolo11n.pt")

    model.train(
        data="datasets/siminspect_v2/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        project="training/runs",
        name="siminspect_yolo_v2",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()