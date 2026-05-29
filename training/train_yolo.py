from ultralytics import YOLO


def main():
    # Small pretrained model for a fast first experiment.
    # This will download the weights the first time if needed.
    model = YOLO("yolo11n.pt")

    model.train(
        data="datasets/siminspect_v1/data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        device="cpu",
        project="training/runs",
        name="siminspect_yolo_v1",
    )


if __name__ == "__main__":
    main()