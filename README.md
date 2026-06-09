# SimInspect: MuJoCo Conveyor Vision Inspection POC

SimInspect is a proof-of-concept computer vision simulation for a conveyor-based inspection system. The project uses MuJoCo to simulate a simple inspection environment with a virtual conveyor, camera, and moving objects. Frames from the simulated camera are processed with computer vision models to identify objects and make basic inspection decisions.

The original version of this project used an OpenCV color-thresholding pipeline to detect and classify objects based on HSV masks. The current version adds a machine learning-based inspection pipeline using a custom-trained YOLO object detection model.

## Project Video:
 https://youtu.be/-mSG7MIhLtg (ML Version V2)

https://youtu.be/2-qUo2HlPfA (ML Version)

 https://youtu.be/IaN0UBeMf0Y (OpenCV Version)


## Project Status Model V2 Update

This repo has been updated with an improved ML-based object detection version of the inspection system. This new dataset combines the original model's dataset as well as a new dataset. The YOLO model was trained on a new custom MuJoCo-generated dataset containing three simulated object classes:

- `red_apple`
- `green_box`
- `blue_fish`

| Version | Main Change | Result |
|---|---|---|
| v1 | Initial YOLO detector trained on first labeled MuJoCo dataset | Strong green_box and red_apple detection, inconsistent blue_fish detection |
| v2 | Improved annotations and dataset quality, especially for blue_fish | ~90% average confidence across all three object classes |

The dataset was manually annotated using MakeSense.ai and exported in YOLO format. The trained model is integrated into the MuJoCo simulation loop to detect objects from the virtual inspection camera and display bounding boxes, class labels, confidence scores, and routing decisions in real time.

## Current Features

- MuJoCo-based conveyor inspection simulation
- Virtual inspection camera for frame capture
- Custom YOLO object detection model
- Three object classes: red apple, green box, and blue fish
- Real-time bounding-box overlays with confidence scores
- Basic routing logic based on detected class
- OpenCV-based display and visualization
- Dataset organization scripts for YOLO training
- Dataset validation script for checking image-label consistency

## ML Pipeline

The ML version follows this workflow:

1. Generate synthetic images from the MuJoCo scene.
2. Split images into train, validation, and test folders.
3. Manually annotate objects using MakeSense.ai.
4. Export annotations in YOLO format.
5. Train a custom YOLO model on the labeled dataset.
6. Load the trained model into the main simulation loop.
7. Run real-time object detection on frames from the MuJoCo inspection camera.

## Notes

This is an early proof-of-concept. The first trained model uses a small manually labeled dataset, so detection quality may vary depending on camera angle, object spacing, and lighting. The next improvement is to expand the dataset with more runtime-like examples, including objects entering the frame, single-object scenes, varied object positions, and additional blue fish examples to improve weak-class performance.
