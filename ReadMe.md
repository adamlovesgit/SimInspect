# SimInspect: MuJoCo Conveyor Vision Inspection POC

SimInspect is a proof-of-concept computer vision simulation for a conveyor-based inspection system. The project uses MuJoCo to simulate a simple conveyor environment and OpenCV to process frames from a virtual inspection camera.

The current version uses a rule-based OpenCV pipeline to detect and classify colored objects as they pass through an inspection region. This serves as a baseline for future machine learning-based object recognition.

## Project Status

This repository currently represents an early proof of concept.

The goal of this version is not to build a fully trained ML system yet. Instead, the goal is to prove the full perception loop:

```text
MuJoCo simulation
→ virtual RGB camera
→ OpenCV frame processing
→ ROI extraction
→ contour-based object detection
→ object crop extraction
→ baseline classification
→ accept/reject routing
→ visual overlay
