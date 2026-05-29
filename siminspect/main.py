import time
import mujoco
import mujoco.viewer
import cv2
from pathlib import Path
from ultralytics import YOLO


XML = """
<mujoco model="first_inspection_sim">
    <option timestep="0.01" gravity="0 0 -9.81"/>

    <worldbody>
        <!-- Floor -->
        <geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>

        <!-- Conveyor placeholder -->
        <body name="conveyor" pos="0 0 0.05">
            <geom name="belt" type="box" size="1.5 0.35 0.05" rgba="0.05 0.05 0.05 1"/>
        </body>

        <body name="camera_mount" pos="0 0 2.0" euler="0 0 0">
    <geom name="camera_body"
          type="box"
          size="0.10 0.06 0.04"
          rgba="0.1 0.1 1 1"
          contype="0"
          conaffinity="0"/>

    <geom name="camera_lens"
          type="cylinder"
          pos="0 0 -0.07"
          size="0.025 0.02"
          euler="1.5708 0 0"
          rgba="0 0 0 1"
          contype="0"
          conaffinity="0"/>

    <camera name="inspection_camera"
            pos="0 0 -0.12"
            euler="0 0 0"
            fovy="45"/>
</body>

        <!-- Object being inspected -->
        <body name="red_apple" pos="-1.2 0 0.25">
            <freejoint name="red_object_joint"/>
            <geom name="red_object_geom" type="sphere" size=".10" mass="0.1" rgba="1 0 0 1"/>
        </body>

        <body name="green_box" pos="-2.2 0.3 0.25">
            <freejoint name="green_object_joint"/>
            <geom name="green_object_geom" type="box" size="0.08 0.08 0.08" mass="0.1" rgba="0 1 0 1"/>
        </body>

        <body name="blue_fish" pos="-3.2 0.3 0.25">
            <freejoint name="blue_object_joint"/>
            <geom name="blue_object_geom" type="ellipsoid" size="0.18 0.07 0.05" mass="0.1" rgba="0 0 1 1"/>
        </body>

    </worldbody>
</mujoco>
"""
PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_MODEL_PATH = Path(r"C:\Users\hadam\mujocoProj\models\siminspect_yolo.pt")

CONFIDENCE_THRESHOLD = 0.25

CLASS_NAMES = {
    0: "red_apple",
    1: "green_box",
    2: "blue_fish",
}

ROUTING_RULES = {
    "red_apple": "ACCEPT",
    "green_box": "REJECT",
    "blue_fish": "MANUAL_REVIEW",
}


def detect_objects_yolo(yolo_model, frame):
    """
    Runs YOLO on the full RGB MuJoCo camera frame.

    Input:
        yolo_model: loaded Ultralytics YOLO model
        frame: RGB image from MuJoCo renderer

    Output:
        list of prediction dictionaries
    """

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = yolo_model.predict(
        source=frame_bgr,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    predictions = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            label = CLASS_NAMES.get(class_id, f"unknown_{class_id}")
            decision = ROUTING_RULES.get(label, "MANUAL_REVIEW")

            predictions.append({
                "class_id": class_id,
                "label": label,
                "confidence": confidence,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "decision": decision,
            })

    return predictions
    
def route_prediction(prediction):
    label = prediction["label"]
    confidence = prediction["confidence"]

    if confidence < CONFIDENCE_THRESHOLD:
        return "MANUAL_REVIEW"

    return ROUTING_RULES.get(label, "MANUAL_REVIEW")
    
    
def get_camera_frame(renderer, data):
    renderer.update_scene(data, camera="inspection_camera")
    frame = renderer.render()
    return frame

def draw_yolo_overlay(frame, predictions):
    """
    Draws YOLO bounding boxes, labels, confidence, and routing decisions.
    """

    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    for prediction in predictions:
        x1, y1, x2, y2 = prediction["bbox"]

        label = prediction["label"]
        confidence = prediction["confidence"]
        decision = prediction["decision"]

        text = f"{label} {confidence:.2f} | {decision}"

        cv2.rectangle(
            display_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

    return display_frame

def main():
    
    
    mujoco_model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(mujoco_model)

    red_qpos_addr = mujoco_model.joint("red_object_joint").qposadr
    green_qpos_addr = mujoco_model.joint("green_object_joint").qposadr
    blue_qpos_addr = mujoco_model.joint("blue_object_joint").qposadr

    renderer = mujoco.Renderer(mujoco_model, height=480, width=640)

    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find YOLO model at: {YOLO_MODEL_PATH}")

    print(f"[model] Loading YOLO model from: {YOLO_MODEL_PATH}")
    yolo_model = YOLO(str(YOLO_MODEL_PATH))

    with mujoco.viewer.launch_passive(mujoco_model, data) as viewer:
        while viewer.is_running():
            # Move object manually across the conveyor
            data.qpos[red_qpos_addr] += 0.01
            data.qpos[green_qpos_addr] += 0.01
            data.qpos[blue_qpos_addr ] += 0.01
            mujoco.mj_forward(mujoco_model, data)

            # Capture virtual camera frame
            frame = get_camera_frame(renderer, data)

            # Run YOLO on the full camera frame
            predictions = detect_objects_yolo(yolo_model, frame)

            for prediction in predictions:
                print(
                    f"Detected: {prediction['label']} | "
                    f"confidence: {prediction['confidence']:.2f} | "
                    f"bbox: {prediction['bbox']} | "
                    f"decision: {prediction['decision']}"
                )

            # Draw YOLO overlay
            display_frame = draw_yolo_overlay(frame, predictions)

            cv2.imshow("Inspection Camera", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Reset object after it passes the conveyor
            red_x = data.body("red_apple").xpos[0]
            green_x = data.body("green_box").xpos[0]
            blue_x = data.body("blue_fish").xpos[0]

            if red_x > 1.2:
                data.qpos[red_qpos_addr + 0] = -1.2
                data.qpos[red_qpos_addr + 1] = 0.12
                data.qpos[red_qpos_addr + 2] = 0.25
                data.qpos[red_qpos_addr + 3] = 1
                data.qpos[red_qpos_addr + 4] = 0
                data.qpos[red_qpos_addr + 5] = 0
                data.qpos[red_qpos_addr + 6] = 0

            if green_x > 1.2:
                data.qpos[green_qpos_addr + 0] = -1.2
                data.qpos[green_qpos_addr + 1] = -0.12
                data.qpos[green_qpos_addr + 2] = 0.25
                data.qpos[green_qpos_addr + 3] = 1
                data.qpos[green_qpos_addr + 4] = 0
                data.qpos[green_qpos_addr + 5] = 0
                data.qpos[green_qpos_addr + 6] = 0

            if blue_x > 1.2:
                data.qpos[blue_qpos_addr + 0] = -1.2
                data.qpos[blue_qpos_addr + 1] = 0
                data.qpos[blue_qpos_addr + 2] = 0.25
                data.qpos[blue_qpos_addr + 3] = 1
                data.qpos[blue_qpos_addr + 4] = 0
                data.qpos[blue_qpos_addr + 5] = 0
                data.qpos[blue_qpos_addr + 6] = 0

            viewer.sync()
            time.sleep(0.01)

    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()