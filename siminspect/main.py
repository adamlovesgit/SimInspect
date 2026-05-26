import time
from xml.parsers.expat import model
import mujoco
import mujoco.viewer
import cv2



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

        <!-- Visual inspection zone -->
        <body name="inspection_zone" pos="0 0 0.12">
            <geom name="zone" type="box" size="0.08 0.38 0.005" rgba="0.1 0.1 1 1"/>
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
        <body name="red_object" pos="-1.2 0 0.25">
            <freejoint name="red_object_joint"/>
            <geom name="red_object_geom" type="box" size="0.08 0.08 0.08" mass="0.1" rgba="1 0 0 1"/>
        </body>

        <body name="green_object" pos="-2.2 0.3 0.25">
            <freejoint name="green_object_joint"/>
            <geom name="green_object_geom" type="box" size="0.08 0.08 0.08" mass="0.1" rgba="0 1 0 1"/>
        </body>

    </worldbody>
</mujoco>
"""
class DetectedObject:
    def __init__(self, bbox, crop, area):
        self.bbox = bbox          # (x, y, w, h)
        self.crop = crop          # cropped RGB image of the object
        self.area = area          # contour area
        
def detect_objects(roi):
    """
    Detects colored objects inside the inspection ROI.

    Input:
        roi: RGB crop from the camera frame

    Output:
        list of DetectedObject instances
    """

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    # Red wraps around HSV, so use two masks
    red_mask_1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red_mask_2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    red_mask = red_mask_1 | red_mask_2

    # Green mask
    green_mask = cv2.inRange(hsv, (40, 80, 80), (90, 255, 255))

    # Combine all object-color masks
    combined_mask = red_mask | green_mask

    # Clean small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        combined_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 300:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        crop = roi[y:y+h, x:x+w]

        detections.append(
            DetectedObject(
                bbox=(x, y, w, h),
                crop=crop,
                area=area
            )
        )

    return detections

def process_frame(frame):
    """
    Prepares the raw MuJoCo camera frame for object detection.

    Input:
        frame: RGB image from MuJoCo renderer

    Output:
        roi: cropped inspection-zone image
        roi_box: coordinates of ROI in original frame: (x1, y1, x2, y2)
    """

    height, width, _ = frame.shape

    # Tune these values based on your camera view
    x1 = int(width * 0.35)
    x2 = int(width * 0.65)

    y1 = int(height * 0.30)
    y2 = int(height * 0.70)

    roi = frame[y1:y2, x1:x2]

    return roi, (x1, y1, x2, y2)

def classify_object(detection):
    """
    Classifies a detected object crop.

    Input:
        detection: DetectedObject containing crop and bbox

    Output:
        dictionary with label and confidence
    """

    crop = detection.crop

    if crop is None or crop.size == 0:
        return {
            "label": "UNKNOWN",
            "confidence": 0.0
        }

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    red_mask_1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    red_mask_2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
    red_mask = red_mask_1 | red_mask_2

    green_mask = cv2.inRange(hsv, (40, 80, 80), (90, 255, 255))

    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)

    total_pixels = crop.shape[0] * crop.shape[1]

    if total_pixels == 0:
        return {
            "label": "UNKNOWN",
            "confidence": 0.0
        }

    if red_pixels > green_pixels and red_pixels > 100:
        return {
            "label": "RED_BLOCK",
            "confidence": red_pixels / total_pixels
        }

    if green_pixels > red_pixels and green_pixels > 100:
        return {
            "label": "GREEN_BLOCK",
            "confidence": green_pixels / total_pixels
        }

    return {
        "label": "UNKNOWN",
        "confidence": 0.0
    }

    
def route_prediction(prediction):
    label = prediction["label"]
    confidence = prediction["confidence"]

    if confidence < 0.10:
        return "MANUAL_REVIEW"

    if label == "GREEN_BLOCK":
        return "ACCEPT"

    if label == "RED_BLOCK":
        return "REJECT"

    return "MANUAL_REVIEW"
    
    
def get_camera_frame(renderer, data):
    renderer.update_scene(data, camera="inspection_camera")
    frame = renderer.render()
    return frame

def draw_pipeline_overlay(frame, roi_box, detections, predictions):
    """
    Draws ROI, bounding boxes, labels, confidence, and decisions.
    """

    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    x1, y1, x2, y2 = roi_box

    # Draw inspection ROI
    cv2.rectangle(
        display_frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        "INSPECTION ROI",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    for detection, prediction in zip(detections, predictions):
        x, y, w, h = detection.bbox

        # Convert ROI-local bbox to full-frame bbox
        full_x = x1 + x
        full_y = y1 + y

        label = prediction["label"]
        confidence = prediction["confidence"]
        decision = route_prediction(prediction)

        cv2.rectangle(
            display_frame,
            (full_x, full_y),
            (full_x + w, full_y + h),
            (0, 255, 255),
            2
        )

        text = f"{label} {confidence:.2f} | {decision}"

        cv2.putText(
            display_frame,
            text,
            (full_x, full_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

    return display_frame

def main():
    
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    red_qpos_addr = model.joint("red_object_joint").qposadr
    green_qpos_addr = model.joint("green_object_joint").qposadr
    renderer = mujoco.Renderer(model, height=480, width=640)

    already_inspected = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Move object manually across the conveyor
            data.qpos[red_qpos_addr] += 0.005
            data.qpos[green_qpos_addr] += 0.005
            mujoco.mj_forward(model, data)

            # Capture virtual camera frame
            frame = get_camera_frame(renderer, data)

            # 1. Process frame: crop inspection ROI
            roi, roi_box = process_frame(frame)

            # 2. Detect objects inside ROI
            detections = detect_objects(roi)

            # 3. Classify each detected object
            predictions = []

            for detection in detections:
                prediction = classify_object(detection)
                predictions.append(prediction)

                decision = route_prediction(prediction)

                print(
                    f"Detected: {prediction['label']} | "
                    f"confidence: {prediction['confidence']:.2f} | "
                    f"decision: {decision}"
                )

            # 4. Draw overlay
            display_frame = draw_pipeline_overlay(
                frame=frame,
                roi_box=roi_box,
                detections=detections,
                predictions=predictions
            )

            cv2.imshow("Inspection Camera", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Reset object after it passes the conveyor
            red_x = data.body("red_object").xpos[0]
            green_x = data.body("green_object").xpos[0]

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

            viewer.sync()
            time.sleep(0.01)

    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()