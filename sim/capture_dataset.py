from pathlib import Path
import random
import math

import cv2
import mujoco
import numpy as np


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = PROJECT_ROOT / "sim" / "scene.xml"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "siminspect_v2" / "raw_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Config
# -----------------------------
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
NUM_IMAGES = 200
CAMERA_NAME = "inspection_camera"

OBJECT_JOINT_NAMES = [
    "red_apple_joint",
    "green_box_joint",
    "blue_fish_joint",
]

SETTLE_STEPS = 10


# -----------------------------
# Helpers
# -----------------------------
def get_freejoint_qpos_adr(model, joint_name):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id == -1:
        raise ValueError(f"Could not find joint named '{joint_name}' in scene.xml")

    joint_type = model.jnt_type[joint_id]
    if joint_type != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"Joint '{joint_name}' is not a free joint")

    qpos_adr = model.jnt_qposadr[joint_id]
    qvel_adr = model.jnt_dofadr[joint_id]

    return qpos_adr, qvel_adr


def yaw_to_quat(yaw_radians):
    """
    Quaternion for rotation about z-axis only.
    Returns [qw, qx, qy, qz]
    """
    half = yaw_radians / 2.0
    return np.array([
        math.cos(half),  # qw
        0.0,             # qx
        0.0,             # qy
        math.sin(half)   # qz
    ], dtype=np.float64)


def sample_non_overlapping_positions(num_objects, x_range, y_range, min_dist):
    positions = []

    for _ in range(num_objects):
        for _ in range(100):
            x = random.uniform(*x_range)
            y = random.uniform(*y_range)

            too_close = False
            for px, py in positions:
                dist = math.sqrt((x - px) ** 2 + (y - py) ** 2)
                if dist < min_dist:
                    too_close = True
                    break

            if not too_close:
                positions.append((x, y))
                break
        else:
            raise RuntimeError("Could not sample non-overlapping object positions.")

    return positions


def randomize_objects_freejoint(model, data, joint_info):
    """
    joint_info = list of (joint_name, qpos_adr, qvel_adr)
    """

    positions = sample_non_overlapping_positions(
        num_objects=len(joint_info),
        x_range=(-0.7, 0.7),
        y_range=(-0.25, 0.25),
        min_dist=0.28
    )

    for (joint_name, qpos_adr, qvel_adr), (x, y) in zip(joint_info, positions):
        z = 0.35
        yaw = random.uniform(0, 2 * math.pi)
        quat = yaw_to_quat(yaw)

        # qpos for free joint = [x, y, z, qw, qx, qy, qz]
        data.qpos[qpos_adr : qpos_adr + 7] = np.array([
            x, y, z,
            quat[0], quat[1], quat[2], quat[3]
        ])

        # qvel for free joint = 6 values
        data.qvel[qvel_adr : qvel_adr + 6] = 0.0


def render_frame(model, data, renderer):
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=CAMERA_NAME)
    rgb = renderer.render()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


# -----------------------------
# Main
# -----------------------------
def main():
    print(f"Loading scene from: {SCENE_PATH}")

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    joint_info = []
    for joint_name in OBJECT_JOINT_NAMES:
        qpos_adr, qvel_adr = get_freejoint_qpos_adr(model, joint_name)
        joint_info.append((joint_name, qpos_adr, qvel_adr))

    renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

    print(f"Saving images to: {OUTPUT_DIR}")
    print(f"Capturing {NUM_IMAGES} images...")

    for i in range(NUM_IMAGES):
        mujoco.mj_resetData(model, data)

        randomize_objects_freejoint(model, data, joint_info)

        # Let objects settle slightly
        mujoco.mj_forward(model, data)
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(model, data)

        frame = render_frame(model, data, renderer)

        output_path = OUTPUT_DIR / f"frame_{i:05d}.jpg"
        cv2.imwrite(str(output_path), frame)

        if i % 50 == 0:
            print(f"Saved {i}/{NUM_IMAGES}")

    print("Capture complete.")


if __name__ == "__main__":
    main()