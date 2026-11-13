import argparse
from pathlib import Path
import time

import mujoco
import numpy as np


MODEL_PATH = Path(__file__).with_name("ur5e.xml")
OBJECTS = {
    "target_body": ("target", "site", [0.2, 0.6, 0.7]),
    "obs_1_body": ("obs_1", "geom", [0.1, 0.2, 0.3]),
    "obs_2_body": ("obs_2", "geom", [0.2, 0.2, 0.3]),
    "obs_3_body": ("obs_3", "geom", [0.3, 0.2, 0.3]),
}


def object_position(model, data, object_name, object_type):
    object_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE if object_type == "site" else mujoco.mjtObj.mjOBJ_GEOM,
        object_name,
    )
    return (
        data.site_xpos[object_id].copy()
        if object_type == "site"
        else data.geom_xpos[object_id].copy()
    )


def main(visualize=False, duration=3.0):
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    if model.nmocap != len(OBJECTS):
        raise AssertionError(f"Expected {len(OBJECTS)} mocap bodies, found {model.nmocap}")

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    viewer = None
    try:
        if visualize:
            from mujoco import viewer as mujoco_viewer

            viewer = mujoco_viewer.launch_passive(model, data)
            viewer.sync()
            time.sleep(1.0)

        for body_name, (object_name, object_type, new_position) in OBJECTS.items():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            mocap_id = model.body_mocapid[body_id]
            if mocap_id < 0:
                raise AssertionError(f"{body_name} is not attached to a mocap body")

            before = object_position(model, data, object_name, object_type)
            data.mocap_pos[mocap_id] = new_position
            mujoco.mj_forward(model, data)
            after = object_position(model, data, object_name, object_type)

            if not np.allclose(after, new_position):
                raise AssertionError(
                    f"{object_name} did not move to {new_position}; actual position: {after}"
                )
            if np.allclose(before, after):
                raise AssertionError(f"{object_name} did not move from its initial position")

            print(f"PASS {body_name}: {before} -> {after}")
            if viewer is not None:
                viewer.sync()
                time.sleep(0.5)

        if viewer is not None:
            print(f"Visualization complete; keeping the viewer open for {duration:.1f} seconds.")
            deadline = time.monotonic() + duration
            while time.monotonic() < deadline and viewer.is_running():
                viewer.sync()
                time.sleep(1 / 30)
    finally:
        if viewer is not None:
            viewer.close()

    print("Mocap object validation passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and visualize movable MuJoCo objects.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open a MuJoCo viewer and show each object being repositioned.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Seconds to keep the viewer open after validation (default: 3).",
    )
    args = parser.parse_args()
    if args.duration < 0:
        parser.error("--duration must be non-negative")
    main(visualize=args.visualize, duration=args.duration)
