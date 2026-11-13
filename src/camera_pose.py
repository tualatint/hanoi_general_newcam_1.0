"""Apply a saved free-camera pose to the policy camera (cam_pc)."""
from __future__ import annotations

import json

import mujoco
import numpy as np


def apply_camera_pose(env, path: str, camera_name: str = "cam_pc") -> dict:
    pose = json.load(open(path))
    cid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA,
                            camera_name)
    if cid < 0:
        raise ValueError(f"camera {camera_name!r} not found")
    env.model.cam_pos[cid] = np.asarray(pose["pos"], dtype=np.float64)
    env.model.cam_quat[cid] = np.asarray(
        pose["quat_wxyz"], dtype=np.float64)
    try:
        env.model.cam_fovy[cid] = float(env.model.vis.global_.fovy)
    except Exception:
        pass
    mujoco.mj_forward(env.model, env.data)
    return pose
