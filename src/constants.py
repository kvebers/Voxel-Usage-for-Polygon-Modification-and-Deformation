from __future__ import annotations
import numpy as np
from typing import Any

DRIFT_DAMP_COEFF: float = 5.0
DRIFT_DAMP_THRESHOLD: float = 0.3
DEFAULT_BASE_ROT: np.ndarray = np.array(
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32
)
DEFAULT_BASE_OFFSET: np.ndarray = np.array([0.0, 0.0, 1.5], dtype=np.float32)
LIGHT_DIR: np.ndarray = np.array([-1.0, -0.5, 1.0], dtype=np.float32)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)
FACE_DEFS: list = [
    ((-1, 0, 0), lambda b: (b[0], b[0], b[1], b[4], b[2], b[5])),
    ((+1, 0, 0), lambda b: (b[3], b[3], b[1], b[4], b[2], b[5])),
    ((0, -1, 0), lambda b: (b[0], b[3], b[1], b[1], b[2], b[5])),
    ((0, +1, 0), lambda b: (b[0], b[3], b[4], b[4], b[2], b[5])),
    ((0, 0, -1), lambda b: (b[0], b[3], b[1], b[4], b[2], b[2])),
    ((0, 0, +1), lambda b: (b[0], b[3], b[1], b[4], b[5], b[5])),
]

DEFAULTS: dict[str, Any] = {
    "mesh": {
        "path": "obj/teapot.obj",
        "resolution": 16,
    },
    "camera": {
        "distance": 8.0,
        "yaw": 135.0,
        "pitch": 25.0,
        "target": [0.0, 0.0, 1.5],
        "fov": 50,
        "near": 0.1,
        "far": 200.0,
        "position": None,
    },
    "window": {
        "width": 1280,
        "height": 720,
        "title": "Voxel Destruction — AVBD + GLSL Shaders",
        "clear_color": [0.08, 0.09, 0.12, 1.0],
    },
    "simulation": {
        "fps": 60,
        "substeps": 4,
        "start_paused": True,
    },
    "ground": {
        "position": [0.0, 0.0, -0.5],
        "half_extents": [50.0, 50.0, 0.5],
        "density": 0.0,
        "ke": 1e6,
        "kd": 1e3,
        "kf": 1e3,
        "mu": 0.5,
    },
    "voxels": {
        "density": 500.0,
        "padding": 0.01,
        "ensure_connected": False,
        "greedy_merge": False,
    },
    "ball": {
        "enabled": False,
        "count": 1,
        "radius_factor": 0.2,
        "height_factor": 2.0,
        "density": 20000.0,
        "ke": 1e6,
        "kd": 1e3,
        "kf": 1e3,
        "mu": 0.5,
        "color": [0.3, 0.35, 0.85],
    },
    "solver": {
        "iterations": 10,
        "rigid_body_contact_buffer_size": 4096,
        "rigid_contact_k_start": 1e5,
        "rigid_avbd_beta": 1e4,
        "rigid_joint_linear_ke": 1e6,
        "rigid_joint_angular_ke": 1e6,
        "rigid_joint_linear_kd": 10.0,
        "rigid_joint_angular_kd": 10.0,
    },
    "joints": {
        "enabled": True,
        "linear_break_force": 5e5,
        "angular_break_torque": 5e5,
        "damage_rate": 5.0,
        "heal_rate": 0.01,
        "instant_break_force": 2e6,
        "instant_break_torque": 2e6,
    },
    "render": {
        "default_mode": "mesh",
        "mesh_color": [0.6, 0.85, 0.9],
    },
    "mesh_splitter": {
        "base_rot": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        "base_offset": [0.0, 0.0, 1.5],
        "deform_method": "rigid",
    },
    "force_modes": {
        "strength": 5000.0,
    },
    "elasticity": {
        "stiffness": 0.0,
        "damping": 0.0,
    },
}
