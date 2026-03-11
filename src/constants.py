import numpy as np

DRIFT_DAMP_COEFF = 5.0
DRIFT_DAMP_THRESHOLD = 0.3
DEFAULT_BASE_ROT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
LIGHT_DIR = np.array([-1.0, -0.5, 1.0], dtype=np.float32)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)
FACE_DEFS = [
    (
        (-1, 0, 0),
        lambda block_bounds: (
            block_bounds[0],
            block_bounds[0],
            block_bounds[1],
            block_bounds[4],
            block_bounds[2],
            block_bounds[5],
        ),
    ),
    (
        (+1, 0, 0),
        lambda block_bounds: (
            block_bounds[3],
            block_bounds[3],
            block_bounds[1],
            block_bounds[4],
            block_bounds[2],
            block_bounds[5],
        ),
    ),
    (
        (0, -1, 0),
        lambda block_bounds: (
            block_bounds[0],
            block_bounds[3],
            block_bounds[1],
            block_bounds[1],
            block_bounds[2],
            block_bounds[5],
        ),
    ),
    (
        (0, +1, 0),
        lambda block_bounds: (
            block_bounds[0],
            block_bounds[3],
            block_bounds[4],
            block_bounds[4],
            block_bounds[2],
            block_bounds[5],
        ),
    ),
    (
        (0, 0, -1),
        lambda block_bounds: (
            block_bounds[0],
            block_bounds[3],
            block_bounds[1],
            block_bounds[4],
            block_bounds[2],
            block_bounds[2],
        ),
    ),
    (
        (0, 0, +1),
        lambda block_bounds: (
            block_bounds[0],
            block_bounds[3],
            block_bounds[1],
            block_bounds[4],
            block_bounds[5],
            block_bounds[5],
        ),
    ),
]
DEFAULTS = {
    "objects": [
        {
            "path": "obj/teapot.obj",
            "offset": [0, 0.0, 0.0],
            "color": [0.7, 0.95, 0.2],
            "resolution": 8,
        },
    ],
    "camera": {
        "target": [0.0, 0.0, 1.5],
        "position": [-5.0, -4.0, 3.0],
        "fov": 50,
        "near": 0.1,
        "far": 200.0,
    },
    "window": {
        "width": 1280,
        "height": 720,
        "title": "Vokseļu izmantošana poligonu modifikācijā un deformācijā.",
        "clear_color": [0.08, 0.09, 0.12, 1.0],
    },
    "simulation": {
        "fps": 60,
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
        "fill_mode": True,
        "ensure_connected": False,
        "greedy_merge": False,
    },
    "balls": [],
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
    "elasticity": {
        "stiffness": 0.0,
        "damping": 0.0,
    },
}
