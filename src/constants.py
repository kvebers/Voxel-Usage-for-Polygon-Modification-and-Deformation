import numpy as np

# Temportary bug fix for the sliding, but needs to be adressed later because it super bad
DRIFT_DAMP_BUG_FIX_COEFF = 5.0  # TODO remover later, in version 2.0 or smt
DRIFT_DAMP_BUGFIX_THRESHOLD = 0.3  # TODO remover later, in version 2.0 or smt


# Rotation Matrix for elements, but messed up needs to be reworked later #TODO
DEFAULT_ROTATION_ELEMENT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)

# Light normalization for the simulation
LIGHT_DIR = np.array([-1.0, -0.5, 1.0], dtype=np.float32)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)

# 6 faces cube
FACE_DEFS = [
    ((-1, 0, 0), lambda faces: (faces[0], faces[0], faces[1], faces[4], faces[2], faces[5])),
    ((+1, 0, 0), lambda faces: (faces[3], faces[3], faces[1], faces[4], faces[2], faces[5])),
    ((0, -1, 0), lambda faces: (faces[0], faces[3], faces[1], faces[1], faces[2], faces[5])),
    ((0, +1, 0), lambda faces: (faces[0], faces[3], faces[4], faces[4], faces[2], faces[5])),
    ((0, 0, -1), lambda faces: (faces[0], faces[3], faces[1], faces[4], faces[2], faces[2])),
    ((0, 0, +1), lambda faces: (faces[0], faces[3], faces[1], faces[4], faces[5], faces[5])),
]

# Default scene config, everything here can be overriden by scene files from scenes/ or scene.jsonc
DEFAULTS = {
    # list of objects to load in
    "objects": [
        {
            "path": "obj/teapot.obj",  # obj
            "offset": [0, 0.0, 0.0],  # cordinate
            "color": [0.7, 0.95, 0.2],  # color to use
            "resolution": 8,  # resolution 8 for fast and working up to 32 to recommended if you have better device go more
        }
    ],
    # Cameraa
    "camera": {
        "target": [0.0, 0.0, 1.5],  # target to look at I like to add the cordinate of the object
        "position": [-5.0, -4.0, 3.0],  # position
        "fov": 50,  # camera FOV
        "near": 0.1,  # object cliping parameter
        "far": 100.0,  # far cliping parameter
    },
    # window settings
    "window": {
        "width": 1280,
        "height": 720,
        "title": "Vokseļu izmantošana poligonu modifikācijā un deformācijā.",
        "clear_color": [0.08, 0.09, 0.12, 1.0],  # background
    },
    # sim time
    "simulation": {
        "fps": 60,  # fps
        "start_paused": True,  # sim start paused
    },
    "ground": {
        "position": [0.0, 0.0, -0.5],  # ground need to reduce to remove cliping side effect
        "half_extents": [50.0, 50.0, 0.5],  # size
        "density": 0.0,
        "ke": 1e6,  # contact stiffness
        "kd": 1e3,  # how fast fill fall back
        "kf": 1e3,  # contact friction
        "mu": 0.5,  # combined friction
    },
    "voxels": {
        "density": 500.0,  # density
        "padding": 0.01,  # padding but should be removed #TODO glitchy does not help could be usefull with # enable false
        "fill_mode": True,  # fill mode false or true for shell generation
        "ensure_connected": False,  # makes sure that voxels are connected to each other
        "greedy_merge": False,  # just merge the voxel grid
    },
    # ball
    "balls": [],
    "solver": {
        "iterations": 10,  # itterations need for stabilit / perfomance change lower to get more performance higer stability
        "rigid_body_contact_buffer_size": 4096,  # total buffer size lately found out bigger is better
        "rigid_contact_k_start": 1e5,  # stifness penalty
        "rigid_avbd_beta": 1e4,  # AVBD regularization weight
        "rigid_joint_linear_ke": 1e6,  # block sprint penalty
        "rigid_joint_angular_ke": 1e6,  # joint stiffnes
        "rigid_joint_linear_kd": 10.0,  # joint regulation or dambing
        "rigid_joint_angular_kd": 10.0,  # joint regulation or angular damping
    },
    # Basically since AVBD newton did not expose the parameters I needed to do it manually.
    # I tried to make the API fork and pull request but it works to slowly, but that is #TODO for later
    "joints": {
        "enabled": True,
        "angular_break_torque": 5e5,  # angular break damage accumulation
        "damage_rate": 0.1,  # damage per sec per stress
        "heal_rate": 0.01,  # healing from stress should be high for elastic low for rigid, can simulate multiple impacts with 0 # not realistic but it is what it is
        "instant_break_torque": 2e6,  # force for instant angular break
    },
    "render": {
        "default_mode": "mesh",  # mesh or voxel
        "mesh_color": [0.6, 0.85, 0.9],  # TODO remove in patch 2 or smt
    },
    # Seperate component from newton force applier that helps to return blocks in their place
    "elasticity": {
        "stiffness": 0.0,  # stiffness coeficient of the object
        "damping": 0.0,  # how quickly it normalized, coefficient is sqrt of stiffness
    },
}
