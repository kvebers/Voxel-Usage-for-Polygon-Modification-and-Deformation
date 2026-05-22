import math
import numpy as np
import pygame
from pygame.locals import *
from src.math_utils import perspective_matrix, look_at_matrix


def _yaw_pitch_from_direction(d):
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return 0.0, 0.0
    n = d / dist
    pitch = math.degrees(-math.asin(float(np.clip(n[2], -1.0, 1.0))))
    yaw = math.degrees(math.atan2(float(-n[1]), float(-n[0])))
    return yaw, pitch


def init_camera(cfg):
    ccfg = cfg.camera
    target = np.array(ccfg.target, dtype=np.float32)
    pos_cfg = getattr(ccfg, "position", None)
    if pos_cfg is not None:
        pos = np.array(pos_cfg, dtype=np.float32)
    else:
        yaw_r = math.radians(ccfg.yaw)
        pitch_r = math.radians(ccfg.pitch)
        d = ccfg.distance
        pos = target + np.array(
            [d * math.cos(pitch_r) * math.cos(yaw_r),
             d * math.cos(pitch_r) * math.sin(yaw_r),
             d * math.sin(pitch_r)],
            dtype=np.float32,
        )
    yaw, pitch = _yaw_pitch_from_direction(target - pos)
    return {
        "position": pos,
        "yaw": yaw,
        "pitch": pitch,
        "fov": ccfg.fov,
        "near": ccfg.near,
        "far": ccfg.far,
        "speed": getattr(ccfg, "speed", 0.02),
        "mouse_sensitivity": getattr(ccfg, "mouse_sensitivity", 0.15),
    }


def _camera_basis_vectors(yaw_deg):
    yaw_r = math.radians(yaw_deg)
    fwd = np.array([-math.cos(yaw_r), -math.sin(yaw_r), 0.0], np.float32)
    right = np.array([-math.sin(yaw_r), math.cos(yaw_r), 0.0], np.float32)
    up = np.array([0.0, 0.0, 1.0], np.float32)
    return fwd, right, up


def movement(cam):
    keys = pygame.key.get_pressed()
    if not any([keys[K_w], keys[K_s], keys[K_a], keys[K_d], keys[K_q], keys[K_e]]):
        return
    speed = cam["speed"]
    fwd, right, up = _camera_basis_vectors(cam["yaw"])
    if keys[K_w]:
        cam["position"] += fwd * speed
    if keys[K_s]:
        cam["position"] -= fwd * speed
    if keys[K_d]:
        cam["position"] += right * speed
    if keys[K_a]:
        cam["position"] -= right * speed
    if keys[K_e]:
        cam["position"] += up * speed
    if keys[K_q]:
        cam["position"] -= up * speed


def compute_view_projection(cam, width, height):
    aspect = width / max(height, 1)
    proj = perspective_matrix(cam["fov"], aspect, cam["near"], cam["far"])
    eye = cam["position"]
    yaw_r = math.radians(cam["yaw"])
    pitch_r = math.radians(cam["pitch"])
    look_fwd = np.array(
        [-math.cos(pitch_r) * math.cos(yaw_r),
         -math.cos(pitch_r) * math.sin(yaw_r),
         -math.sin(pitch_r)],
        dtype=np.float32,
    )
    view = look_at_matrix(eye, eye + look_fwd, np.array([0, 0, 1], np.float32))
    return proj, view, eye
