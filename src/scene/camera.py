import math
import numpy as np
import pygame
from pygame.locals import *
from src.utils.math_utils import perspective_matrix, look_at_matrix


def camera_rotation(direction):
    dist = float(np.linalg.norm(direction))
    if dist < 1e-6:
        return 0.0, 0.0
    normalized_dir = direction / dist
    pitch = math.degrees(-math.asin(float(np.clip(normalized_dir[2], -1.0, 1.0))))
    yaw = math.degrees(math.atan2(float(-normalized_dir[1]), float(-normalized_dir[0])))
    return yaw, pitch


def init_camera(cfg):
    camera_config = cfg.camera
    target = np.array(camera_config.target, dtype=np.float32)
    camera_pos = np.array(camera_config.position, dtype=np.float32)
    yaw, pitch = camera_rotation(target - camera_pos)
    return {
        "position": camera_pos,
        "yaw": yaw,
        "pitch": pitch,
        "fov": camera_config.fov,
        "near": camera_config.near,
        "far": camera_config.far,
        "speed": getattr(camera_config, "speed", 0.02),
        "mouse_sensitivity": getattr(camera_config, "mouse_sensitivity", 0.15),
    }


def camera_vectors(yaw_deg):
    yaw = math.radians(yaw_deg)
    forward = np.array([-math.cos(yaw), -math.sin(yaw), 0.0], np.float32)
    right = np.array([-math.sin(yaw), math.cos(yaw), 0.0], np.float32)
    up = np.array([0.0, 0.0, 1.0], np.float32)
    return forward, right, up


def movement(cam):
    keys = pygame.key.get_pressed()
    if not any([keys[K_w], keys[K_s], keys[K_a], keys[K_d], keys[K_q], keys[K_e]]):
        return
    speed = cam["speed"]
    fwd, right, up = camera_vectors(cam["yaw"])
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
        [
            -math.cos(pitch_r) * math.cos(yaw_r),
            -math.cos(pitch_r) * math.sin(yaw_r),
            -math.sin(pitch_r),
        ],
        dtype=np.float32,
    )
    view = look_at_matrix(eye, eye + look_fwd, np.array([0, 0, 1], np.float32))
    return proj, view, eye
