import math
import numpy as np


def perspective_matrix(fov_deg, aspect, near, far):
    focal_length = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = focal_length / aspect
    matrix[1, 1] = focal_length
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def look_at_matrix(eye, target, up):
    eye = np.asarray(eye, np.float32)
    target = np.asarray(target, np.float32)
    up = np.asarray(up, np.float32)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    view_matrix = np.eye(4, dtype=np.float32)
    view_matrix[0, :3] = right
    view_matrix[1, :3] = camera_up
    view_matrix[2, :3] = -forward
    view_matrix[0, 3] = -np.dot(right, eye)
    view_matrix[1, 3] = -np.dot(camera_up, eye)
    view_matrix[2, 3] = np.dot(forward, eye)
    return view_matrix


def quat_to_mat3(q):
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def quat_to_mat4(q, pos):
    rotation_mat = quat_to_mat3(q)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation_mat
    matrix[:3, 3] = pos
    return matrix


# does spinning
def batch_quat_to_mat3(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rotation_matrices = np.empty((len(q), 3, 3), dtype=np.float32)
    rotation_matrices[:, 0, 0] = 1 - 2 * (y2 + z2)
    rotation_matrices[:, 0, 1] = 2 * (xy - wz)
    rotation_matrices[:, 0, 2] = 2 * (xz + wy)
    rotation_matrices[:, 1, 0] = 2 * (xy + wz)
    rotation_matrices[:, 1, 1] = 1 - 2 * (x2 + z2)
    rotation_matrices[:, 1, 2] = 2 * (yz - wx)
    rotation_matrices[:, 2, 0] = 2 * (xz - wy)
    rotation_matrices[:, 2, 1] = 2 * (yz + wx)
    rotation_matrices[:, 2, 2] = 1 - 2 * (x2 + y2)
    return rotation_matrices
