import numpy as np
import warp as wp

from src.constants import DRIFT_DAMP_COEFF, DRIFT_DAMP_THRESHOLD


def subtract_component_mean(arr, labels):
    result = arr.copy()
    for comp_id in np.unique(labels):
        mask = labels == comp_id
        result[mask] -= arr[mask].mean(axis=0, keepdims=True)
    return result


def drifting_bug_fix_up(lin_vel, voxel_mass):
    lin_vel = np.nan_to_num(lin_vel, nan=0.0, posinf=0.0, neginf=0.0)
    speed = np.linalg.norm(lin_vel, axis=1, keepdims=True)
    blend = 1.0 / (1.0 + (speed / DRIFT_DAMP_THRESHOLD) ** 2)
    return (DRIFT_DAMP_COEFF * voxel_mass) * lin_vel * blend


def elastic_restoring_delta(displacement, lin_vel, labels, stiffness, damping, voxel_mass):
    deformation = subtract_component_mean(displacement, labels)
    delta = (stiffness * voxel_mass) * deformation
    if damping != 0.0:
        vel_deform = subtract_component_mean(lin_vel, labels)
        delta += (damping * voxel_mass) * vel_deform
    return delta


class ForceApplier:
    def __init__(
        self,
        rest_positions,
        voxel_body_start,
        voxel_count,
        stiffness=0.0,
        damping=0.0,
        voxel_mass=1.0,
        neighbor_pairs=None,
    ):
        self.voxel_body_start = voxel_body_start
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.voxel_mass = float(voxel_mass)
        self.rest_pos = np.array(rest_positions, dtype=np.float32)[:, :3]
        self._neighbor_pairs = list(neighbor_pairs) if neighbor_pairs is not None else []

    def apply_elastic(self, state, device, broken=None):
        voxel_count = len(self.rest_pos)
        voxel_body_start = self.voxel_body_start
        voxel_mass = self.voxel_mass
        body_f = state.body_f.numpy().copy()
        lin_vel = state.body_qd.numpy()[voxel_body_start : voxel_body_start + voxel_count, :3].astype(np.float32)
        body_f[voxel_body_start : voxel_body_start + voxel_count, :3] -= drifting_bug_fix_up(lin_vel, voxel_mass)
        if self.stiffness == 0.0 and self.damping == 0.0:
            state.body_f = wp.array(body_f, dtype=state.body_f.dtype, device=device)
            return
        cur_pos = state.body_q.numpy()[voxel_body_start : voxel_body_start + voxel_count, :3].astype(np.float32)
        displacement = cur_pos - self.rest_pos
        if broken is not None and len(broken) > 0 and np.any(broken):
            labels = self.component_labels(broken)
        else:
            labels = np.zeros(voxel_count, dtype=np.int32)
        body_f[voxel_body_start : voxel_body_start + voxel_count, :3] -= elastic_restoring_delta(
            displacement,
            lin_vel,
            labels,
            self.stiffness,
            self.damping,
            voxel_mass,
        )
        state.body_f = wp.array(body_f, dtype=state.body_f.dtype, device=device)

    def component_labels(self, broken):
        n = len(self.rest_pos)
        parent = np.arange(n, dtype=np.int32)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for joint_index, (voxel_a, voxel_b) in enumerate(self._neighbor_pairs):
            if not broken[joint_index]:
                root_a, root_b = find(voxel_a), find(voxel_b)
                if root_a != root_b:
                    parent[root_a] = root_b
        return np.array([find(i) for i in range(n)], dtype=np.int32)
