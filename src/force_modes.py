import numpy as np
import warp as wp

from src.constants import DRIFT_DAMP_COEFF, DRIFT_DAMP_THRESHOLD


def _subtract_component_mean(arr, labels):
    result = arr.copy()
    for comp_id in np.unique(labels):
        mask = labels == comp_id
        result[mask] -= arr[mask].mean(axis=0, keepdims=True)
    return result


def _drift_damp_delta(lin_vel, m):
    lin_vel = np.nan_to_num(lin_vel, nan=0.0, posinf=0.0, neginf=0.0)
    speed = np.linalg.norm(lin_vel, axis=1, keepdims=True)
    blend = 1.0 / (1.0 + (speed / DRIFT_DAMP_THRESHOLD) ** 2)
    return (DRIFT_DAMP_COEFF * m) * lin_vel * blend


def _elastic_restoring_delta(displacement, lin_vel, labels, stiffness, damping, m):
    deformation = _subtract_component_mean(displacement, labels)
    delta = (stiffness * m) * deformation
    if damping != 0.0:
        vel_deform = _subtract_component_mean(lin_vel, labels)
        delta += (damping * m) * vel_deform
    return delta


class ForceApplier:
    def __init__(
        self,
        rest_positions,
        voxel_body_start,
        voxel_count,
        boundary_fraction=0.20,
        stiffness=0.0,
        damping=0.0,
        voxel_mass=1.0,
        neighbor_pairs=None,
    ):
        self.vbs = voxel_body_start
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.voxel_mass = float(voxel_mass)
        self.rest_pos = np.array(rest_positions, dtype=np.float32)[:, :3]
        self._neighbor_pairs = (
            list(neighbor_pairs) if neighbor_pairs is not None else []
        )
        pos = np.array(rest_positions, dtype=np.float64)
        center = pos.mean(axis=0)

        z = pos[:, 2]
        z_min, z_max = z.min(), z.max()
        span = max(z_max - z_min, 1e-8)

        self.top_idx = np.where(z >= z_max - span * boundary_fraction)[0]
        self.bot_idx = np.where(z <= z_min + span * boundary_fraction)[0]

        def _tangential(idx):
            """90-degree CCW rotation in XY of the radial unit vector."""
            p = pos[idx, :2] - center[:2]
            n = np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-8)
            return np.column_stack([-p[:, 1], p[:, 0]]) / n

        self.top_tang = _tangential(self.top_idx)  # (n_top, 2)
        self.bot_tang = _tangential(self.bot_idx)  # (n_bot, 2)

    def apply(self, state, mode, strength, device):
        if mode == 0 or strength == 0.0:
            return

        body_f = state.body_f.numpy().copy()
        top = self.vbs + self.top_idx
        bot = self.vbs + self.bot_idx

        if mode == 1:  # squeeze
            body_f[top, 2] -= strength
            body_f[bot, 2] += strength
        elif mode == 2:  # tension
            body_f[top, 2] += strength
            body_f[bot, 2] -= strength
        elif mode == 3:  # shear
            body_f[top, 0] += strength
            body_f[bot, 0] -= strength
        elif mode == 4:  # twist
            body_f[top, 0] += strength * self.top_tang[:, 0]
            body_f[top, 1] += strength * self.top_tang[:, 1]
            body_f[bot, 0] -= strength * self.bot_tang[:, 0]
            body_f[bot, 1] -= strength * self.bot_tang[:, 1]

        state.body_f = wp.array(body_f, dtype=state.body_f.dtype, device=device)

    def apply_elastic(self, state, device, broken=None):
        n, vbs, m = len(self.rest_pos), self.vbs, self.voxel_mass
        body_f = state.body_f.numpy().copy()
        lin_vel = state.body_qd.numpy()[vbs : vbs + n, :3].astype(np.float32)

        body_f[vbs : vbs + n, :3] -= _drift_damp_delta(lin_vel, m)

        if self.stiffness == 0.0 and self.damping == 0.0:
            state.body_f = wp.array(body_f, dtype=state.body_f.dtype, device=device)
            return

        cur_pos = state.body_q.numpy()[vbs : vbs + n, :3].astype(np.float32)
        displacement = cur_pos - self.rest_pos

        if broken is not None and len(broken) > 0 and np.any(broken):
            labels = self._component_labels(broken)
        else:
            labels = np.zeros(n, dtype=np.int32)

        body_f[vbs : vbs + n, :3] -= _elastic_restoring_delta(
            displacement, lin_vel, labels, self.stiffness, self.damping, m
        )
        state.body_f = wp.array(body_f, dtype=state.body_f.dtype, device=device)

    def _component_labels(self, broken):
        """Union-Find: label each voxel with its connected component index."""
        n = len(self.rest_pos)
        parent = np.arange(n, dtype=np.int32)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for ji, (ia, ib) in enumerate(self._neighbor_pairs):
            if not broken[ji]:
                ra, rb = find(ia), find(ib)
                if ra != rb:
                    parent[ra] = rb

        return np.array([find(i) for i in range(n)], dtype=np.int32)
