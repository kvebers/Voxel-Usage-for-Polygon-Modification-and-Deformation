import numpy as np
import warp as wp


def health_to_color(health):
    if health <= 0.0:
        return (0.9, 0.15, 0.1)
    elif health <= 0.25:
        interp_factor = health / 0.25
        return (
            0.9 + interp_factor * 0.1,
            0.15 + interp_factor * 0.4,
            0.1 - interp_factor * 0.1,
        )
    elif health <= 0.5:
        interp_factor = (health - 0.25) / 0.25
        return (
            1.0 - interp_factor * 0.05,
            0.55 + interp_factor * 0.35,
            interp_factor * 0.1,
        )
    elif health <= 0.75:
        interp_factor = (health - 0.5) / 0.25
        return (
            0.95 - interp_factor * 0.75,
            0.9 - interp_factor * 0.05,
            0.1 + interp_factor * 0.7,
        )
    else:
        interp_factor = (health - 0.75) / 0.25
        return (
            0.2 + interp_factor * 0.15,
            0.85 - interp_factor * 0.15,
            0.8 + interp_factor * 0.15,
        )


def health_to_colors(health_values):
    health_values = np.asarray(health_values, dtype=np.float32)
    r = np.empty_like(health_values)
    g = np.empty_like(health_values)
    b = np.empty_like(health_values)
    mask = health_values <= 0.0
    r[mask], g[mask], b[mask] = 0.9, 0.15, 0.1
    mask = (health_values > 0.0) & (health_values <= 0.25)
    interp_factor = health_values[mask] / 0.25
    r[mask] = 0.9 + interp_factor * 0.1
    g[mask] = 0.15 + interp_factor * 0.4
    b[mask] = 0.1 - interp_factor * 0.1
    mask = (health_values > 0.25) & (health_values <= 0.5)
    interp_factor = (health_values[mask] - 0.25) / 0.25
    r[mask] = 1.0 - interp_factor * 0.05
    g[mask] = 0.55 + interp_factor * 0.35
    b[mask] = interp_factor * 0.1
    mask = (health_values > 0.5) & (health_values <= 0.75)
    interp_factor = (health_values[mask] - 0.5) / 0.25
    r[mask] = 0.95 - interp_factor * 0.75
    g[mask] = 0.9 - interp_factor * 0.05
    b[mask] = 0.1 + interp_factor * 0.7
    mask = health_values > 0.75
    interp_factor = (health_values[mask] - 0.75) / 0.25
    r[mask] = 0.2 + interp_factor * 0.15
    g[mask] = 0.85 - interp_factor * 0.15
    b[mask] = 0.8 + interp_factor * 0.15
    return np.stack([r, g, b], axis=1)


class JointBreaker:
    def __init__(
        self,
        scene,
        model,
        linear_break_force=5e5,
        angular_break_torque=5e5,
        damage_rate=0.4,
        heal_rate=0.05,
        instant_break_force=2e6,
        instant_break_torque=2e6,
        max_breaks_per_step=5,
    ):
        self.neighbor_pairs = scene["neighbor_pairs"]
        self.positions = scene["positions"]
        self.vbs = scene["voxel_body_start"]
        self.voxel_count = scene["voxel_count"]
        self.solver = scene["solver"]
        joint_count = len(self.neighbor_pairs)
        self.damage = np.zeros(joint_count, np.float32)
        self.broken = np.zeros(joint_count, dtype=bool)
        self.linear_break_force = linear_break_force
        self.angular_break_torque = angular_break_torque
        self.damage_rate = damage_rate
        self.heal_rate = heal_rate
        self.instant_break_force = instant_break_force
        self.instant_break_torque = instant_break_torque
        self.enabled = joint_count > 0 and model.joint_count >= joint_count
        self.max_counts = np.zeros(self.voxel_count, np.int32)
        if not self.enabled:
            return
        self._init_joint_tracking(model, joint_count, max_breaks_per_step, scene)

    def _init_joint_tracking(self, model, n, max_breaks_per_step, scene):
        if "joint_start" in scene:
            self.joint_enabled_offset = scene["joint_start"]
        else:
            self.joint_enabled_offset = model.joint_count - n
        self._constraint_start = self.solver.joint_constraint_start.numpy().copy()
        self._constraint_dim = self.solver.joint_constraint_dim.numpy().copy()
        for voxel_a, voxel_b in self.neighbor_pairs:
            self.max_counts[voxel_a] += 1
            self.max_counts[voxel_b] += 1
        self._voxel_joints = [[] for _ in range(self.voxel_count)]
        for joint_index, (voxel_a, voxel_b) in enumerate(self.neighbor_pairs):
            self._voxel_joints[voxel_a].append(joint_index)
            self._voxel_joints[voxel_b].append(joint_index)
        self._max_breaks_per_step = max_breaks_per_step
        pairs = np.array(self.neighbor_pairs, dtype=np.int32).reshape(-1, 2)
        self._parent_body_indices = self.vbs + pairs[:, 0]
        self._child_body_indices = self.vbs + pairs[:, 1]
        self._joint_enabled_cache = model.joint_enabled.numpy().copy()

    def get_voxel_colors(self):
        pairs = np.array(self.neighbor_pairs, dtype=np.int32).reshape(-1, 2)
        active = ~self.broken
        counts = np.zeros(self.voxel_count, np.int32)
        np.add.at(counts, pairs[active, 0], 1)
        np.add.at(counts, pairs[active, 1], 1)
        safe_max = np.maximum(self.max_counts, 1)
        t = np.clip(counts.astype(np.float32) / safe_max, 0.0, 1.0)
        return health_to_colors(t)

    def compute_damage(self, body_torques, dt):
        diff = body_torques[self._parent_body_indices] - body_torques[self._child_body_indices]
        ang_stress = np.linalg.norm(diff, axis=1)
        instant = ang_stress > self.instant_break_torque
        norm_stress = np.maximum(0.0, ang_stress / max(self.angular_break_torque, 1e-8) - 0.5)
        active = ~self.broken & ~instant
        self.damage = np.where(
            active & (norm_stress > 0.0),
            np.minimum(1.0, self.damage + norm_stress * self.damage_rate * dt * 60.0),
            np.where(
                active,
                np.maximum(0.0, self.damage - self.heal_rate * dt * 60.0),
                self.damage,
            ),
        )
        return instant

    def get_newly_broken(self, instant):
        joint_count = len(self.neighbor_pairs)
        newly_broken = (~self.broken) & (instant | (self.damage >= 1.0))
        newly_broken_indices = np.where(newly_broken)[0]
        if len(newly_broken_indices) > self._max_breaks_per_step:
            newly_broken_indices = newly_broken_indices[: self._max_breaks_per_step]
            newly_broken = np.zeros(joint_count, dtype=bool)
            newly_broken[newly_broken_indices] = True
        return newly_broken, newly_broken_indices

    def apply_breaks(self, model, newly_broken, newly_broken_indices, joint_enabled, device):
        self.broken |= newly_broken
        self.damage[newly_broken] = 1.0
        offset = self.joint_enabled_offset
        for joint_index in newly_broken_indices:
            model_index = offset + joint_index
            if model_index < len(joint_enabled):
                joint_enabled[model_index] = 0
        self.flush_to_solver(model, joint_enabled, device)

    def update(self, model, dt, device, body_torques=None):
        if not self.enabled:
            return
        joint_enabled = self._joint_enabled_cache
        if body_torques is None:
            body_torques = self.solver.body_torques.numpy()
        instant = self.compute_damage(body_torques, dt)
        newly_broken, newly_broken_indices = self.get_newly_broken(instant)

        if np.any(newly_broken):
            self.apply_breaks(model, newly_broken, newly_broken_indices, joint_enabled, device)

    def mark_broken(self, ji, joint_enabled):
        self.broken[ji] = True
        self.damage[ji] = 1.0
        model_index = self.joint_enabled_offset + ji
        if model_index < len(joint_enabled):
            joint_enabled[model_index] = 0

    def get_penalty_data(self):
        solver = self.solver
        return (
            solver.joint_penalty_k.numpy(),
            solver.joint_penalty_k_max.numpy(),
            solver.joint_penalty_kd.numpy(),
            solver.joint_penalty_k_min.numpy(),
        )

    def get_state_data(self):
        solver = self.solver
        return (
            solver.joint_sigma_prev.numpy(),
            solver.joint_sigma_start.numpy(),
            solver.joint_kappa_prev.numpy(),
            solver.joint_dkappa_prev.numpy(),
            solver.joint_C_fric.numpy(),
        )

    def no_broken_penalties(self, penalty_k, penalty_k_max, penalty_kd, penalty_k_min):
        for ji in range(len(self.neighbor_pairs)):
            if not self.broken[ji]:
                continue
            model_index = self.joint_enabled_offset + ji
            c_start = int(self._constraint_start[model_index])
            c_dim = int(self._constraint_dim[model_index])
            for c in range(c_start, c_start + c_dim):
                if c < len(penalty_k):
                    penalty_k[c] = 0.0
                    penalty_k_max[c] = 0.0
                    penalty_kd[c] = 0.0
                    penalty_k_min[c] = 0.0

    def no_broken_state(self, sigma_prev, sigma_start, kappa_prev, dkappa_prev, c_fric):
        zero3 = (0.0, 0.0, 0.0)
        for ji in range(len(self.neighbor_pairs)):
            if not self.broken[ji]:
                continue
            model_index = self.joint_enabled_offset + ji
            if model_index < len(sigma_prev):
                sigma_prev[model_index] = zero3
            if model_index < len(sigma_start):
                sigma_start[model_index] = zero3
            if model_index < len(kappa_prev):
                kappa_prev[model_index] = zero3
            if model_index < len(dkappa_prev):
                dkappa_prev[model_index] = zero3
            if model_index < len(c_fric):
                c_fric[model_index] = zero3

    def sync_penalty_arrays(self, penalty_k, penalty_k_max, penalty_kd, penalty_k_min, device):
        solver = self.solver
        solver.joint_penalty_k = wp.array(penalty_k, dtype=solver.joint_penalty_k.dtype, device=device)
        solver.joint_penalty_k_max = wp.array(penalty_k_max, dtype=solver.joint_penalty_k_max.dtype, device=device)
        solver.joint_penalty_kd = wp.array(penalty_kd, dtype=solver.joint_penalty_kd.dtype, device=device)
        solver.joint_penalty_k_min = wp.array(penalty_k_min, dtype=solver.joint_penalty_k_min.dtype, device=device)

    def sync_state_arrays(self, sigma_prev, sigma_start, kappa_prev, dkappa_prev, c_fric, device):
        solver = self.solver
        solver.joint_sigma_prev = wp.array(sigma_prev, dtype=solver.joint_sigma_prev.dtype, device=device)
        solver.joint_sigma_start = wp.array(sigma_start, dtype=solver.joint_sigma_start.dtype, device=device)
        solver.joint_kappa_prev = wp.array(kappa_prev, dtype=solver.joint_kappa_prev.dtype, device=device)
        solver.joint_dkappa_prev = wp.array(dkappa_prev, dtype=solver.joint_dkappa_prev.dtype, device=device)
        solver.joint_C_fric = wp.array(c_fric, dtype=solver.joint_C_fric.dtype, device=device)

    def flush_to_solver(self, model, joint_enabled, device):
        self._joint_enabled_cache = joint_enabled.copy()
        model.joint_enabled = wp.array(joint_enabled, dtype=model.joint_enabled.dtype, device=device)
        penalty_arrays = self.get_penalty_data()
        self.no_broken_penalties(*penalty_arrays)
        self.sync_penalty_arrays(*penalty_arrays, device)
        state_arrays = self.get_state_data()
        self.no_broken_state(*state_arrays)
        self.sync_state_arrays(*state_arrays, device)
        try:
            self.solver.notify_model_changed(None)
        except Exception:
            pass
