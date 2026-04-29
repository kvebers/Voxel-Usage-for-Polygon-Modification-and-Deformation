import numpy as np
import warp as wp


def health_to_color(t):
    if t <= 0.0:
        return (0.9, 0.15, 0.1)
    elif t <= 0.25:
        s = t / 0.25
        return (0.9 + s * 0.1, 0.15 + s * 0.4, 0.1 - s * 0.1)
    elif t <= 0.5:
        s = (t - 0.25) / 0.25
        return (1.0 - s * 0.05, 0.55 + s * 0.35, s * 0.1)
    elif t <= 0.75:
        s = (t - 0.5) / 0.25
        return (0.95 - s * 0.75, 0.9 - s * 0.05, 0.1 + s * 0.7)
    else:
        s = (t - 0.75) / 0.25
        return (0.2 + s * 0.15, 0.85 - s * 0.15, 0.8 + s * 0.15)


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
        n = len(self.neighbor_pairs)

        self.damage = np.zeros(n, np.float32)
        self.broken = np.zeros(n, dtype=bool)
        self.linear_break_force = linear_break_force
        self.angular_break_torque = angular_break_torque
        self.damage_rate = damage_rate
        self.heal_rate = heal_rate
        self.instant_break_force = instant_break_force
        self.instant_break_torque = instant_break_torque

        self.enabled = n > 0 and model.joint_count >= n

        self.max_counts = np.zeros(self.voxel_count, np.int32)
        if not self.enabled:
            return
        if "joint_start" in scene:
            self.joint_enabled_offset = scene["joint_start"]
        else:
            self.joint_enabled_offset = model.joint_count - n
        self._constraint_start = self.solver.joint_constraint_start.numpy().copy()
        self._constraint_dim = self.solver.joint_constraint_dim.numpy().copy()

        for ia, ib in self.neighbor_pairs:
            self.max_counts[ia] += 1
            self.max_counts[ib] += 1

        self._voxel_joints = [[] for _ in range(self.voxel_count)]
        for ji, (ia, ib) in enumerate(self.neighbor_pairs):
            self._voxel_joints[ia].append(ji)
            self._voxel_joints[ib].append(ji)
        self._break_log_count = 0
        self._max_breaks_per_step = max_breaks_per_step
        pairs = np.array(self.neighbor_pairs, dtype=np.int32).reshape(-1, 2)
        self._ba = self.vbs + pairs[:, 0]
        self._bb = self.vbs + pairs[:, 1]
        self._joint_enabled_cache = model.joint_enabled.numpy().copy()

    def get_voxel_colors(self):
        pairs = np.array(self.neighbor_pairs, dtype=np.int32).reshape(-1, 2)
        active = ~self.broken
        counts = np.zeros(self.voxel_count, np.int32)
        np.add.at(counts, pairs[active, 0], 1)
        np.add.at(counts, pairs[active, 1], 1)
        safe_max = np.maximum(self.max_counts, 1)
        t = np.clip(counts.astype(np.float32) / safe_max, 0.0, 1.0)
        colors = np.zeros((self.voxel_count, 3), np.float32)
        for i in range(self.voxel_count):
            colors[i] = health_to_color(t[i])
        return colors

    def _compute_damage(self, body_torques, dt):
        diff = body_torques[self._ba] - body_torques[self._bb]
        ang_stress = np.linalg.norm(diff, axis=1)
        instant = ang_stress > self.instant_break_torque
        norm_stress = np.maximum(
            0.0, ang_stress / max(self.angular_break_torque, 1e-8) - 0.5
        )
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

    def _get_newly_broken(self, instant):
        n = len(self.neighbor_pairs)
        newly_broken = (~self.broken) & (instant | (self.damage >= 1.0))
        newly_broken_indices = np.where(newly_broken)[0]
        if len(newly_broken_indices) > self._max_breaks_per_step:
            newly_broken_indices = newly_broken_indices[: self._max_breaks_per_step]
            newly_broken = np.zeros(n, dtype=bool)
            newly_broken[newly_broken_indices] = True
        return newly_broken, newly_broken_indices

    def update(self, model, dt, device, body_torques=None):
        if not self.enabled:
            return

        joint_enabled = self._joint_enabled_cache
        if body_torques is None:
            body_torques = self.solver.body_torques.numpy()

        instant = self._compute_damage(body_torques, dt)
        newly_broken, newly_broken_indices = self._get_newly_broken(instant)

        broken_count = int(np.sum(newly_broken))
        if broken_count > 0:
            self.broken |= newly_broken
            self.damage[newly_broken] = 1.0
            offset = self.joint_enabled_offset
            for ji in newly_broken_indices:
                model_idx = offset + ji
                if model_idx < len(joint_enabled):
                    joint_enabled[model_idx] = 0
            self._flush_to_solver(model, joint_enabled, device)
            if self._break_log_count < 15:
                print(
                    f"  [break] {broken_count} this step, {int(np.sum(self.broken))} total"
                )
                self._break_log_count += 1

    def _mark_broken(self, ji, joint_enabled):
        self.broken[ji] = True
        self.damage[ji] = 1.0
        model_idx = self.joint_enabled_offset + ji
        if model_idx < len(joint_enabled):
            joint_enabled[model_idx] = 0

    def _extract_penalty_arrays(self):
        s = self.solver
        return (
            s.joint_penalty_k.numpy(),
            s.joint_penalty_k_max.numpy(),
            s.joint_penalty_kd.numpy(),
            s.joint_penalty_k_min.numpy(),
        )

    def _extract_state_arrays(self):
        s = self.solver
        return (
            s.joint_sigma_prev.numpy(),
            s.joint_sigma_start.numpy(),
            s.joint_kappa_prev.numpy(),
            s.joint_dkappa_prev.numpy(),
            s.joint_C_fric.numpy(),
        )

    def _zero_broken_penalties(self, penalty_k, penalty_k_max, penalty_kd, penalty_k_min):
        for ji in range(len(self.neighbor_pairs)):
            if not self.broken[ji]:
                continue
            model_idx = self.joint_enabled_offset + ji
            c_start = int(self._constraint_start[model_idx])
            c_dim = int(self._constraint_dim[model_idx])
            for c in range(c_start, c_start + c_dim):
                if c < len(penalty_k):
                    penalty_k[c] = 0.0
                    penalty_k_max[c] = 0.0
                    penalty_kd[c] = 0.0
                    penalty_k_min[c] = 0.0

    def _zero_broken_state(self, sigma_prev, sigma_start, kappa_prev, dkappa_prev, c_fric):
        zero3 = (0.0, 0.0, 0.0)
        for ji in range(len(self.neighbor_pairs)):
            if not self.broken[ji]:
                continue
            model_idx = self.joint_enabled_offset + ji
            if model_idx < len(sigma_prev):
                sigma_prev[model_idx] = zero3
            if model_idx < len(sigma_start):
                sigma_start[model_idx] = zero3
            if model_idx < len(kappa_prev):
                kappa_prev[model_idx] = zero3
            if model_idx < len(dkappa_prev):
                dkappa_prev[model_idx] = zero3
            if model_idx < len(c_fric):
                c_fric[model_idx] = zero3

    def _sync_penalty_arrays(self, penalty_k, penalty_k_max, penalty_kd, penalty_k_min, device):
        s = self.solver
        s.joint_penalty_k = wp.array(penalty_k, dtype=s.joint_penalty_k.dtype, device=device)
        s.joint_penalty_k_max = wp.array(penalty_k_max, dtype=s.joint_penalty_k_max.dtype, device=device)
        s.joint_penalty_kd = wp.array(penalty_kd, dtype=s.joint_penalty_kd.dtype, device=device)
        s.joint_penalty_k_min = wp.array(penalty_k_min, dtype=s.joint_penalty_k_min.dtype, device=device)

    def _sync_state_arrays(self, sigma_prev, sigma_start, kappa_prev, dkappa_prev, c_fric, device):
        s = self.solver
        s.joint_sigma_prev = wp.array(sigma_prev, dtype=s.joint_sigma_prev.dtype, device=device)
        s.joint_sigma_start = wp.array(sigma_start, dtype=s.joint_sigma_start.dtype, device=device)
        s.joint_kappa_prev = wp.array(kappa_prev, dtype=s.joint_kappa_prev.dtype, device=device)
        s.joint_dkappa_prev = wp.array(dkappa_prev, dtype=s.joint_dkappa_prev.dtype, device=device)
        s.joint_C_fric = wp.array(c_fric, dtype=s.joint_C_fric.dtype, device=device)

    def _flush_to_solver(self, model, joint_enabled, device):
        self._joint_enabled_cache = joint_enabled.copy()
        model.joint_enabled = wp.array(
            joint_enabled, dtype=model.joint_enabled.dtype, device=device
        )

        penalty_arrays = self._extract_penalty_arrays()
        self._zero_broken_penalties(*penalty_arrays)
        self._sync_penalty_arrays(*penalty_arrays, device)

        state_arrays = self._extract_state_arrays()
        self._zero_broken_state(*state_arrays)
        self._sync_state_arrays(*state_arrays, device)

        try:
            self.solver.notify_model_changed(None)
        except Exception:
            pass
