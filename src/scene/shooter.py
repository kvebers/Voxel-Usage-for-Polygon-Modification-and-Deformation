import math
import numpy as np
import warp as wp


class Shooter:
    def __init__(self, shooter_cfg, ball_bodies, radius):
        self.speed  = float(getattr(shooter_cfg, "speed", 8.0))
        self.color  = tuple(getattr(shooter_cfg, "color", [0.9, 0.3, 0.2]))
        self.bodies = ball_bodies
        self.radius = radius
        self._idx   = 0
        self._active = set()  # bodies that have been fired

    def shoot(self, camera, scene):
        if not self.bodies:
            return
        body = self.bodies[self._idx]
        self._idx = (self._idx + 1) % len(self.bodies)
        self._active.add(body)

        yaw   = math.radians(camera["yaw"])
        pitch = math.radians(camera["pitch"])
        fwd   = np.array([
            -math.cos(pitch) * math.cos(yaw),
            -math.cos(pitch) * math.sin(yaw),
            -math.sin(pitch),
        ], dtype=np.float32)

        # Spawn exactly at camera — always free space, never inside geometry.
        spawn = np.array(camera["position"], dtype=np.float32)
        vel   = fwd * self.speed

        device     = wp.get_cuda_device()
        body_q_np  = scene["state_current"].body_q.numpy().copy()
        body_qd_np = scene["state_current"].body_qd.numpy().copy()

        body_q_np[body, :3]  = spawn
        body_q_np[body, 3:]  = [0.0, 0.0, 0.0, 1.0]
        body_qd_np[body, :3] = vel
        body_qd_np[body, 3:] = 0.0

        # Only update state_current — let the solver compute state_next naturally.
        # Writing to state_next confuses Newton's warm-start and can cause explosions.
        q_arr  = wp.array(body_q_np,  dtype=wp.transform,      device=device)
        qd_arr = wp.array(body_qd_np, dtype=wp.spatial_vector, device=device)
        wp.copy(scene["state_current"].body_q,  q_arr)
        wp.copy(scene["state_current"].body_qd, qd_arr)

        print(f"[shooter] fired body={body} from {spawn} vel={vel}")
