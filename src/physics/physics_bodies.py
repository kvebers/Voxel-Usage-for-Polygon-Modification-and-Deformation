import numpy as np
import warp as wp
import newton


def add_ground(builder, cfg):
    ground_config = cfg.ground
    gx, gy, gz = ground_config.position
    builder.add_body(xform=wp.transform(p=wp.vec3(gx, gy, gz), q=wp.quat_identity()))
    hx, hy, hz = ground_config.half_extents
    ground_shape_cfg = newton.ModelBuilder.ShapeConfig(
        density=ground_config.density, ke=ground_config.ke, kd=ground_config.kd, kf=ground_config.kf, mu=ground_config.mu
    )
    builder.add_shape_box(body=builder.body_count - 1, hx=hx, hy=hy, hz=hz, cfg=ground_shape_cfg)


def add_voxel_bodies(builder, positions, half, cfg, block_halves_world=None):
    voxel_body_start = builder.body_count
    voxel_config = cfg.voxels
    shape_kwargs = dict(density=voxel_config.density, kf=getattr(voxel_config, "kf", 1e4), mu=getattr(voxel_config, "mu", 1.0))
    ke = getattr(voxel_config, "ke", None)
    kd = getattr(voxel_config, "kd", None)
    if ke is not None:
        shape_kwargs["ke"] = ke
    if kd is not None:
        shape_kwargs["kd"] = kd
    voxel_shape_cfg = newton.ModelBuilder.ShapeConfig(**shape_kwargs)
    identity_quat = wp.quat_identity()
    body_index = voxel_body_start
    if block_halves_world is not None:
        for i, pos in enumerate(positions):
            builder.add_body(xform=wp.transform(p=wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])), q=identity_quat))
            hx, hy, hz = block_halves_world[i]
            builder.add_shape_box(body=body_index, hx=hx, hy=hy, hz=hz, cfg=voxel_shape_cfg)
            body_index += 1
    else:
        for pos in positions:
            builder.add_body(xform=wp.transform(p=wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])), q=identity_quat))
            builder.add_shape_box(body=body_index, hx=half, hy=half, hz=half, cfg=voxel_shape_cfg)
            body_index += 1
    return voxel_body_start


def quat_from_rotation_matrix(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return wp.quat(float(x), float(y), float(z), float(w))


def add_walls(builder, cfg):
    walls_cfg = getattr(cfg, "walls", [])
    if not walls_cfg:
        return
    for wall_config in walls_cfg:
        # corners[0..3] define the visible face CCW from front; p0→p1 = local X, p0→p3 = local Y
        corners = np.array(wall_config.corners, dtype=np.float64)
        p0, p1, p2, p3 = corners[0], corners[1], corners[2], corners[3]
        center = (p0 + p1 + p2 + p3) / 4.0
        edge_x = p1 - p0
        edge_y = p3 - p0
        hx = float(np.linalg.norm(edge_x) / 2.0)
        hy = float(np.linalg.norm(edge_y) / 2.0)
        hz = float(wall_config.depth) / 2.0
        right = edge_x / (2.0 * hx)
        normal = np.cross(right, edge_y)
        normal /= np.linalg.norm(normal)
        up = np.cross(normal, right)
        R = np.column_stack([right, up, normal])
        q = quat_from_rotation_matrix(R)
        wall_shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=wall_config.ke,
            kd=wall_config.kd,
            kf=wall_config.kf,
            mu=wall_config.mu,
        )
        builder.add_body(xform=wp.transform(p=wp.vec3(float(center[0]), float(center[1]), float(center[2])), q=q))
        builder.add_shape_box(body=builder.body_count - 1, hx=hx, hy=hy, hz=hz, cfg=wall_shape_cfg)


def add_ball(builder, positions, half, extent, cfg):
    balls_cfg = getattr(cfg, "balls", [])
    if not balls_cfg:
        return [], [], []
    max_z = float(positions[:, 2].max())
    ball_bodies = []
    ball_radii = []
    for ball_config in balls_cfg:
        ball_radius = extent * ball_config.radius_factor
        ball_shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=ball_config.density, ke=ball_config.ke, kd=ball_config.kd, kf=ball_config.kf, mu=ball_config.mu
        )
        pos = getattr(ball_config, "position", None)
        if pos is not None:
            spawn = wp.vec3(float(pos[0]), float(pos[1]), float(pos[2]))
        else:
            height_factor = getattr(ball_config, "height_factor", 2.0)
            ball_spawn_z = max_z + half + ball_radius + extent * height_factor
            offset = getattr(ball_config, "offset", [0.0, 0.0, 0.0])
            spawn = wp.vec3(float(offset[0]), float(offset[1]), ball_spawn_z + float(offset[2]))
        body = builder.add_body(xform=wp.transform(p=spawn, q=wp.quat_identity()))
        builder.add_shape_sphere(body, radius=ball_radius, cfg=ball_shape_cfg)
        ball_bodies.append(body)
        ball_radii.append(ball_radius)
    return ball_bodies, ball_radii, balls_cfg


def compute_joint_offsets(pairs, positions, joint_world_offsets):
    if joint_world_offsets is not None:
        world_offsets_arr = np.asarray(joint_world_offsets)
        return world_offsets_arr[:, 0], world_offsets_arr[:, 1]
    parent_joint_offsets = (positions[pairs[:, 1]] - positions[pairs[:, 0]]) * 0.5
    return parent_joint_offsets, -parent_joint_offsets


def add_joints(builder, neighbor_pairs, positions, voxel_body_start, joint_world_offsets=None):
    if not neighbor_pairs:
        return
    identity_quat = wp.quat_identity()
    pairs = np.asarray(neighbor_pairs, dtype=np.int32)
    parent_joint_offsets, child_joint_offsets = compute_joint_offsets(pairs, positions, joint_world_offsets)
    for joint_index in range(len(pairs)):
        parent_body = voxel_body_start + int(pairs[joint_index, 0])
        child_body = voxel_body_start + int(pairs[joint_index, 1])
        parent_ox, parent_oy, parent_oz = (
            float(parent_joint_offsets[joint_index, 0]),
            float(parent_joint_offsets[joint_index, 1]),
            float(parent_joint_offsets[joint_index, 2]),
        )
        child_ox, child_oy, child_oz = (
            float(child_joint_offsets[joint_index, 0]),
            float(child_joint_offsets[joint_index, 1]),
            float(child_joint_offsets[joint_index, 2]),
        )
        builder.add_joint_fixed(
            parent_body,
            child_body,
            parent_xform=wp.transform(p=wp.vec3(parent_ox, parent_oy, parent_oz), q=identity_quat),
            child_xform=wp.transform(p=wp.vec3(child_ox, child_oy, child_oz), q=identity_quat),
        )


def create_solver(model, cfg):
    solver_config = cfg.solver
    return newton.solvers.SolverVBD(
        model,
        iterations=solver_config.iterations,
        rigid_body_contact_buffer_size=solver_config.rigid_body_contact_buffer_size,
        rigid_contact_k_start=solver_config.rigid_contact_k_start,
        rigid_avbd_beta=solver_config.rigid_avbd_beta,
        rigid_joint_linear_ke=solver_config.rigid_joint_linear_ke,
        rigid_joint_angular_ke=solver_config.rigid_joint_angular_ke,
        rigid_joint_linear_kd=solver_config.rigid_joint_linear_kd,
        rigid_joint_angular_kd=solver_config.rigid_joint_angular_kd,
    )
