import numpy as np
import warp as wp
import newton


def _add_ground(builder, cfg):
    gnd = cfg.ground
    gx, gy, gz = gnd.position
    builder.add_body(xform=wp.transform(p=wp.vec3(gx, gy, gz), q=wp.quat_identity()))
    hx, hy, hz = gnd.half_extents
    ground_cfg = newton.ModelBuilder.ShapeConfig(
        density=gnd.density, ke=gnd.ke, kd=gnd.kd, kf=gnd.kf, mu=gnd.mu
    )
    builder.add_shape_box(body=builder.body_count - 1, hx=hx, hy=hy, hz=hz, cfg=ground_cfg)


def _add_voxel_bodies(builder, positions, half, cfg, block_halves_world=None):
    voxel_body_start = builder.body_count
    vox = cfg.voxels
    shape_kwargs = dict(
        density=vox.density,
        kf=getattr(vox, "kf", 1e4),
        mu=getattr(vox, "mu", 1.0),
    )
    ke = getattr(vox, "ke", None)
    kd = getattr(vox, "kd", None)
    if ke is not None:
        shape_kwargs["ke"] = ke
    if kd is not None:
        shape_kwargs["kd"] = kd
    voxel_cfg = newton.ModelBuilder.ShapeConfig(**shape_kwargs)
    q = wp.quat_identity()
    body_idx = voxel_body_start
    if block_halves_world is not None:
        for i, pos in enumerate(positions):
            builder.add_body(xform=wp.transform(p=wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])), q=q))
            hx, hy, hz = block_halves_world[i]
            builder.add_shape_box(body=body_idx, hx=hx, hy=hy, hz=hz, cfg=voxel_cfg)
            body_idx += 1
    else:
        for pos in positions:
            builder.add_body(xform=wp.transform(p=wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])), q=q))
            builder.add_shape_box(body=body_idx, hx=half, hy=half, hz=half, cfg=voxel_cfg)
            body_idx += 1
    return voxel_body_start


def _add_ball(builder, positions, half, extent, cfg):
    bcfg = cfg.ball
    if not getattr(bcfg, "enabled", True):
        return [], 0.0
    count = max(1, int(getattr(bcfg, "count", 1)))
    ball_radius = extent * bcfg.radius_factor
    max_z = float(positions[:, 2].max())
    ball_spawn_z = max_z + half + ball_radius + extent * bcfg.height_factor
    ball_shape_cfg = newton.ModelBuilder.ShapeConfig(
        density=bcfg.density, ke=bcfg.ke, kd=bcfg.kd, kf=bcfg.kf, mu=bcfg.mu
    )
    spacing = ball_radius * 3.0
    x_start = -(count - 1) * spacing * 0.5
    ball_bodies = []
    for i in range(count):
        x = x_start + i * spacing
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(x, 0.0, ball_spawn_z), q=wp.quat_identity())
        )
        builder.add_shape_sphere(body, radius=ball_radius, cfg=ball_shape_cfg)
        ball_bodies.append(body)
    return ball_bodies, ball_radius


def _compute_joint_offsets(pairs, positions, joint_world_offsets):
    if joint_world_offsets is not None:
        wo = np.asarray(joint_world_offsets, dtype=np.float64)
        return wo[:, 0], wo[:, 1]
    oa_all = (positions[pairs[:, 1]] - positions[pairs[:, 0]]) * 0.5
    return oa_all, -oa_all


def _add_joints(builder, neighbor_pairs, positions, voxel_body_start, joint_world_offsets=None):
    if not neighbor_pairs:
        return
    q = wp.quat_identity()
    pairs = np.asarray(neighbor_pairs, dtype=np.int32)
    oa_all, ob_all = _compute_joint_offsets(pairs, positions, joint_world_offsets)
    for ji in range(len(pairs)):
        ba = voxel_body_start + int(pairs[ji, 0])
        bb = voxel_body_start + int(pairs[ji, 1])
        ox_a, oy_a, oz_a = float(oa_all[ji, 0]), float(oa_all[ji, 1]), float(oa_all[ji, 2])
        ox_b, oy_b, oz_b = float(ob_all[ji, 0]), float(ob_all[ji, 1]), float(ob_all[ji, 2])
        builder.add_joint_fixed(
            ba, bb,
            parent_xform=wp.transform(p=wp.vec3(ox_a, oy_a, oz_a), q=q),
            child_xform=wp.transform(p=wp.vec3(ox_b, oy_b, oz_b), q=q),
        )


def _create_solver(model, cfg):
    scfg = cfg.solver
    return newton.solvers.SolverVBD(
        model,
        iterations=scfg.iterations,
        rigid_body_contact_buffer_size=scfg.rigid_body_contact_buffer_size,
        rigid_contact_k_start=scfg.rigid_contact_k_start,
        rigid_avbd_beta=scfg.rigid_avbd_beta,
        rigid_joint_linear_ke=scfg.rigid_joint_linear_ke,
        rigid_joint_angular_ke=scfg.rigid_joint_angular_ke,
        rigid_joint_linear_kd=scfg.rigid_joint_linear_kd,
        rigid_joint_angular_kd=scfg.rigid_joint_angular_kd,
    )
