import numpy as np
import warp as wp
import newton
from src.voxels.voxel_scene import create_voxel_geometry, get_voxel_positions, bind_vertices_to_voxels
from src.scene_setup import scale_greedy_to_world
from src.physics.physics_bodies import add_ground, add_voxel_bodies, add_ball, add_walls, add_joints, create_solver, add_shooter_balls
from src.physics.joints import JointBreaker
from src.mesh.mesh_split import MeshSplitter


def compute_object_world_geometry(obj_data, cfg):
    mesh_verts = obj_data["centered_verts"]
    coords = obj_data["coords"]
    coord_to_index = obj_data["coord_to_index"]
    resolution = obj_data["resolution"]
    world_offset = tuple(obj_data["offset"])
    extent, _, voxel_size, half, grid_min = create_voxel_geometry(mesh_verts, resolution, cfg)
    positions = get_voxel_positions(coords, resolution, cfg, mesh_verts, half, world_offset=world_offset)
    bindings, offsets = bind_vertices_to_voxels(mesh_verts, coords, coord_to_index, grid_min, voxel_size, resolution)
    block_halves_world, joint_world_offsets = scale_greedy_to_world(obj_data["block_halves_voxel"], obj_data["joint_voxel_offsets"], voxel_size)
    return (positions, bindings, offsets, block_halves_world, joint_world_offsets, extent, half)


def finalize_model(builder, cfg):
    builder.color()
    model = builder.finalize(device=wp.get_cuda_device())
    state_current = model.state()
    state_next = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_current)
    solver = create_solver(model, cfg)
    return model, state_current, state_next, solver


def add_objects_to_builder(builder, all_obj_data, cfg):
    per_obj = []
    joint_offset = 0
    first_extent = None
    all_positions_list = []

    for obj_data in all_obj_data:
        neighbor_pairs = obj_data["neighbor_pairs"]
        (positions, bindings, offsets, block_halves_world, joint_world_offsets, extent, half) = compute_object_world_geometry(obj_data, cfg)
        if first_extent is None:
            first_extent = (extent, half)
        voxel_body_start = add_voxel_bodies(builder, positions, half, cfg, block_halves_world=block_halves_world)
        if cfg.joints.enabled:
            add_joints(builder, neighbor_pairs, positions, voxel_body_start, joint_world_offsets=joint_world_offsets)
        all_positions_list.append(positions)
        per_obj.append(
            {
                "voxel_body_start": voxel_body_start,
                "voxel_count": len(obj_data["coords"]),
                "positions": positions,
                "neighbor_pairs": neighbor_pairs,
                "bindings": bindings,
                "offsets": offsets,
                "half": half,
                "block_halves_world": block_halves_world,
                "joint_start": joint_offset,
            }
        )
        joint_offset += len(neighbor_pairs)
    return per_obj, all_positions_list, first_extent


def apply_ball_start_forces(state_current, state_next, ball_bodies, ball_cfgs):
    if not ball_bodies:
        return
    device = wp.get_cuda_device()
    body_qd_np = state_current.body_qd.numpy().copy()
    for body, ball_config in zip(ball_bodies, ball_cfgs):
        vel = getattr(ball_config, "initial_velocity", [0.0, 0.0, 0.0])
        body_qd_np[body, 0] = float(vel[0])
        body_qd_np[body, 1] = float(vel[1])
        body_qd_np[body, 2] = float(vel[2])
    vel_arr = wp.array(body_qd_np, dtype=wp.spatial_vector, device=device)
    wp.copy(state_current.body_qd, vel_arr)
    wp.copy(state_next.body_qd, vel_arr)


def build_scene_multi(all_obj_data, cfg):
    builder = newton.ModelBuilder()
    add_ground(builder, cfg)
    add_walls(builder, cfg)
    per_obj, all_positions_list, first_extent = add_objects_to_builder(builder, all_obj_data, cfg)
    first_object_extent, first_voxel_half = first_extent
    all_positions = np.concatenate(all_positions_list, axis=0)
    ball_bodies, ball_radii, ball_cfgs = add_ball(builder, all_positions, first_voxel_half, first_object_extent, cfg)
    shooter_bodies, shooter_radius = add_shooter_balls(builder, cfg)
    model, state_current, state_next, solver = finalize_model(builder, cfg)
    apply_ball_start_forces(state_current, state_next, ball_bodies, ball_cfgs)
    for obj in per_obj:
        obj["solver"] = solver
    return {
        "model": model,
        "state_current": state_current,
        "state_next": state_next,
        "control": model.control(),
        "contacts": model.contacts(),
        "solver": solver,
        "ball_bodies": ball_bodies,
        "ball_radii": ball_radii,
        "ball_cfgs": ball_cfgs,
        "shooter_bodies": shooter_bodies,
        "shooter_radius": shooter_radius,
        "objects": per_obj,
    }


def create_joint_breaker(obj_scene, model, cfg):
    joint_cfg = cfg.joints
    scene_for_jb = obj_scene if joint_cfg.enabled else {**obj_scene, "neighbor_pairs": []}
    return JointBreaker(
        scene_for_jb,
        model,
        angular_break_torque=joint_cfg.angular_break_torque,
        damage_rate=joint_cfg.damage_rate,
        heal_rate=joint_cfg.heal_rate,
        instant_break_torque=joint_cfg.instant_break_torque,
        max_breaks_per_step=getattr(joint_cfg, "max_breaks_per_step", 5),
    )


def create_mesh_splitter(indices, obj_scene, centered_verts, neighbor_pairs):
    return MeshSplitter(
        indices,
        obj_scene["bindings"],
        obj_scene["offsets"],
        obj_scene["voxel_count"],
        neighbor_pairs,
        obj_scene["positions"],
        obj_scene["half"],
        centered_verts,
    )
