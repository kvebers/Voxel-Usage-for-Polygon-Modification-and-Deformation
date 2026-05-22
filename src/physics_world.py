import numpy as np
import warp as wp
import newton
from src.voxel_scene import _compute_voxel_geometry, _compute_voxel_positions, bind_vertices_to_voxels
from src.scene_setup import _scale_greedy_to_world
from src.physics_bodies import _add_ground, _add_voxel_bodies, _add_ball, _add_joints, _create_solver
from src.joints import JointBreaker
from src.mesh_split import MeshSplitter


def _compute_object_world_geometry(obj_data, cfg):
    mesh_verts = obj_data["centered_verts"]
    coords = obj_data["coords"]
    coord_to_idx = obj_data["coord_to_idx"]
    resolution = obj_data["resolution"]
    world_offset = tuple(obj_data["offset"])
    extent, _, voxel_size, half, grid_min = _compute_voxel_geometry(mesh_verts, resolution, cfg)
    positions = _compute_voxel_positions(coords, resolution, cfg, mesh_verts, half, world_offset=world_offset)
    bindings, offsets = bind_vertices_to_voxels(mesh_verts, coords, coord_to_idx, grid_min, voxel_size, resolution)
    block_halves_world, joint_world_offsets = _scale_greedy_to_world(
        obj_data["block_halves_voxel"], obj_data["joint_voxel_offsets"], voxel_size
    )
    return positions, bindings, offsets, block_halves_world, joint_world_offsets, extent, half


def _finalize_model(builder, cfg):
    builder.color()
    model = builder.finalize(device=wp.get_cuda_device())
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    solver = _create_solver(model, cfg)
    return model, state_0, state_1, solver


def build_scene_multi(all_obj_data, cfg):
    builder = newton.ModelBuilder()
    _add_ground(builder, cfg)

    per_obj = []
    joint_offset = 0
    first_extent = None
    all_positions_list = []

    for obj_data in all_obj_data:
        neighbor_pairs = obj_data["neighbor_pairs"]
        positions, bindings, offsets, block_halves_world, joint_world_offsets, extent, half = (
            _compute_object_world_geometry(obj_data, cfg)
        )
        if first_extent is None:
            first_extent = (extent, half)

        voxel_body_start = _add_voxel_bodies(builder, positions, half, cfg, block_halves_world=block_halves_world)
        if cfg.joints.enabled:
            _add_joints(builder, neighbor_pairs, positions, voxel_body_start, joint_world_offsets=joint_world_offsets)

        all_positions_list.append(positions)
        per_obj.append({
            "voxel_body_start": voxel_body_start,
            "voxel_count": len(obj_data["coords"]),
            "positions": positions,
            "neighbor_pairs": neighbor_pairs,
            "bindings": bindings,
            "offsets": offsets,
            "half": half,
            "block_halves_world": block_halves_world,
            "joint_start": joint_offset,
        })
        joint_offset += len(neighbor_pairs)

    extent0, half0 = first_extent
    all_positions = np.concatenate(all_positions_list, axis=0)
    ball_bodies, ball_radius = _add_ball(builder, all_positions, half0, extent0, cfg)
    model, state_0, state_1, solver = _finalize_model(builder, cfg)

    for obj in per_obj:
        obj["solver"] = solver

    return {
        "model": model,
        "state_0": state_0,
        "state_1": state_1,
        "control": model.control(),
        "contacts": model.contacts(),
        "solver": solver,
        "ball_bodies": ball_bodies,
        "ball_radius": ball_radius,
        "objects": per_obj,
    }


def create_joint_breaker(obj_scene, model, cfg):
    jcfg = cfg.joints
    scene_for_jb = obj_scene if jcfg.enabled else {**obj_scene, "neighbor_pairs": []}
    return JointBreaker(
        scene_for_jb,
        model,
        linear_break_force=jcfg.linear_break_force,
        angular_break_torque=jcfg.angular_break_torque,
        damage_rate=jcfg.damage_rate,
        heal_rate=jcfg.heal_rate,
        instant_break_force=jcfg.instant_break_force,
        instant_break_torque=jcfg.instant_break_torque,
        max_breaks_per_step=getattr(jcfg, "max_breaks_per_step", 5),
    )


def create_mesh_splitter(indices, obj_scene, centered_verts, neighbor_pairs, cfg):
    return MeshSplitter(
        indices,
        obj_scene["bindings"],
        obj_scene["offsets"],
        obj_scene["voxel_count"],
        neighbor_pairs,
        obj_scene["positions"],
        obj_scene["half"],
        centered_verts,
        cfg,
    )
