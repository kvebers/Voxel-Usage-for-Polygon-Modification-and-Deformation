import numpy as np
from types import SimpleNamespace
from src.mesh.load_obj import load_obj
from src.voxels.voxel_scene import voxelize_and_build_topology


def obj_config_from_spec(obj_spec, cfg):
    return SimpleNamespace(
        path=obj_spec.path,
        offset=list(obj_spec.offset) if hasattr(obj_spec, "offset") else [0.0, 0.0, 0.0],
        color=list(obj_spec.color) if hasattr(obj_spec, "color") else list(cfg.render.mesh_color),
        resolution=obj_spec.resolution if hasattr(obj_spec, "resolution") else cfg.mesh.resolution,
        rot=list(obj_spec.rot) if hasattr(obj_spec, "rot") else None,
        scale=float(obj_spec.scale) if hasattr(obj_spec, "scale") else 1.0,
    )


def get_object_configs(cfg):
    if hasattr(cfg, "objects") and cfg.objects:
        return [obj_config_from_spec(obj_spec, cfg) for obj_spec in cfg.objects]
    return [
        SimpleNamespace(
            path=cfg.mesh.path,
            offset=[0.0, 0.0, 0.0],
            color=list(cfg.render.mesh_color),
            resolution=cfg.mesh.resolution,
        )
    ]


def load_and_preprocess_mesh(obj_config):
    vertices, indices = load_obj(obj_config.path)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    verts = vertices - center
    if obj_config.rot is not None:
        verts = (np.array(obj_config.rot, dtype=np.float32) @ verts.T).T
    if obj_config.scale != 1.0:
        verts = verts * obj_config.scale
    return verts, indices


def load_and_voxelize_one(cfg, obj_config, ctx=None):
    fill_mode = bool(getattr(cfg.voxels, "fill_mode", True))
    ensure_connected = bool(getattr(cfg.voxels, "ensure_connected", False))
    greedy_merge = bool(getattr(cfg.voxels, "greedy_merge", False))
    centered_verts, indices = load_and_preprocess_mesh(obj_config)
    (
        _,
        _,
        _,
        coords,
        coord_to_index,
        neighbor_pairs,
        block_halves_voxel,
        joint_voxel_offsets,
    ) = voxelize_and_build_topology(
        centered_verts,
        indices,
        obj_config.resolution,
        fill_mode=fill_mode,
        ensure_connected=ensure_connected,
        greedy_merge=greedy_merge,
        ctx=ctx,
    )
    return {
        "centered_verts": centered_verts,
        "indices": indices,
        "coords": coords,
        "coord_to_index": coord_to_index,
        "neighbor_pairs": neighbor_pairs,
        "block_halves_voxel": block_halves_voxel,
        "joint_voxel_offsets": joint_voxel_offsets,
        "resolution": obj_config.resolution,
        "offset": obj_config.offset,
        "color": obj_config.color,
    }


def scale_greedy_to_world(block_halves_voxel, joint_voxel_offsets, voxel_size):
    block_halves_world = None
    joint_world_offsets = None
    if block_halves_voxel is not None:
        block_halves_world = [
            (half_x * voxel_size, half_z * voxel_size, half_y * voxel_size)
            for half_x, half_y, half_z in block_halves_voxel
        ]
    if joint_voxel_offsets is not None:
        joint_world_offsets = [
            (
                (
                    parent_offset[0] * voxel_size,
                    -parent_offset[2] * voxel_size,
                    parent_offset[1] * voxel_size,
                ),
                (
                    child_offset[0] * voxel_size,
                    -child_offset[2] * voxel_size,
                    child_offset[1] * voxel_size,
                ),
            )
            for parent_offset, child_offset in joint_voxel_offsets
        ]
    return block_halves_world, joint_world_offsets
