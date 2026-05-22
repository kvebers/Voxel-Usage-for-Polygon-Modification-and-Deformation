import numpy as np
from types import SimpleNamespace
from src.load_obj import load_obj
from src.voxel_scene import voxelize_and_build_topology


def _obj_cfg_from_spec(o, cfg):
    return SimpleNamespace(
        path=o.path,
        offset=list(o.offset) if hasattr(o, "offset") else [0.0, 0.0, 0.0],
        color=list(o.color) if hasattr(o, "color") else list(cfg.render.mesh_color),
        resolution=o.resolution if hasattr(o, "resolution") else cfg.mesh.resolution,
        rot=list(o.rot) if hasattr(o, "rot") else None,
        scale=float(o.scale) if hasattr(o, "scale") else 1.0,
    )


def get_object_configs(cfg):
    if hasattr(cfg, "objects") and cfg.objects:
        return [_obj_cfg_from_spec(o, cfg) for o in cfg.objects]
    return [
        SimpleNamespace(
            path=cfg.mesh.path,
            offset=[0.0, 0.0, 0.0],
            color=list(cfg.render.mesh_color),
            resolution=cfg.mesh.resolution,
        )
    ]


def _load_and_preprocess_mesh(obj_cfg):
    vertices, indices = load_obj(obj_cfg.path)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    verts = vertices - center
    if obj_cfg.rot is not None:
        verts = (np.array(obj_cfg.rot, dtype=np.float32) @ verts.T).T
    if obj_cfg.scale != 1.0:
        verts = verts * obj_cfg.scale
    return verts, indices


def load_and_voxelize_one(cfg, obj_cfg, ctx=None):
    fill_mode = bool(getattr(cfg.voxels, "fill_mode", True))
    ensure_connected = bool(getattr(cfg.voxels, "ensure_connected", False))
    greedy_merge = bool(getattr(cfg.voxels, "greedy_merge", False))
    centered_verts, indices = _load_and_preprocess_mesh(obj_cfg)
    (_, _, _, coords, coord_to_idx, neighbor_pairs, block_halves_voxel, joint_voxel_offsets) = (
        voxelize_and_build_topology(
            centered_verts, indices, obj_cfg.resolution,
            fill_mode=fill_mode, ensure_connected=ensure_connected,
            greedy_merge=greedy_merge, ctx=ctx,
        )
    )
    return {
        "centered_verts": centered_verts,
        "indices": indices,
        "coords": coords,
        "coord_to_idx": coord_to_idx,
        "neighbor_pairs": neighbor_pairs,
        "block_halves_voxel": block_halves_voxel,
        "joint_voxel_offsets": joint_voxel_offsets,
        "resolution": obj_cfg.resolution,
        "offset": obj_cfg.offset,
        "color": obj_cfg.color,
    }


def _scale_greedy_to_world(block_halves_voxel, joint_voxel_offsets, voxel_size):
    block_halves_world = None
    joint_world_offsets = None
    if block_halves_voxel is not None:
        block_halves_world = [
            (hx * voxel_size, hz * voxel_size, hy * voxel_size)
            for hx, hy, hz in block_halves_voxel
        ]
    if joint_voxel_offsets is not None:
        joint_world_offsets = [
            (
                (oa[0] * voxel_size, -oa[2] * voxel_size, oa[1] * voxel_size),
                (ob[0] * voxel_size, -ob[2] * voxel_size, ob[1] * voxel_size),
            )
            for oa, ob in joint_voxel_offsets
        ]
    return block_halves_world, joint_world_offsets
