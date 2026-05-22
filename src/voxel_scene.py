import numpy as np
from scipy.spatial import KDTree
from src.voxel_gpu import voxelize_gpu
from src.voxel_repair import make_hollow, repair_isolated_voxels
from src.voxel_topology import greedy_merge_grid, build_block_topology


def _build_greedy_topology(active_grid):
    blocks = greedy_merge_grid(active_grid)
    block_centers, block_halves_voxel, neighbor_pairs, joint_voxel_offsets = (
        build_block_topology(blocks, active_grid.shape)
    )
    coords = [tuple(float(v) for v in c) for c in block_centers]
    return coords, {}, neighbor_pairs, block_halves_voxel, joint_voxel_offsets


def _build_regular_topology(active_grid):
    filled = np.argwhere(active_grid > 0)
    coords = [(int(ix), int(iy), int(iz)) for ix, iy, iz in filled]
    coord_to_idx = {c: i for i, c in enumerate(coords)}
    neighbor_pairs = []
    for ix, iy, iz in coords:
        for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            nb = (ix + dx, iy + dy, iz + dz)
            if nb in coord_to_idx:
                neighbor_pairs.append((coord_to_idx[(ix, iy, iz)], coord_to_idx[nb]))
    return coords, coord_to_idx, neighbor_pairs, None, None


def voxelize_and_build_topology(
    mesh_verts, mesh_indices, resolution=32, fill_mode=True,
    ensure_connected=False, greedy_merge=False, ctx=None,
):
    tri_verts = mesh_verts[mesh_indices].reshape(-1, 3)
    grid_filled = voxelize_gpu(tri_verts, resolution=resolution, ctx=ctx)
    if fill_mode:
        grid_hollow = None
        active_grid = grid_filled
    else:
        grid_hollow = make_hollow(grid_filled)
        active_grid = grid_hollow
    if ensure_connected and fill_mode:
        active_grid = repair_isolated_voxels(active_grid, grid_filled)

    if greedy_merge:
        coords, coord_to_idx, neighbor_pairs, block_halves_voxel, joint_voxel_offsets = (
            _build_greedy_topology(active_grid)
        )
    else:
        coords, coord_to_idx, neighbor_pairs, block_halves_voxel, joint_voxel_offsets = (
            _build_regular_topology(active_grid)
        )
    return (
        grid_filled, grid_hollow, active_grid,
        coords, coord_to_idx, neighbor_pairs,
        block_halves_voxel, joint_voxel_offsets,
    )


def bind_vertices_to_voxels(mesh_verts, coords, coord_to_idx, grid_min, voxel_size, resolution):
    coords_arr = np.array(coords, dtype=np.float32)
    voxel_centers = grid_min + (coords_arr + 0.5) * voxel_size
    _, nearest = KDTree(voxel_centers).query(mesh_verts)
    bindings = nearest.astype(np.int32)
    offsets = (mesh_verts - voxel_centers[bindings]).astype(np.float32)
    return bindings, offsets


def _compute_voxel_geometry(mesh_verts, resolution, cfg):
    pad = cfg.voxels.padding
    vmin = mesh_verts.min(axis=0)
    vmax = mesh_verts.max(axis=0)
    extent = (vmax - vmin).max()
    usable = 1.0 - 2.0 * pad
    voxel_size = (1.0 / resolution) / usable * extent
    half = voxel_size * 0.5
    grid_min = vmin - pad * extent / usable
    return extent, usable, voxel_size, half, grid_min


def _apply_position_offsets(raw, cfg, half, world_offset):
    gnd = cfg.ground
    ground_top = gnd.position[2] + gnd.half_extents[2]
    z_offset = ground_top - raw[:, 2].min() + half
    ox, oy, oz = world_offset
    raw[:, 0] += ox
    raw[:, 1] += oy
    raw[:, 2] += z_offset + oz


def _compute_voxel_positions(coords, resolution, cfg, mesh_verts, half, world_offset=(0.0, 0.0, 0.0)):
    pad = cfg.voxels.padding
    vmin = mesh_verts.min(axis=0)
    extent = (mesh_verts.max(axis=0) - vmin).max()
    usable = 1.0 - 2.0 * pad
    coords_arr = np.array(coords, dtype=np.float64)
    if len(coords_arr) == 0:
        raise ValueError("Voxel Cord Err")
    n = (coords_arr + 0.5) / resolution
    p = (n - pad) / usable * extent + vmin
    raw = np.empty((len(coords_arr), 3), dtype=np.float64)
    raw[:, 0] = p[:, 0]
    raw[:, 1] = -p[:, 2]
    raw[:, 2] = p[:, 1]
    _apply_position_offsets(raw, cfg, half, world_offset)
    return raw
