import numpy as np
from scipy.spatial import KDTree
from src.voxels.voxel_gpu import voxelize_gpu
from src.voxels.voxel_repair import make_hollow, repair_isolated_voxels
from src.voxels.voxel_topology import greedy_merge_grid, build_block_topology


def build_greedy_mesh_topology(active_grid):
    """
    Add joints for the greedy mesh
    """
    blocks = greedy_merge_grid(active_grid)
    block_centers, block_halves_voxel, neighbor_pairs, joint_voxel_offsets = build_block_topology(blocks, active_grid.shape)
    coords = [tuple(float(v) for v in c) for c in block_centers]
    return coords, {}, neighbor_pairs, block_halves_voxel, joint_voxel_offsets


def build_mesh_topology(active_grid):
    """
    Just gets the joints
    """
    filled = np.argwhere(active_grid > 0)
    coords = [(int(ix), int(iy), int(iz)) for ix, iy, iz in filled]
    coord_to_index = {c: i for i, c in enumerate(coords)}
    neighbor_pairs = []
    for ix, iy, iz in coords:  # TODO bottle neck
        for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            neighbor_coord = (ix + dx, iy + dy, iz + dz)
            if neighbor_coord in coord_to_index:
                neighbor_pairs.append((coord_to_index[(ix, iy, iz)], coord_to_index[neighbor_coord]))
    return coords, coord_to_index, neighbor_pairs, None, None


def voxelize_and_build_topology(mesh_verts, mesh_indices, resolution=32, fill_mode=True, ensure_connected=False, greedy_merge=False, ctx=None):
    tri_verts = mesh_verts[mesh_indices].reshape(-1, 3)  # ensures tris
    grid_filled = voxelize_gpu(tri_verts, resolution=resolution, ctx=ctx)
    if fill_mode:  # return grid structure
        grid_hollow = None
        active_grid = grid_filled
    else:  # empty the nest
        grid_hollow = make_hollow(grid_filled)
        active_grid = grid_hollow
    if ensure_connected and fill_mode:  # fix the
        active_grid = repair_isolated_voxels(active_grid, grid_filled)
    if greedy_merge:
        (coords, coord_to_index, neighbor_pairs, block_halves_voxel, joint_voxel_offsets) = build_greedy_mesh_topology(active_grid)
    else:
        (coords, coord_to_index, neighbor_pairs, block_halves_voxel, joint_voxel_offsets) = build_mesh_topology(active_grid)
    return (grid_filled, grid_hollow, active_grid, coords, coord_to_index, neighbor_pairs, block_halves_voxel, joint_voxel_offsets)


def bind_vertices_to_voxels(mesh_verts, coords, coord_to_index, grid_min, voxel_size, resolution):
    coords_arr = np.array(coords, dtype=np.float32)
    voxel_centers = grid_min + (coords_arr + 0.5) * voxel_size  # world position
    _, nearest = KDTree(voxel_centers).query(mesh_verts)  # find closest voxel
    bindings = nearest.astype(np.int32)  # which voxel owns each vertex
    offsets = (mesh_verts - voxel_centers[bindings]).astype(np.float32)  # how far vertex is from its voxel center
    return bindings, offsets


def create_voxel_geometry_helper(mesh_verts, resolution, cfg):
    pad = cfg.voxels.padding  # clear
    vmin = mesh_verts.min(axis=0)
    vmax = mesh_verts.max(axis=0)
    extent = (vmax - vmin).max()  # mesh size
    usable = 1.0 - 2.0 * pad  # apply padding
    voxel_size = (1.0 / resolution) / usable * extent  # vox      according to the world
    half = voxel_size * 0.5
    grid_min = vmin - pad * extent / usable  # start of the grid
    return extent, usable, voxel_size, half, grid_min


def add_offset_bug_fix(raw, cfg, half, world_offset):
    """
    Adds offset from ground prevents bug
    """
    ground_config = cfg.ground
    ground_top = ground_config.position[2] + ground_config.half_extents[2]
    z_offset = ground_top - raw[:, 2].min() + half
    ox, oy, oz = world_offset
    raw[:, 0] += ox
    raw[:, 1] += oy
    raw[:, 2] += z_offset + oz


def get_voxel_positions(coords, resolution, cfg, mesh_verts, half, world_offset=(0.0, 0.0, 0.0)):
    """
    Get voxel position in world space
    """
    pad = cfg.voxels.padding
    vmin = mesh_verts.min(axis=0)
    extent = (mesh_verts.max(axis=0) - vmin).max()
    usable = 1.0 - 2.0 * pad
    coords_arr = np.array(coords, dtype=np.float64)
    if len(coords_arr) == 0:
        raise ValueError("Voxel Cord Err")
    normalized_coords = (coords_arr + 0.5) / resolution
    voxel_world_coords = (normalized_coords - pad) / usable * extent + vmin
    raw = np.empty((len(coords_arr), 3))
    raw[:, 0] = voxel_world_coords[:, 0]
    raw[:, 1] = -voxel_world_coords[:, 2]
    raw[:, 2] = voxel_world_coords[:, 1]
    add_offset_bug_fix(raw, cfg, half, world_offset)  # bug fix
    return raw
