import numpy as np
from src.constants import FACE_DEFS


def greedy_merge_grid(grid: np.ndarray) -> list:
    remaining = grid.astype(bool).copy()
    blocks = []
    for x0, y0, z0 in np.argwhere(remaining):
        x0, y0, z0 = int(x0), int(y0), int(z0)
        if not remaining[x0, y0, z0]:
            continue
        x1 = x0
        while x1 + 1 < grid.shape[0] and remaining[x1 + 1, y0, z0]:
            x1 += 1
        y1 = y0
        while y1 + 1 < grid.shape[1] and remaining[x0 : x1 + 1, y1 + 1, z0].all():
            y1 += 1
        z1 = z0
        while z1 + 1 < grid.shape[2] and remaining[x0 : x1 + 1, y0 : y1 + 1, z1 + 1].all():
            z1 += 1
        blocks.append((x0, y0, z0, x1, y1, z1))
        remaining[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] = False
    return blocks


def _build_block_geometry(blocks: list, grid_shape: tuple):
    voxel_to_block = np.full(grid_shape, -1, dtype=np.int32)
    for bi, (x0, y0, z0, x1, y1, z1) in enumerate(blocks):
        voxel_to_block[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] = bi
    block_centers = [
        ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0)
        for x0, y0, z0, x1, y1, z1 in blocks
    ]
    block_halves_voxel = [
        ((x1 - x0 + 1) / 2.0, (y1 - y0 + 1) / 2.0, (z1 - z0 + 1) / 2.0)
        for x0, y0, z0, x1, y1, z1 in blocks
    ]
    return voxel_to_block, block_centers, block_halves_voxel


def _scan_block_faces(blocks, voxel_to_block, grid_shape):
    pair_joints: dict = {}
    for bi, blk in enumerate(blocks):
        for (dx, dy, dz), face_fn in FACE_DEFS:
            fx0, fx1, fy0, fy1, fz0, fz1 = face_fn(blk)
            fxs, fys, fzs = np.meshgrid(
                np.arange(fx0, fx1 + 1),
                np.arange(fy0, fy1 + 1),
                np.arange(fz0, fz1 + 1),
                indexing="ij",
            )
            fxs, fys, fzs = fxs.ravel(), fys.ravel(), fzs.ravel()
            axs, ays, azs = fxs + dx, fys + dy, fzs + dz
            in_bounds = (
                (0 <= axs) & (axs < grid_shape[0])
                & (0 <= ays) & (ays < grid_shape[1])
                & (0 <= azs) & (azs < grid_shape[2])
            )
            axs, ays, azs = axs[in_bounds], ays[in_bounds], azs[in_bounds]
            fxs, fys, fzs = fxs[in_bounds], fys[in_bounds], fzs[in_bounds]
            others = voxel_to_block[axs, ays, azs]
            valid = (others >= 0) & (others != bi)
            for fx, fy, fz, other in zip(fxs[valid], fys[valid], fzs[valid], others[valid]):
                key = (min(bi, int(other)), max(bi, int(other)))
                jx = float(fx) + (1 + dx) / 2
                jy = float(fy) + (1 + dy) / 2
                jz = float(fz) + (1 + dz) / 2
                pair_joints.setdefault(key, set()).add((jx, jy, jz))
    return pair_joints


def _build_joint_offsets(pair_joints, block_centers):
    neighbor_pairs = []
    joint_voxel_offsets = []
    for (a, b), pts in pair_joints.items():
        cx_a = block_centers[a][0] + 0.5
        cy_a = block_centers[a][1] + 0.5
        cz_a = block_centers[a][2] + 0.5
        cx_b = block_centers[b][0] + 0.5
        cy_b = block_centers[b][1] + 0.5
        cz_b = block_centers[b][2] + 0.5
        for jx, jy, jz in pts:
            neighbor_pairs.append((a, b))
            joint_voxel_offsets.append((
                (jx - cx_a, jy - cy_a, jz - cz_a),
                (jx - cx_b, jy - cy_b, jz - cz_b),
            ))
    return neighbor_pairs, joint_voxel_offsets


def _enumerate_joint_contacts(blocks, voxel_to_block, block_centers, grid_shape):
    pair_joints = _scan_block_faces(blocks, voxel_to_block, grid_shape)
    return _build_joint_offsets(pair_joints, block_centers)


def build_block_topology(blocks: list, grid_shape: tuple):
    voxel_to_block, block_centers, block_halves_voxel = _build_block_geometry(blocks, grid_shape)
    neighbor_pairs, joint_voxel_offsets = _enumerate_joint_contacts(
        blocks, voxel_to_block, block_centers, grid_shape
    )
    return block_centers, block_halves_voxel, neighbor_pairs, joint_voxel_offsets
