import numpy as np
from scipy.spatial import KDTree
from skimage.measure import marching_cubes


def _compute_blended_positions(voxel_pos, voxel_rots, vox_centers_mesh, sv, bi, bw, base_rot):
    blended = np.zeros((len(sv), 3), dtype=np.float32)
    for k in range(bi.shape[1]):
        vi = bi[:, k]
        w = bw[:, k, None]
        off_k = sv - vox_centers_mesh[vi]
        woff_k = (base_rot @ off_k.T).T
        p_k = voxel_pos[vi] + np.einsum("nij,nj->ni", voxel_rots[vi], woff_k)
        blended += w * p_k
    return blended


def _deform_rigid(
    *,
    voxel_pos,
    voxel_q,
    voxel_rots,
    bindings,
    offsets,
    base_rot,
    base_offset,
    n_total,
    mask,
    vox_centers_mesh,
    split_indices=None,
    blend_indices=None,
    blend_weights=None,
    split_mesh_verts=None,
    reach_mask=None,
    **_,
):
    bv = bindings[mask]
    positions = np.empty((n_total, 3), dtype=np.float32)

    if blend_indices is not None and blend_weights is not None and split_mesh_verts is not None:
        bi = blend_indices[mask]
        bw = blend_weights[mask].copy()
        sv = split_mesh_verts[mask]
        if reach_mask is not None:
            bw *= reach_mask.astype(np.float32)
        bw /= np.maximum(bw.sum(axis=1, keepdims=True), 1e-8)
        positions[mask] = _compute_blended_positions(
            voxel_pos, voxel_rots, vox_centers_mesh, sv, bi, bw, base_rot
        )
    else:
        world_offsets = (base_rot @ offsets[mask].T).T
        positions[mask] = voxel_pos[bv] + np.einsum("nij,nj->ni", voxel_rots[bv], world_offsets)

    if np.any(~mask):
        positions[~mask] = base_offset
    return positions


def _build_scalar_grid(voxel_pos, voxel_radius):
    pad = voxel_radius * 2.0
    mins = voxel_pos.min(axis=0) - pad
    maxs = voxel_pos.max(axis=0) + pad
    extent = maxs - mins
    cell_size = max(voxel_radius * 0.6, 1e-6)
    res = np.clip(np.ceil(extent / cell_size).astype(int), 8, 64)
    xs = np.linspace(mins[0], maxs[0], int(res[0]))
    ys = np.linspace(mins[1], maxs[1], int(res[1]))
    zs = np.linspace(mins[2], maxs[2], int(res[2]))
    spacing = (
        float(xs[-1] - xs[0]) / max(int(res[0]) - 1, 1),
        float(ys[-1] - ys[0]) / max(int(res[1]) - 1, 1),
        float(zs[-1] - zs[0]) / max(int(res[2]) - 1, 1),
    )
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    dists, _ = KDTree(voxel_pos).query(grid_pts)
    scalar = dists.reshape(int(res[0]), int(res[1]), int(res[2]))
    return scalar, mins, spacing


def _deform_marching_cubes(
    *,
    voxel_pos,
    voxel_q,
    voxel_rots,
    bindings,
    offsets,
    base_rot,
    base_offset,
    n_total,
    mask,
    vox_centers_mesh,
    split_indices=None,
    **_,
):
    finite_mask = np.all(np.isfinite(voxel_pos), axis=1)
    if not finite_mask.any():
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)
    voxel_pos = voxel_pos[finite_mask]
    voxel_radius = float(np.abs(offsets[mask]).max()) * 1.2 if mask.any() else 0.5
    scalar, mins, spacing = _build_scalar_grid(voxel_pos, voxel_radius)
    if scalar.min() >= voxel_radius:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)
    verts, faces, _, _ = marching_cubes(scalar, level=voxel_radius, spacing=spacing)
    verts = (verts + mins).astype(np.float32)
    faces = faces.astype(np.uint32)
    return verts, faces


DEFORM_METHODS = {
    "rigid": _deform_rigid,
    "marching_cubes": _deform_marching_cubes,
}
