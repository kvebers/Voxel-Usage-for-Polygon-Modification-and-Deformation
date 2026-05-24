import numpy as np


def compute_normals(pos, index_tris, n_verts):
    if len(index_tris) == 0:
        return np.zeros((n_verts, 3), dtype=np.float32)
    v0, v1, v2 = (
        pos[index_tris[:, 0]],
        pos[index_tris[:, 1]],
        pos[index_tris[:, 2]],
    )
    fn = np.cross(v1 - v0, v2 - v0)
    fn_rep = np.repeat(fn, 3, axis=0)
    flat = index_tris.reshape(-1)
    nx = np.bincount(flat, weights=fn_rep[:, 0], minlength=n_verts)
    ny = np.bincount(flat, weights=fn_rep[:, 1], minlength=n_verts)
    nz = np.bincount(flat, weights=fn_rep[:, 2], minlength=n_verts)
    normals = np.stack([nx, ny, nz], axis=1).astype(np.float32)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    return normals


def compute_model_space_centers(mesh_verts, orig_bindings, orig_offsets, n_voxels):
    vox_model = np.zeros((n_voxels, 3), dtype=np.float32)
    vox_counts = np.zeros(n_voxels, dtype=np.int32)
    valid_vis = np.where(orig_bindings >= 0)[0]
    if len(valid_vis) > 0:
        vox_ids = orig_bindings[valid_vis]
        np.add.at(vox_model, vox_ids, mesh_verts[valid_vis] - orig_offsets[valid_vis])
        np.add.at(vox_counts, vox_ids, 1)
    valid = vox_counts > 0
    vox_model[valid] /= vox_counts[valid, None]
    return vox_model, valid


def build_vox_to_verts_index(n_voxels, split_bindings):
    vox_to_verts = [[] for _ in range(n_voxels)]
    valid_vi = np.where(split_bindings >= 0)[0]
    if len(valid_vi) > 0:
        bv_arr = split_bindings[valid_vi]
        order = np.argsort(bv_arr, kind="stable")
        sorted_vis = valid_vi[order]
        sorted_bvs = bv_arr[order]
        splits = np.searchsorted(sorted_bvs, np.arange(n_voxels + 1))
        for bv in range(n_voxels):
            vox_to_verts[bv] = sorted_vis[splits[bv] : splits[bv + 1]].tolist()
    return vox_to_verts
