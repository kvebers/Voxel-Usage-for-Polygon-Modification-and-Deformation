import numpy as np

from src.draw_helpers import batch_quat_to_mat3
from scipy.spatial import KDTree
from skimage.measure import marching_cubes

_DEFAULT_BASE_ROT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
_DEFAULT_BASE_OFFSET = np.array([0.0, 0.0, 1.5], dtype=np.float32)


def _compute_normals(pos, idx_tris, n_verts):
    if len(idx_tris) == 0:
        return np.zeros((n_verts, 3), dtype=np.float32)
    v0, v1, v2 = pos[idx_tris[:, 0]], pos[idx_tris[:, 1]], pos[idx_tris[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)  # (n_tris, 3)
    fn_rep = np.repeat(fn, 3, axis=0)  # each face normal repeated for its 3 verts
    flat = idx_tris.reshape(-1)  # (3*n_tris,)
    nx = np.bincount(flat, weights=fn_rep[:, 0], minlength=n_verts)
    ny = np.bincount(flat, weights=fn_rep[:, 1], minlength=n_verts)
    nz = np.bincount(flat, weights=fn_rep[:, 2], minlength=n_verts)
    normals = np.stack([nx, ny, nz], axis=1).astype(np.float32)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    return normals


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

    if (
        blend_indices is not None
        and blend_weights is not None
        and split_mesh_verts is not None
    ):
        bi = blend_indices[mask]
        bw = blend_weights[mask].copy()
        sv = split_mesh_verts[mask]

        if reach_mask is not None:
            bw *= reach_mask.astype(np.float32)

        bw /= np.maximum(bw.sum(axis=1, keepdims=True), 1e-8)

        blended = np.zeros((int(mask.sum()), 3), dtype=np.float32)
        for k in range(bi.shape[1]):
            vi = bi[:, k]
            w = bw[:, k, None]
            off_k = sv - vox_centers_mesh[vi]
            woff_k = (base_rot @ off_k.T).T
            p_k = voxel_pos[vi] + np.einsum("nij,nj->ni", voxel_rots[vi], woff_k)
            blended += w * p_k
        positions[mask] = blended
    else:
        world_offsets = (base_rot @ offsets[mask].T).T
        positions[mask] = voxel_pos[bv] + np.einsum(
            "nij,nj->ni", voxel_rots[bv], world_offsets
        )

    if np.any(~mask):
        positions[~mask] = base_offset
    return positions


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
    tree = KDTree(voxel_pos)
    dists, _ = tree.query(grid_pts)
    scalar = dists.reshape(int(res[0]), int(res[1]), int(res[2]))

    level = voxel_radius
    if scalar.min() >= level:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)

    verts, faces, _, _ = marching_cubes(scalar, level=level, spacing=spacing)
    verts = (verts + mins).astype(np.float32)
    faces = faces.astype(np.uint32)
    return verts, faces


DEFORM_METHODS = {
    "rigid": _deform_rigid,
    "marching_cubes": _deform_marching_cubes,
}


class MeshSplitter:
    def __init__(
        self,
        indices,
        bindings,
        offsets,
        n_voxels,
        neighbor_pairs,
        positions,
        voxel_half,
        mesh_verts,
        cfg=None,
    ):
        ms_cfg = getattr(cfg, "mesh_splitter", None) if cfg is not None else None
        if ms_cfg is not None:
            self.BASE_ROT = np.array(ms_cfg.base_rot, dtype=np.float32)
            self.BASE_OFFSET = np.array(ms_cfg.base_offset, dtype=np.float32)
            method_name = getattr(ms_cfg, "deform_method", "rigid")
        else:
            self.BASE_ROT = _DEFAULT_BASE_ROT
            self.BASE_OFFSET = _DEFAULT_BASE_OFFSET
            method_name = "rigid"
        if method_name not in DEFORM_METHODS:
            raise ValueError(f"Incorrect Method")
        self._deform_fn = DEFORM_METHODS[method_name]
        self.n_voxels = n_voxels
        self._adj_list = [[] for _ in range(n_voxels)]
        self.joint_map = {}
        self._joint_pairs = list(neighbor_pairs)
        for ji, (a, b) in enumerate(neighbor_pairs):
            self._adj_list[a].append((b, ji))
            self._adj_list[b].append((a, ji))
            self.joint_map[(min(a, b), max(a, b))] = ji
        self.broken = np.zeros(len(neighbor_pairs), dtype=bool)
        self._reach_mask = None
        self._broken_dirty = False
        positions_arr = np.array(positions, dtype=np.float32)
        vox_centers_shifted = positions_arr @ self.BASE_ROT
        orig_bindings = np.array(bindings, dtype=np.int32)
        orig_offsets = np.array(offsets, dtype=np.float32).reshape(-1, 3)
        vox_model = np.zeros((n_voxels, 3), dtype=np.float32)
        vox_counts = np.zeros(n_voxels, dtype=np.int32)
        bound_mask = orig_bindings >= 0
        valid_vis = np.where(bound_mask)[0]
        if len(valid_vis) > 0:
            vox_ids = orig_bindings[valid_vis]
            np.add.at(
                vox_model, vox_ids, mesh_verts[valid_vis] - orig_offsets[valid_vis]
            )
            np.add.at(vox_counts, vox_ids, 1)
        valid = vox_counts > 0
        vox_model[valid] /= vox_counts[valid, None]
        if valid.any():
            mesh_offset = (vox_centers_shifted[valid] - vox_model[valid]).mean(axis=0)
        else:
            mesh_offset = np.zeros(3, dtype=np.float32)
        vox_centers_mesh = vox_centers_shifted - mesh_offset
        vox_centers_mesh[valid] = vox_model[valid]
        self.vox_centers_mesh = vox_centers_mesh
        vox_tree = KDTree(vox_centers_mesh)
        orig_indices = indices.reshape(-1, 3)
        v0s = mesh_verts[orig_indices[:, 0]]
        v1s = mesh_verts[orig_indices[:, 1]]
        v2s = mesh_verts[orig_indices[:, 2]]
        n_tris = len(orig_indices)

        if n_tris > 0:
            centroids = (v0s + v1s + v2s) / 3.0
            _, q = vox_tree.query(centroids)
            owners_1 = q.astype(np.int32)
            owners_3 = np.repeat(owners_1, 3)
            verts_all = np.stack([v0s, v1s, v2s], axis=1).reshape(-1, 3)
            off_1 = verts_all - vox_centers_mesh[owners_3]
            bases = np.arange(n_tris, dtype=np.uint32) * 3
            idx_1 = (bases[:, None] + np.array([0, 1, 2], dtype=np.uint32)).ravel()
        else:
            owners_3 = np.empty(0, dtype=np.int32)
            off_1 = np.empty((0, 3), dtype=np.float32)
            idx_1 = np.empty(0, dtype=np.uint32)

        bind_arr, off_arr, idx_arr = owners_3, off_1, idx_1

        self.split_bindings = bind_arr.astype(np.int32)
        self.split_offsets = off_arr.astype(np.float32).reshape(-1, 3)
        self.split_indices = idx_arr.astype(np.uint32)
        self.n_split_verts = len(self.split_bindings)
        self._vox_to_verts = [[] for _ in range(n_voxels)]
        valid_vi = np.where(self.split_bindings >= 0)[0]
        if len(valid_vi) > 0:
            bv_arr = self.split_bindings[valid_vi]
            order = np.argsort(bv_arr, kind="stable")
            sorted_vis = valid_vi[order]
            sorted_bvs = bv_arr[order]
            splits = np.searchsorted(sorted_bvs, np.arange(n_voxels + 1))
            for bv in range(n_voxels):
                self._vox_to_verts[bv] = sorted_vis[
                    splits[bv] : splits[bv + 1]
                ].tolist()
        self.split_mesh_verts = (
            self.split_offsets + vox_centers_mesh[self.split_bindings]
        )
        K = min(4, n_voxels)
        lbs_dists, lbs_idx = vox_tree.query(self.split_mesh_verts, k=K)
        lbs_dists = lbs_dists.reshape(-1, K)
        lbs_idx = lbs_idx.reshape(-1, K)
        primary_in_set = (lbs_idx == self.split_bindings[:, None]).any(axis=1)
        missing = ~primary_in_set
        if missing.any():
            lbs_idx[missing, -1] = self.split_bindings[missing]
            mv = self.split_mesh_verts[missing]
            pv = vox_centers_mesh[self.split_bindings[missing]]
            lbs_dists[missing, -1] = np.linalg.norm(mv - pv, axis=1).astype(np.float32)

        sigma = 2.0 * voxel_half
        lbs_w = np.exp(-(lbs_dists**2) / (2.0 * sigma**2)).astype(np.float32)
        lbs_w /= np.maximum(lbs_w.sum(axis=1, keepdims=True), 1e-8)
        self.blend_indices = lbs_idx.astype(np.int32)
        self.blend_weights = lbs_w

        self._cached_out = None
        self._cached_idx = None
        self._last_voxel_slice = None
        self.gpu_dirty = True
        self.local_offsets = (self.BASE_ROT @ self.split_offsets.T).T
        _mask = self.split_bindings >= 0
        _bv = self.split_bindings
        _init_pos = np.empty((self.n_split_verts, 3), dtype=np.float32)
        _init_pos[_mask] = positions_arr[_bv[_mask]] + self.local_offsets[_mask]
        if np.any(~_mask):
            _init_pos[~_mask] = self.BASE_OFFSET
        self.rest_normals = _compute_normals(
            _init_pos, self.split_indices.reshape(-1, 3), self.n_split_verts
        )

        self.current_index_count = len(self.split_indices)

        self._update_reach_mask()

    def set_broken(self, broken):
        if not np.array_equal(broken, self.broken):
            newly_broken = np.where(broken & ~self.broken)[0]
            self.broken = broken.copy()
            if self._reach_mask is None or not newly_broken.size:
                self._update_reach_mask()
            else:
                self._update_reach_mask_incremental(newly_broken)
            self._broken_dirty = True

    def _bfs_voxels(self, starts, depth=3):
        reachable = set(starts)
        frontier = list(starts)
        for _ in range(depth):
            nxt = []
            for v in frontier:
                for nb, ji in self._adj_list[v]:
                    if not self.broken[ji] and nb not in reachable:
                        reachable.add(nb)
                        nxt.append(nb)
            frontier = nxt
            if not frontier:
                break
        return reachable

    def _affected_voxels(self, newly_broken_joints, depth=3):
        seeds = set()
        for ji in newly_broken_joints:
            a, b = self._joint_pairs[ji]
            seeds.add(a)
            seeds.add(b)
        frontier = list(seeds)
        visited = set(seeds)
        for _ in range(depth):
            nxt = []
            for v in frontier:
                for nb, _ in self._adj_list[v]:
                    if nb not in visited:
                        visited.add(nb)
                        nxt.append(nb)
            frontier = nxt
            if not frontier:
                break
        return visited

    def _apply_reach_mask_for_voxels(self, voxels):
        for bv in voxels:
            verts = self._vox_to_verts[bv]
            if not verts:
                continue
            reachable = self._bfs_voxels([bv])
            vi_arr = np.array(verts, dtype=np.int32)
            reachable_arr = np.fromiter(reachable, dtype=np.int32)
            self._reach_mask[vi_arr] = np.isin(
                self.blend_indices[vi_arr], reachable_arr
            )

    def _update_reach_mask(self):
        K = self.blend_indices.shape[1]

        if not np.any(self.broken):
            self._reach_mask = np.ones((self.n_split_verts, K), dtype=bool)
            return

        self._reach_mask = np.ones((self.n_split_verts, K), dtype=bool)
        self._apply_reach_mask_for_voxels(range(self.n_voxels))

    def _update_reach_mask_incremental(self, newly_broken_joints):
        affected = self._affected_voxels(newly_broken_joints)
        self._apply_reach_mask_for_voxels(affected)

    def deform_split_mesh(self, transforms, voxel_body_start, n_voxels):
        self.last_voxel_slice = transforms[
            voxel_body_start : voxel_body_start + n_voxels
        ]
        return None, None
