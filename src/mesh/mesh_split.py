import numpy as np
from scipy.spatial import KDTree
from src.constants import DEFAULT_ROTATION_ELEMENT
from src.mesh.mesh_helpers import (
    compute_normals,
    compute_model_space_centers,
    build_vox_to_verts_index,
    compute_normals,
    compute_model_space_centers,
    build_vox_to_verts_index,
)
from src.utils.math_utils import batch_quat_to_mat3


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
    ):
        self.BASE_ROT = DEFAULT_ROTATION_ELEMENT
        self.build_adjacency(n_voxels, neighbor_pairs)
        vox_tree, positions_arr = self.compute_voxel_centers(positions, mesh_verts, bindings, offsets, n_voxels)
        self.assign_vertices_to_voxels(indices, mesh_verts, n_voxels, vox_tree)
        self.compute_blend_weights(vox_tree, voxel_half, n_voxels)
        self.initialize_rest_geometry(positions_arr)

    def build_adjacency(self, n_voxels, neighbor_pairs):
        self.n_voxels = n_voxels
        self._adj_list = [[] for _ in range(n_voxels)]
        self.joint_map = {}
        self._joint_pairs = list(neighbor_pairs)
        for joint_index, (voxel_a, voxel_b) in enumerate(neighbor_pairs):
            self._adj_list[voxel_a].append((voxel_b, joint_index))
            self._adj_list[voxel_b].append((voxel_a, joint_index))
            self.joint_map[(min(voxel_a, voxel_b), max(voxel_a, voxel_b))] = joint_index
        self.broken = np.zeros(len(neighbor_pairs), dtype=bool)
        self._reach_mask = None
        self._broken_dirty = False

    def compute_voxel_centers(self, positions, mesh_verts, bindings, offsets, n_voxels):
        positions_arr = np.array(positions, dtype=np.float32)
        vox_centers_shifted = positions_arr @ self.BASE_ROT
        orig_bindings = np.array(bindings, dtype=np.int32)
        orig_offsets = np.array(offsets, dtype=np.float32).reshape(-1, 3)
        vox_model, valid = compute_model_space_centers(mesh_verts, orig_bindings, orig_offsets, n_voxels)
        if valid.any():
            mesh_offset = (vox_centers_shifted[valid] - vox_model[valid]).mean(axis=0)
        else:
            mesh_offset = np.zeros(3, dtype=np.float32)
        vox_centers_mesh = vox_centers_shifted - mesh_offset
        vox_centers_mesh[valid] = vox_model[valid]
        self.vox_centers_mesh = vox_centers_mesh
        return KDTree(vox_centers_mesh), positions_arr

    def assign_vertices_to_voxels(self, indices, mesh_verts, n_voxels, vox_tree):
        vox_centers_mesh = self.vox_centers_mesh
        orig_indices = indices.reshape(-1, 3)
        v0s = mesh_verts[orig_indices[:, 0]]
        v1s = mesh_verts[orig_indices[:, 1]]
        v2s = mesh_verts[orig_indices[:, 2]]
        n_tris = len(orig_indices)

        if n_tris > 0:
            centroids = (v0s + v1s + v2s) / 3.0
            _, nearest_voxel_index = vox_tree.query(centroids)
            tri_owners = nearest_voxel_index.astype(np.int32)
            vert_owners = np.repeat(tri_owners, 3)
            verts_all = np.stack([v0s, v1s, v2s], axis=1).reshape(-1, 3)
            vert_offsets = verts_all - vox_centers_mesh[vert_owners]
            tri_base_indices = np.arange(n_tris, dtype=np.uint32) * 3
            flat_indices = (tri_base_indices[:, None] + np.array([0, 1, 2], dtype=np.uint32)).ravel()
        else:
            vert_owners = np.empty(0, dtype=np.int32)
            vert_offsets = np.empty((0, 3), dtype=np.float32)
            flat_indices = np.empty(0, dtype=np.uint32)

        self.split_bindings = vert_owners.astype(np.int32)
        self.split_offsets = vert_offsets.astype(np.float32).reshape(-1, 3)
        self.split_indices = flat_indices.astype(np.uint32)
        self.n_split_verts = len(self.split_bindings)
        self._vox_to_verts = build_vox_to_verts_index(n_voxels, self.split_bindings)
        self.split_mesh_verts = self.split_offsets + vox_centers_mesh[self.split_bindings]

    def ensure_primary_in_candidates(self, lbs_index, lbs_dists):
        primary_in_set = (lbs_index == self.split_bindings[:, None]).any(axis=1)
        missing = ~primary_in_set
        if missing.any():
            lbs_index[missing, -1] = self.split_bindings[missing]
            mv = self.split_mesh_verts[missing]
            pv = self.vox_centers_mesh[self.split_bindings[missing]]
            lbs_dists[missing, -1] = np.linalg.norm(mv - pv, axis=1).astype(np.float32)

    def compute_blend_weights(self, vox_tree, voxel_half, n_voxels):
        k_neighbors = min(4, n_voxels)
        lbs_dists, lbs_index = vox_tree.query(self.split_mesh_verts, k=k_neighbors)
        lbs_dists = lbs_dists.reshape(-1, k_neighbors)
        lbs_index = lbs_index.reshape(-1, k_neighbors)
        self.ensure_primary_in_candidates(lbs_index, lbs_dists)
        sigma = 2.0 * voxel_half
        lbs_w = np.exp(-(lbs_dists**2) / (2.0 * sigma**2)).astype(np.float32)
        lbs_w /= np.maximum(lbs_w.sum(axis=1, keepdims=True), 1e-8)
        self.blend_indices = lbs_index.astype(np.int32)
        self.blend_weights = lbs_w

    def compute_rest_positions(self, positions_arr):
        mask = self.split_bindings >= 0
        voxel_bindings = self.split_bindings
        init_pos = np.empty((self.n_split_verts, 3), dtype=np.float32)
        init_pos[mask] = positions_arr[voxel_bindings[mask]] + self.local_offsets[mask]
        return init_pos

    def initialize_rest_geometry(self, positions_arr):
        self._cached_out = None
        self._cached_index = None
        self._last_voxel_slice = None
        self.gpu_dirty = True
        self.local_offsets = (self.BASE_ROT @ self.split_offsets.T).T
        init_pos = self.compute_rest_positions(positions_arr)
        self.rest_normals = compute_normals(init_pos, self.split_indices.reshape(-1, 3), self.n_split_verts)
        self.current_index_count = len(self.split_indices)
        self.update_reach_mask()

    def set_broken(self, broken):
        if not np.array_equal(broken, self.broken):
            newly_broken = np.where(broken & ~self.broken)[0]
            self.broken = broken.copy()
            if self._reach_mask is None or not newly_broken.size:
                self.update_reach_mask()
            else:
                self.update_reach_mask_incremental(newly_broken)
            self._broken_dirty = True

    def bfs_voxels(self, starts, depth=3):
        reachable = set(starts)
        frontier = list(starts)
        for _ in range(depth):
            next_frontier = []
            for voxel in frontier:
                for neighbor_voxel, joint_index in self._adj_list[voxel]:
                    if not self.broken[joint_index] and neighbor_voxel not in reachable:
                        reachable.add(neighbor_voxel)
                        next_frontier.append(neighbor_voxel)
            frontier = next_frontier
            if not frontier:
                break
        return reachable

    def affected_voxels(self, newly_broken_joints, depth=3):
        seeds = set()
        for joint_index in newly_broken_joints:
            voxel_a, voxel_b = self._joint_pairs[joint_index]
            seeds.add(voxel_a)
            seeds.add(voxel_b)
        frontier = list(seeds)
        visited = set(seeds)
        for _ in range(depth):
            next_frontier = []
            for voxel in frontier:
                for neighbor_voxel, _ in self._adj_list[voxel]:
                    if neighbor_voxel not in visited:
                        visited.add(neighbor_voxel)
                        next_frontier.append(neighbor_voxel)
            frontier = next_frontier
            if not frontier:
                break
        return visited

    def apply_reach_mask_for_voxels(self, voxels):
        for voxel_index in voxels:
            verts = self._vox_to_verts[voxel_index]
            if not verts:
                continue
            reachable = self.bfs_voxels([voxel_index])
            vi_arr = np.array(verts, dtype=np.int32)
            reachable_arr = np.fromiter(reachable, dtype=np.int32)
            self._reach_mask[vi_arr] = np.isin(self.blend_indices[vi_arr], reachable_arr)

    def update_reach_mask(self):
        k_neighbors = self.blend_indices.shape[1]
        if not np.any(self.broken):
            self._reach_mask = np.ones((self.n_split_verts, k_neighbors), dtype=bool)
            return
        self._reach_mask = np.ones((self.n_split_verts, k_neighbors), dtype=bool)
        self.apply_reach_mask_for_voxels(range(self.n_voxels))

    def update_reach_mask_incremental(self, newly_broken_joints):
        affected = self.affected_voxels(newly_broken_joints)
        self.apply_reach_mask_for_voxels(affected)

    def deform_split_mesh(self, transforms, voxel_body_start, n_voxels):
        self.last_voxel_slice = transforms[voxel_body_start : voxel_body_start + n_voxels]
        return None, None
