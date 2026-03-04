import ctypes
import numpy as np
from OpenGL.GL import *
from src.draw_helpers import *
from src.shaders import *

_LIGHT_DIR = np.array([-1.0, -0.5, 1.0], dtype=np.float32)
_LIGHT_DIR /= np.linalg.norm(_LIGHT_DIR)


def _cache_locs(prog, names):
    return {n: glGetUniformLocation(prog, n) for n in names}


def _create_tbo(size_bytes, fmt=GL_RGB32F):
    buf = glGenBuffers(1)
    glBindBuffer(GL_TEXTURE_BUFFER, buf)
    glBufferData(GL_TEXTURE_BUFFER, size_bytes, None, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_TEXTURE_BUFFER, 0)
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_BUFFER, tex)
    glTexBuffer(GL_TEXTURE_BUFFER, fmt, buf)
    glBindTexture(GL_TEXTURE_BUFFER, 0)
    return buf, tex


class ObjectRenderer:
    def __init__(self, renderer, mesh_indices, voxel_count, half, block_halves=None):
        self._r = renderer
        self.voxel_count = voxel_count
        self.half = half
        self.block_halves = (
            np.asarray(block_halves, dtype=np.float32)
            if block_halves is not None
            else None
        )

        self.mesh_vao = create_vao()
        max_verts = int(mesh_indices.max()) + 1 if len(mesh_indices) > 0 else 1
        self._mesh_vbo_capacity = max_verts * 3 * 6 * 4
        self.mesh_vbo = create_buffer(
            size=self._mesh_vbo_capacity, usage=GL_DYNAMIC_DRAW
        )

        self.mesh_ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.mesh_ebo)
        idx_data = mesh_indices.astype(np.uint32)
        self._mesh_ebo_capacity = max(len(idx_data) * 3 * 4, idx_data.nbytes)
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER, self._mesh_ebo_capacity, None, GL_DYNAMIC_DRAW
        )
        glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, idx_data.nbytes, idx_data)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        setup_mesh_vao(self.mesh_vao, self.mesh_vbo, self.mesh_ebo)
        self.mesh_index_count = len(idx_data)

        self.cube_inst_vbo = create_buffer(size=voxel_count * 64, usage=GL_DYNAMIC_DRAW)
        self.cube_color_vbo = create_buffer(
            size=voxel_count * 12, usage=GL_DYNAMIC_DRAW
        )
        self.voxel_vao = create_vao()
        setup_instanced_color_vao(
            self.voxel_vao,
            renderer.cube_geom_vbo,
            self.cube_inst_vbo,
            self.cube_color_vbo,
        )
        self._voxel_pos_buf = self._voxel_pos_tex = None
        self._voxel_quat_buf = self._voxel_quat_tex = None
        self._last_valid_pos = None
        self._last_valid_quat = None

    def setup_gpu_deform(self, mesh_splitter):
        n_voxels = self.voxel_count

        idx = mesh_splitter.split_indices.astype(np.uint32)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.mesh_ebo)
        if idx.nbytes > self._mesh_ebo_capacity:
            self._mesh_ebo_capacity = idx.nbytes
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)
        else:
            glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, idx.nbytes, idx)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        self.mesh_index_count = len(idx)

        static = np.empty((mesh_splitter.n_split_verts, 6), dtype=np.float32)
        static[:, :3] = mesh_splitter.local_offsets
        static[:, 3:] = mesh_splitter.rest_normals
        glBindBuffer(GL_ARRAY_BUFFER, self.mesh_vbo)
        if static.nbytes > self._mesh_vbo_capacity:
            self._mesh_vbo_capacity = static.nbytes
            glBufferData(GL_ARRAY_BUFFER, static.nbytes, static, GL_STATIC_DRAW)
        else:
            glBufferSubData(GL_ARRAY_BUFFER, 0, static.nbytes, static)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        binding_data = mesh_splitter.split_bindings.astype(np.float32)
        self.mesh_binding_vbo = create_buffer(binding_data, usage=GL_STATIC_DRAW)

        self._voxel_pos_buf, self._voxel_pos_tex = _create_tbo(n_voxels * 12, GL_RGB32F)
        self._voxel_quat_buf, self._voxel_quat_tex = _create_tbo(
            n_voxels * 16, GL_RGBA32F
        )

        setup_mesh_vao_gpu(
            self.mesh_vao, self.mesh_vbo, self.mesh_binding_vbo, self.mesh_ebo
        )

        glUseProgram(self._r.prog_mesh)
        glUniform1i(glGetUniformLocation(self._r.prog_mesh, "uVoxelPos"), 0)
        glUniform1i(glGetUniformLocation(self._r.prog_mesh, "uVoxelQuat"), 1)
        glUseProgram(0)

    def update_voxel_transforms(self, voxel_slice):
        pos = np.ascontiguousarray(voxel_slice[:, :3], dtype=np.float32)
        quat = np.ascontiguousarray(voxel_slice[:, 3:7], dtype=np.float32)

        nan_mask = ~np.isfinite(pos).all(axis=1)
        if nan_mask.any():
            if self._last_valid_pos is not None:
                pos[nan_mask] = self._last_valid_pos[nan_mask]
                quat[nan_mask] = self._last_valid_quat[nan_mask]
        valid_mask = ~nan_mask
        if self._last_valid_pos is None:
            self._last_valid_pos = pos.copy()
            self._last_valid_quat = quat.copy()
        elif valid_mask.any():
            self._last_valid_pos[valid_mask] = pos[valid_mask]
            self._last_valid_quat[valid_mask] = quat[valid_mask]

        glBindBuffer(GL_TEXTURE_BUFFER, self._voxel_pos_buf)
        glBufferSubData(GL_TEXTURE_BUFFER, 0, pos.nbytes, pos)
        glBindBuffer(GL_TEXTURE_BUFFER, 0)
        glBindBuffer(GL_TEXTURE_BUFFER, self._voxel_quat_buf)
        glBufferSubData(GL_TEXTURE_BUFFER, 0, quat.nbytes, quat)
        glBindBuffer(GL_TEXTURE_BUFFER, 0)

    def update_voxel_instances(self, transforms, voxel_body_start):
        bodies = transforms[voxel_body_start : voxel_body_start + self.voxel_count]
        pos = bodies[:, :3]
        rots = batch_quat_to_mat3(bodies[:, 3:7])
        matrices = np.zeros((self.voxel_count, 4, 4), dtype=np.float32)
        if self.block_halves is not None:
            # Per-block non-uniform scale: multiply each column j of rot by h[j]
            matrices[:, :3, :3] = rots * self.block_halves[:, np.newaxis, :]
        else:
            matrices[:, :3, :3] = rots * self.half
        matrices[:, :3, 3] = pos
        matrices[:, 3, 3] = 1.0
        matrices = matrices.transpose(0, 2, 1)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube_inst_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, matrices.nbytes, matrices)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def update_voxel_colors(self, colors):
        glBindBuffer(GL_ARRAY_BUFFER, self.cube_color_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, colors.nbytes, colors)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw_mesh_mode(self, proj, view, cam_pos, color, index_count=None):
        r = self._r
        locs = r._uloc_mesh
        glUseProgram(r.prog_mesh)
        glUniformMatrix4fv(locs["uProj"], 1, GL_TRUE, proj)
        glUniformMatrix4fv(locs["uView"], 1, GL_TRUE, view)
        glUniform3fv(locs["uColor"], 1, np.asarray(color, dtype=np.float32))
        glUniform3fv(locs["uCamPos"], 1, np.asarray(cam_pos, dtype=np.float32))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_BUFFER, self._voxel_pos_tex)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_BUFFER, self._voxel_quat_tex)
        glBindVertexArray(self.mesh_vao)
        count = index_count if index_count is not None else self.mesh_index_count
        glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def draw_voxels(self, proj, view, cam_pos):
        r = self._r
        locs = r._uloc_inst_color
        glUseProgram(r.prog_inst_color)
        glUniformMatrix4fv(locs["uProj"], 1, GL_TRUE, proj)
        glUniformMatrix4fv(locs["uView"], 1, GL_TRUE, view)
        glUniform3fv(locs["uCamPos"], 1, np.asarray(cam_pos, dtype=np.float32))
        glBindVertexArray(self.voxel_vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, r.cube_vert_count, self.voxel_count)
        glBindVertexArray(0)


class Renderer:
    def __init__(self, ball_radius, max_balls=1):
        self.ball_radius = ball_radius
        self.prog_mesh = compile_shader_program(VERT_MESH, FRAG_BLINN)
        self.prog_inst = compile_shader_program(VERT_INSTANCED, FRAG_BLINN)
        self.prog_inst_color = compile_shader_program(
            VERT_INSTANCED_COLOR, FRAG_BLINN_INSTANCED
        )
        self.prog_ground = compile_shader_program(VERT_GROUND, FRAG_GROUND)

        self._uloc_mesh = _cache_locs(
            self.prog_mesh,
            ["uProj", "uView", "uColor", "uLightDir", "uCamPos", "uAmbient"],
        )
        self._uloc_inst = _cache_locs(
            self.prog_inst,
            ["uProj", "uView", "uColor", "uLightDir", "uCamPos", "uAmbient"],
        )
        self._uloc_inst_color = _cache_locs(
            self.prog_inst_color, ["uProj", "uView", "uLightDir", "uCamPos", "uAmbient"]
        )
        self._uloc_ground = _cache_locs(self.prog_ground, ["uProj", "uView"])

        glUseProgram(self.prog_mesh)
        glUniform3fv(self._uloc_mesh["uLightDir"], 1, _LIGHT_DIR)
        glUniform1f(self._uloc_mesh["uAmbient"], 0.4)
        glUseProgram(self.prog_inst)
        glUniform3fv(self._uloc_inst["uLightDir"], 1, _LIGHT_DIR)
        glUniform1f(self._uloc_inst["uAmbient"], 0.4)
        glUseProgram(self.prog_inst_color)
        glUniform3fv(self._uloc_inst_color["uLightDir"], 1, _LIGHT_DIR)
        glUniform1f(self._uloc_inst_color["uAmbient"], 0.4)
        glUseProgram(0)
        ground_data, self.ground_vert_count = make_ground_quad()
        self.ground_vao = create_vao()
        self.ground_vbo = create_buffer(ground_data)
        setup_ground_vao(self.ground_vao, self.ground_vbo)
        cube_data, self.cube_vert_count = make_unit_cube()
        self.cube_geom_vbo = create_buffer(cube_data)
        sphere_data, self.sphere_vert_count = make_uv_sphere(20, 14)
        self.sphere_geom_vbo = create_buffer(sphere_data)
        self._max_balls = max(max_balls, 1)
        self.sphere_inst_vbo = create_buffer(
            size=64 * self._max_balls, usage=GL_DYNAMIC_DRAW
        )
        self.ball_vao = create_vao()
        setup_instanced_vao(self.ball_vao, self.sphere_geom_vbo, self.sphere_inst_vbo)

    def create_object_renderer(
        self, mesh_indices, voxel_count, half, block_halves=None
    ):
        return ObjectRenderer(
            self, mesh_indices, voxel_count, half, block_halves=block_halves
        )

    def update_ball_instances(self, transforms, ball_bodies, radius):
        if not ball_bodies:
            return
        matrices = np.empty((len(ball_bodies), 4, 4), dtype=np.float32)
        for i, body in enumerate(ball_bodies):
            m = quat_to_mat4(transforms[body][3:7], transforms[body][:3])
            m[:3, :3] *= radius
            matrices[i] = m.T
        data = np.ascontiguousarray(matrices)
        glBindBuffer(GL_ARRAY_BUFFER, self.sphere_inst_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw_ground(self, proj, view):
        locs = self._uloc_ground
        glUseProgram(self.prog_ground)
        glUniformMatrix4fv(locs["uProj"], 1, GL_TRUE, proj)
        glUniformMatrix4fv(locs["uView"], 1, GL_TRUE, view)
        glBindVertexArray(self.ground_vao)
        glDrawArrays(GL_TRIANGLES, 0, self.ground_vert_count)
        glBindVertexArray(0)

    def draw_ball(self, proj, view, cam_pos, color, count=1):
        if count == 0:
            return
        locs = self._uloc_inst
        glUseProgram(self.prog_inst)
        glUniformMatrix4fv(locs["uProj"], 1, GL_TRUE, proj)
        glUniformMatrix4fv(locs["uView"], 1, GL_TRUE, view)
        glUniform3fv(locs["uColor"], 1, np.asarray(color, dtype=np.float32))
        glUniform3fv(locs["uCamPos"], 1, np.asarray(cam_pos, dtype=np.float32))
        glBindVertexArray(self.ball_vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, self.sphere_vert_count, count)
        glBindVertexArray(0)
