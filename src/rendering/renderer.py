import ctypes
import numpy as np
from OpenGL.GL import *
from src.utils.math_utils import perspective_matrix, look_at_matrix, quat_to_mat3, quat_to_mat4, batch_quat_to_mat3
from src.rendering.gl_vao import setup_mesh_vao, setup_mesh_vao_gpu, setup_instanced_vao, setup_instanced_color_vao, setup_ground_vao
from src.rendering.gl_geometry import (
    compile_shader_program,
    make_unit_cube,
    make_uv_sphere,
    make_ground_quad,
    compute_vertex_normals,
    create_vao,
    create_buffer,
)
from src.rendering.shaders import *
from src.constants import LIGHT_DIR


def cache_locs(prog, names):
    return {name: glGetUniformLocation(prog, name) for name in names}


def create_tbo(size_bytes, fmt=GL_RGB32F):
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
    def __init__(self, renderer, mesh_indices, voxel_count, voxel_half_extent, block_halves=None):
        self._renderer = renderer
        self.voxel_count = voxel_count
        self.voxel_half_extent = voxel_half_extent
        self.block_halves = np.asarray(block_halves, dtype=np.float32) if block_halves is not None else None
        self.setup_mesh_buffers(mesh_indices)
        self.setup_voxel_buffers(voxel_count, self._renderer)

    def setup_mesh_buffers(self, mesh_indices):
        self.mesh_vao = create_vao()
        max_verts = int(mesh_indices.max()) + 1 if len(mesh_indices) > 0 else 1
        self._mesh_vbo_capacity = max_verts * 3 * 6 * 4
        self.mesh_vbo = create_buffer(size=self._mesh_vbo_capacity, usage=GL_DYNAMIC_DRAW)
        self.mesh_ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.mesh_ebo)
        index_data = mesh_indices.astype(np.uint32)
        self._mesh_ebo_capacity = max(len(index_data) * 3 * 4, index_data.nbytes)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self._mesh_ebo_capacity, None, GL_DYNAMIC_DRAW)
        glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, index_data.nbytes, index_data)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        setup_mesh_vao(self.mesh_vao, self.mesh_vbo, self.mesh_ebo)
        self.mesh_index_count = len(index_data)

    def setup_voxel_buffers(self, voxel_count, renderer):
        self.cube_inst_vbo = create_buffer(size=voxel_count * 64, usage=GL_DYNAMIC_DRAW)
        self.cube_color_vbo = create_buffer(size=voxel_count * 12, usage=GL_DYNAMIC_DRAW)
        self.voxel_vao = create_vao()
        setup_instanced_color_vao(self.voxel_vao, renderer.cube_geom_vbo, self.cube_inst_vbo, self.cube_color_vbo)
        self._voxel_pos_buf = self._voxel_pos_tex = None
        self._voxel_quat_buf = self._voxel_quat_tex = None
        self._last_valid_pos = None
        self._last_valid_quat = None

    def setup_gpu_deform(self, mesh_splitter):
        voxel_count = self.voxel_count
        split_indices = mesh_splitter.split_indices.astype(np.uint32)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.mesh_ebo)
        if split_indices.nbytes > self._mesh_ebo_capacity:
            self._mesh_ebo_capacity = split_indices.nbytes
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, split_indices.nbytes, split_indices, GL_STATIC_DRAW)
        else:
            glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, split_indices.nbytes, split_indices)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        self.mesh_index_count = len(split_indices)
        static_vertex_data = np.empty((mesh_splitter.n_split_verts, 6), dtype=np.float32)
        static_vertex_data[:, :3] = mesh_splitter.local_offsets
        static_vertex_data[:, 3:] = mesh_splitter.rest_normals
        glBindBuffer(GL_ARRAY_BUFFER, self.mesh_vbo)
        if static_vertex_data.nbytes > self._mesh_vbo_capacity:
            self._mesh_vbo_capacity = static_vertex_data.nbytes
            glBufferData(GL_ARRAY_BUFFER, static_vertex_data.nbytes, static_vertex_data, GL_STATIC_DRAW)
        else:
            glBufferSubData(GL_ARRAY_BUFFER, 0, static_vertex_data.nbytes, static_vertex_data)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        binding_data = mesh_splitter.split_bindings.astype(np.float32)
        self.mesh_binding_vbo = create_buffer(binding_data, usage=GL_STATIC_DRAW)
        self._voxel_pos_buf, self._voxel_pos_tex = create_tbo(voxel_count * 12, GL_RGB32F)
        self._voxel_quat_buf, self._voxel_quat_tex = create_tbo(voxel_count * 16, GL_RGBA32F)
        setup_mesh_vao_gpu(self.mesh_vao, self.mesh_vbo, self.mesh_binding_vbo, self.mesh_ebo)
        glUseProgram(self._renderer.prog_mesh)
        glUniform1i(glGetUniformLocation(self._renderer.prog_mesh, "uVoxelPos"), 0)
        glUniform1i(glGetUniformLocation(self._renderer.prog_mesh, "uVoxelQuat"), 1)
        glUseProgram(0)

    def update_voxel_transforms(self, voxel_slice):
        pos = np.ascontiguousarray(voxel_slice[:, :3])
        quat = np.ascontiguousarray(voxel_slice[:, 3:7])

        invalid_mask = ~np.isfinite(pos).all(axis=1)
        if invalid_mask.any():
            if self._last_valid_pos is not None:
                pos[invalid_mask] = self._last_valid_pos[invalid_mask]
                quat[invalid_mask] = self._last_valid_quat[invalid_mask]
        valid_mask = ~invalid_mask
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
        voxel_transforms = transforms[voxel_body_start : voxel_body_start + self.voxel_count]
        positions = voxel_transforms[:, :3]
        rotation_matrices = batch_quat_to_mat3(voxel_transforms[:, 3:7])
        instance_matrices = np.zeros((self.voxel_count, 4, 4), dtype=np.float32)
        if self.block_halves is not None:
            instance_matrices[:, :3, :3] = rotation_matrices * self.block_halves[:, np.newaxis, :]
        else:
            instance_matrices[:, :3, :3] = rotation_matrices * self.voxel_half_extent
        instance_matrices[:, :3, 3] = positions
        instance_matrices[:, 3, 3] = 1.0
        instance_matrices = instance_matrices.transpose(0, 2, 1)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube_inst_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, instance_matrices.nbytes, instance_matrices)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def update_voxel_colors(self, colors):
        glBindBuffer(GL_ARRAY_BUFFER, self.cube_color_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, colors.nbytes, colors)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw_mesh_mode(self, proj, view, cam_pos, color, index_count=None):
        renderer = self._renderer
        locs = renderer.uloc_mesh
        glUseProgram(renderer.prog_mesh)
        glUniformMatrix4fv(locs["uProj"], 1, GL_TRUE, proj)
        glUniformMatrix4fv(locs["uView"], 1, GL_TRUE, view)
        glUniform3fv(locs["uColor"], 1, np.asarray(color, dtype=np.float32))
        glUniform3fv(locs["uCamPos"], 1, np.asarray(cam_pos))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_BUFFER, self._voxel_pos_tex)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_BUFFER, self._voxel_quat_tex)
        glBindVertexArray(self.mesh_vao)
        draw_count = index_count if index_count is not None else self.mesh_index_count
        glDrawElements(GL_TRIANGLES, draw_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def draw_voxels(self, proj, view, cam_pos):
        renderer = self._renderer
        locs = renderer.uloc_inst_color
        glUseProgram(renderer.prog_inst_color)
        glUniformMatrix4fv(locs["uProj"], 1, GL_TRUE, proj)
        glUniformMatrix4fv(locs["uView"], 1, GL_TRUE, view)
        glUniform3fv(locs["uCamPos"], 1, np.asarray(cam_pos))
        glBindVertexArray(self.voxel_vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, renderer.cube_vert_count, self.voxel_count)
        glBindVertexArray(0)


class Renderer:
    def __init__(self):
        self.compile_programs()
        self.setup_lighting()
        self.setup_ground_geometry()
        self.setup_sphere_geometry()

    def compile_programs(self):
        self.prog_mesh = compile_shader_program(VERT_MESH, FRAG_BLINN)
        self.prog_inst = compile_shader_program(VERT_INSTANCED, FRAG_BLINN)
        self.prog_inst_color = compile_shader_program(VERT_INSTANCED_COLOR, FRAG_BLINN_INSTANCED)
        self.prog_ground = compile_shader_program(VERT_GROUND, FRAG_GROUND)
        self.uloc_mesh = cache_locs(self.prog_mesh, ["uProj", "uView", "uColor", "uLightDir", "uCamPos", "uAmbient"])
        self.uloc_inst = cache_locs(self.prog_inst, ["uProj", "uView", "uColor", "uLightDir", "uCamPos", "uAmbient"])
        self.uloc_inst_color = cache_locs(self.prog_inst_color, ["uProj", "uView", "uLightDir", "uCamPos", "uAmbient"])
        self._uloc_ground = cache_locs(self.prog_ground, ["uProj", "uView"])

    def setup_lighting(self):
        glUseProgram(self.prog_mesh)
        glUniform3fv(self.uloc_mesh["uLightDir"], 1, LIGHT_DIR)
        glUniform1f(self.uloc_mesh["uAmbient"], 0.4)
        glUseProgram(self.prog_inst)
        glUniform3fv(self.uloc_inst["uLightDir"], 1, LIGHT_DIR)
        glUniform1f(self.uloc_inst["uAmbient"], 0.4)
        glUseProgram(self.prog_inst_color)
        glUniform3fv(self.uloc_inst_color["uLightDir"], 1, LIGHT_DIR)
        glUniform1f(self.uloc_inst_color["uAmbient"], 0.4)
        glUseProgram(0)

    def setup_ground_geometry(self):
        ground_data, self.ground_vert_count = make_ground_quad()
        self.ground_vao = create_vao()
        self.ground_vbo = create_buffer(ground_data)
        setup_ground_vao(self.ground_vao, self.ground_vbo)
        cube_data, self.cube_vert_count = make_unit_cube() # TODO refactor
        self.cube_geom_vbo = create_buffer(cube_data)

    def setup_sphere_geometry(self):
        sphere_data, self.sphere_vert_count = make_uv_sphere(20, 14)
        self.sphere_geom_vbo = create_buffer(sphere_data)
        self.sphere_inst_vbo = create_buffer(size=64, usage=GL_DYNAMIC_DRAW)
        self.ball_vao = create_vao()
        setup_instanced_vao(self.ball_vao, self.sphere_geom_vbo, self.sphere_inst_vbo)

    def create_object_renderer(self, mesh_indices, voxel_count, voxel_half_extent, block_halves=None):
        return ObjectRenderer(self, mesh_indices, voxel_count, voxel_half_extent, block_halves=block_halves)

    def update_ball_single(self, transforms, body, radius):
        ball_transform = quat_to_mat4(transforms[body][3:7], transforms[body][:3])
        ball_transform[:3, :3] *= radius
        instance_data = np.ascontiguousarray(ball_transform.T.reshape(1, 4, 4))
        glBindBuffer(GL_ARRAY_BUFFER, self.sphere_inst_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, instance_data.nbytes, instance_data)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw_ground(self, proj, view):
        locs = self._uloc_ground
        glUseProgram(self.prog_ground)
        glUniformMatrix4fv(locs["uProj"], 1, GL_TRUE, proj)
        glUniformMatrix4fv(locs["uView"], 1, GL_TRUE, view)
        glBindVertexArray(self.ground_vao)
        glDrawArrays(GL_TRIANGLES, 0, self.ground_vert_count)
        glBindVertexArray(0)

    def draw_ball(self, proj, view, cam_pos, color):
        locs = self.uloc_inst
        glUseProgram(self.prog_inst)
        glUniformMatrix4fv(locs["uProj"], 1, GL_TRUE, proj)
        glUniformMatrix4fv(locs["uView"], 1, GL_TRUE, view)
        glUniform3fv(locs["uColor"], 1, np.asarray(color, dtype=np.float32))
        glUniform3fv(locs["uCamPos"], 1, np.asarray(cam_pos))
        glBindVertexArray(self.ball_vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, self.sphere_vert_count, 1)
        glBindVertexArray(0)
