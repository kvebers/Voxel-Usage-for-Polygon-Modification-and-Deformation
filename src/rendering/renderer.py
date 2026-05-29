import ctypes
import numpy as np
from OpenGL.GL import *
from src.rendering.draw_helpers import *
from src.rendering.shaders import *
from src.constants import LIGHT_DIR


def _cache_locs(prog, names):
    return {name: glGetUniformLocation(prog, name) for name in names}


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
        self._voxel_skin_t_buf = self._voxel_skin_t_tex = None
        self._voxel_quat_buf = self._voxel_quat_tex = None
        self._vox_rest_pos = None
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
        static_vertex_data[:, :3] = mesh_splitter.rest_vert_positions
        static_vertex_data[:, 3:] = mesh_splitter.rest_normals
        glBindBuffer(GL_ARRAY_BUFFER, self.mesh_vbo)
        if static_vertex_data.nbytes > self._mesh_vbo_capacity:
            self._mesh_vbo_capacity = static_vertex_data.nbytes
            glBufferData(GL_ARRAY_BUFFER, static_vertex_data.nbytes, static_vertex_data, GL_STATIC_DRAW)
        else:
            glBufferSubData(GL_ARRAY_BUFFER, 0, static_vertex_data.nbytes, static_vertex_data)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        blend_indices, blend_weights = mesh_splitter.get_lbs_data()
        self.mesh_blend_idx_vbo = create_buffer(blend_indices, usage=GL_DYNAMIC_DRAW)
        self.mesh_blend_wt_vbo = create_buffer(blend_weights, usage=GL_DYNAMIC_DRAW)
        self._vox_rest_pos = np.ascontiguousarray(mesh_splitter._vox_rest_pos[:, :3], dtype=np.float32)
        self._voxel_skin_t_buf, self._voxel_skin_t_tex = _create_tbo(voxel_count * 12, GL_RGB32F)
        self._voxel_quat_buf, self._voxel_quat_tex = _create_tbo(voxel_count * 16, GL_RGBA32F)
        setup_mesh_vao_gpu(self.mesh_vao, self.mesh_vbo, self.mesh_blend_idx_vbo, self.mesh_blend_wt_vbo, self.mesh_ebo)
        glUseProgram(self._renderer.prog_mesh)
        glUniform1i(glGetUniformLocation(self._renderer.prog_mesh, "uVoxelSkinT"), 0)
        glUniform1i(glGetUniformLocation(self._renderer.prog_mesh, "uVoxelQuat"), 1)
        glUseProgram(0)

    def update_lbs_weights(self, blend_indices, blend_weights):
        glBindBuffer(GL_ARRAY_BUFFER, self.mesh_blend_idx_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, blend_indices.nbytes, blend_indices)
        glBindBuffer(GL_ARRAY_BUFFER, self.mesh_blend_wt_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, blend_weights.nbytes, blend_weights)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def update_voxel_transforms(self, voxel_slice):
        pos  = np.ascontiguousarray(voxel_slice[:, :3])
        quat = np.ascontiguousarray(voxel_slice[:, 3:7])

        invalid_mask = ~np.isfinite(pos).all(axis=1)
        if invalid_mask.any():
            if self._last_valid_pos is not None:
                pos[invalid_mask]  = self._last_valid_pos[invalid_mask]
                quat[invalid_mask] = self._last_valid_quat[invalid_mask]
        valid_mask = ~invalid_mask
        if self._last_valid_pos is None:
            self._last_valid_pos  = pos.copy()
            self._last_valid_quat = quat.copy()
        elif valid_mask.any():
            self._last_valid_pos[valid_mask]  = pos[valid_mask]
            self._last_valid_quat[valid_mask] = quat[valid_mask]

        # skin_t[i] = curPos[i] - quatRotate(quat[i], restPos[i])
        # Computed per voxel (cheap) so the shader needs only 2 TBO fetches per influence.
        q  = quat.astype(np.float32)
        rp = self._vox_rest_pos
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        tx = 2.0 * (y * rp[:, 2] - z * rp[:, 1])
        ty = 2.0 * (z * rp[:, 0] - x * rp[:, 2])
        tz = 2.0 * (x * rp[:, 1] - y * rp[:, 0])
        rotated_rest = np.stack([
            rp[:, 0] + w * tx + y * tz - z * ty,
            rp[:, 1] + w * ty + z * tx - x * tz,
            rp[:, 2] + w * tz + x * ty - y * tx,
        ], axis=1).astype(np.float32)
        skin_t = np.ascontiguousarray(pos.astype(np.float32) - rotated_rest)

        glBindBuffer(GL_TEXTURE_BUFFER, self._voxel_skin_t_buf)
        glBufferSubData(GL_TEXTURE_BUFFER, 0, skin_t.nbytes, skin_t)
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
        locs = renderer._uloc_mesh
        glUseProgram(renderer.prog_mesh)
        glUniformMatrix4fv(locs["uProj"], 1, GL_TRUE, proj)
        glUniformMatrix4fv(locs["uView"], 1, GL_TRUE, view)
        glUniform3fv(locs["uColor"], 1, np.asarray(color, dtype=np.float32))
        glUniform3fv(locs["uCamPos"], 1, np.asarray(cam_pos))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_BUFFER, self._voxel_skin_t_tex)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_BUFFER, self._voxel_quat_tex)
        glBindVertexArray(self.mesh_vao)
        draw_count = index_count if index_count is not None else self.mesh_index_count
        glDrawElements(GL_TRIANGLES, draw_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def draw_voxels(self, proj, view, cam_pos, alpha=1.0):
        renderer = self._renderer
        locs = renderer._uloc_inst_color
        glUseProgram(renderer.prog_inst_color)
        glUniformMatrix4fv(locs["uProj"], 1, GL_TRUE, proj)
        glUniformMatrix4fv(locs["uView"], 1, GL_TRUE, view)
        glUniform3fv(locs["uCamPos"], 1, np.asarray(cam_pos))
        glUniform1f(locs["uAlpha"], alpha)
        transparent = alpha < 1.0
        if transparent:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDepthMask(GL_FALSE)
        glBindVertexArray(self.voxel_vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, renderer.cube_vert_count, self.voxel_count)
        glBindVertexArray(0)
        if transparent:
            glDepthMask(GL_TRUE)
            glDisable(GL_BLEND)


class Renderer:
    def __init__(self):
        self._compile_programs()
        self._setup_lighting()
        self._setup_ground_geometry()
        self._setup_sphere_geometry()
        self._wall_vaos = []
        self._wall_vert_counts = []
        self._wall_colors = []

    def _compile_programs(self):
        self.prog_mesh = compile_shader_program(VERT_MESH, FRAG_BLINN)
        self.prog_inst = compile_shader_program(VERT_INSTANCED, FRAG_BLINN)
        self.prog_inst_color = compile_shader_program(VERT_INSTANCED_COLOR, FRAG_BLINN_INSTANCED)
        self.prog_ground = compile_shader_program(VERT_GROUND, FRAG_GROUND)
        self.prog_wall = compile_shader_program(VERT_WALL, FRAG_BLINN)
        self._uloc_mesh = _cache_locs(self.prog_mesh, ["uProj", "uView", "uColor", "uLightDir", "uCamPos", "uAmbient"])
        self._uloc_inst = _cache_locs(self.prog_inst, ["uProj", "uView", "uColor", "uLightDir", "uCamPos", "uAmbient"])
        self._uloc_inst_color = _cache_locs(self.prog_inst_color, ["uProj", "uView", "uLightDir", "uCamPos", "uAmbient", "uAlpha"])
        self._uloc_ground = _cache_locs(self.prog_ground, ["uProj", "uView"])
        self._uloc_wall = _cache_locs(self.prog_wall, ["uProj", "uView", "uColor", "uLightDir", "uCamPos", "uAmbient"])

    def _setup_lighting(self):
        glUseProgram(self.prog_mesh)
        glUniform3fv(self._uloc_mesh["uLightDir"], 1, LIGHT_DIR)
        glUniform1f(self._uloc_mesh["uAmbient"], 0.4)
        glUseProgram(self.prog_inst)
        glUniform3fv(self._uloc_inst["uLightDir"], 1, LIGHT_DIR)
        glUniform1f(self._uloc_inst["uAmbient"], 0.4)
        glUseProgram(self.prog_inst_color)
        glUniform3fv(self._uloc_inst_color["uLightDir"], 1, LIGHT_DIR)
        glUniform1f(self._uloc_inst_color["uAmbient"], 0.4)
        glUniform1f(self._uloc_inst_color["uAlpha"], 1.0)
        glUseProgram(self.prog_wall)
        glUniform3fv(self._uloc_wall["uLightDir"], 1, LIGHT_DIR)
        glUniform1f(self._uloc_wall["uAmbient"], 0.4)
        glUseProgram(0)

    def _setup_ground_geometry(self):
        ground_data, self.ground_vert_count = make_ground_quad()
        self.ground_vao = create_vao()
        self.ground_vbo = create_buffer(ground_data)
        setup_ground_vao(self.ground_vao, self.ground_vbo)
        cube_data, self.cube_vert_count = make_unit_cube()
        self.cube_geom_vbo = create_buffer(cube_data)

    def _setup_sphere_geometry(self):
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

    def setup_walls(self, cfg):
        walls_cfg = getattr(cfg, "walls", [])
        for wall_config in walls_cfg:
            corners = np.array(wall_config.corners, dtype=np.float32)
            p0, p1, p2, p3 = corners[0], corners[1], corners[2], corners[3]
            edge1 = p1 - p0
            edge2 = p3 - p0
            normal = np.cross(edge1, edge2).astype(np.float32)
            n_len = np.linalg.norm(normal)
            if n_len > 0:
                normal /= n_len
            verts = []
            for p in [p0, p1, p2, p0, p2, p3]:
                verts.extend(p)
                verts.extend(normal)
            data = np.array(verts, dtype=np.float32)
            vao = create_vao()
            vbo = create_buffer(data)
            setup_wall_vao(vao, vbo)
            self._wall_vaos.append(vao)
            self._wall_vert_counts.append(6)
            self._wall_colors.append(getattr(wall_config, "color", [0.55, 0.60, 0.65]))

    def draw_walls(self, proj, view, cam_pos):
        if not self._wall_vaos:
            return
        locs = self._uloc_wall
        glUseProgram(self.prog_wall)
        glUniformMatrix4fv(locs["uProj"], 1, GL_TRUE, proj)
        glUniformMatrix4fv(locs["uView"], 1, GL_TRUE, view)
        glUniform3fv(locs["uCamPos"], 1, np.asarray(cam_pos, dtype=np.float32))
        for vao, vert_count, color in zip(self._wall_vaos, self._wall_vert_counts, self._wall_colors):
            glUniform3fv(locs["uColor"], 1, np.asarray(color, dtype=np.float32))
            glBindVertexArray(vao)
            glDrawArrays(GL_TRIANGLES, 0, vert_count)
        glBindVertexArray(0)

    def draw_ball(self, proj, view, cam_pos, color):
        locs = self._uloc_inst
        glUseProgram(self.prog_inst)
        glUniformMatrix4fv(locs["uProj"], 1, GL_TRUE, proj)
        glUniformMatrix4fv(locs["uView"], 1, GL_TRUE, view)
        glUniform3fv(locs["uColor"], 1, np.asarray(color, dtype=np.float32))
        glUniform3fv(locs["uCamPos"], 1, np.asarray(cam_pos))
        glBindVertexArray(self.ball_vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, self.sphere_vert_count, 1)
        glBindVertexArray(0)
