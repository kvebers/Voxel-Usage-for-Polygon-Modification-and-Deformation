import ctypes
from OpenGL.GL import *


def _setup_geom_attribs(geom_vbo):
    stride = 6 * 4
    glBindBuffer(GL_ARRAY_BUFFER, geom_vbo)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))


def _bind_instance_matrix_attribs(inst_vbo):
    glBindBuffer(GL_ARRAY_BUFFER, inst_vbo)
    for i in range(4):
        col_attrib = 2 + i
        glEnableVertexAttribArray(col_attrib)
        glVertexAttribPointer(col_attrib, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(i * 16))
        glVertexAttribDivisor(col_attrib, 1)


def setup_mesh_vao(vao, vbo, ebo):
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    stride = 6 * 4
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBindVertexArray(0)


def setup_mesh_vao_gpu(vao, data_vbo, blend_indices_vbo, blend_weights_vbo, ebo):
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, data_vbo)
    stride = 6 * 4
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glBindBuffer(GL_ARRAY_BUFFER, blend_indices_vbo)
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
    glBindBuffer(GL_ARRAY_BUFFER, blend_weights_vbo)
    glEnableVertexAttribArray(3)
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBindVertexArray(0)


def setup_instanced_vao(vao, geom_vbo, inst_vbo):
    glBindVertexArray(vao)
    _setup_geom_attribs(geom_vbo)
    _bind_instance_matrix_attribs(inst_vbo)
    glBindVertexArray(0)


def setup_instanced_color_vao(vao, geom_vbo, inst_vbo, color_vbo):
    glBindVertexArray(vao)
    _setup_geom_attribs(geom_vbo)
    _bind_instance_matrix_attribs(inst_vbo)
    glBindBuffer(GL_ARRAY_BUFFER, color_vbo)
    glEnableVertexAttribArray(6)
    glVertexAttribPointer(6, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
    glVertexAttribDivisor(6, 1)
    glBindVertexArray(0)


def setup_wall_vao(vao, vbo):
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    stride = 6 * 4
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glBindVertexArray(0)


def setup_ground_vao(vao, vbo):
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
    glBindVertexArray(0)
