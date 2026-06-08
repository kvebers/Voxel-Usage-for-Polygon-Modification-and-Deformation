import ctypes
from OpenGL.GL import *


def setup_geom_attribs(geom_vbo):
    """
    Tell GPU how to read position and normals.
    """
    stride = 6 * 4  # 6 floats per vertex (3 pos + 3 normal), 4 bytes each
    glBindBuffer(GL_ARRAY_BUFFER, geom_vbo)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))  # slot 0
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))  # slot 12 byte


def bind_instance_matrix_attribs(inst_vbo):
    """
    4x4 matric reading
    """
    glBindBuffer(GL_ARRAY_BUFFER, inst_vbo)
    for i in range(4):
        col_attrib = 2 + i  # slots
        glEnableVertexAttribArray(col_attrib)
        glVertexAttribPointer(col_attrib, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(i * 16))  # 64 bytes per matrix, 16 bytes per column
        glVertexAttribDivisor(col_attrib, 1)  # once per instance


def setup_mesh_vao(vao, vbo, ebo):
    """
    Set up VAO for plain mesh rendering. Reads position and normal from vbo, triangles from ebo.
    """
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    stride = 6 * 4  # 6 floats per vertex
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBindVertexArray(0)


def setup_mesh_vao_gpu(vao, data_vbo, binding_vbo, ebo):
    """
    Set up VAO for voxel mesh.
    """
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, data_vbo)
    stride = 6 * 4  # 6 floats per vertex
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))  # slot 0
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))  # slot 1
    glBindBuffer(GL_ARRAY_BUFFER, binding_vbo)
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 4, ctypes.c_void_p(0))  # slot 2: bytes to read
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)  # index buffer for triangles
    glBindVertexArray(0)


def setup_instanced_vao(vao, geom_vbo, inst_vbo):
    """
    Set up VAO for instanced rendering with uniform color.
    """
    glBindVertexArray(vao)
    setup_geom_attribs(geom_vbo) # position + normal
    bind_instance_matrix_attribs(inst_vbo) # model matrix
    glBindVertexArray(0)


def setup_instanced_color_vao(vao, geom_vbo, inst_vbo, color_vbo):
    """
    VAO Geometry + Color
    """
    glBindVertexArray(vao)
    setup_geom_attribs(geom_vbo) # position and normals
    bind_instance_matrix_attribs(inst_vbo) # model informatio
    glBindBuffer(GL_ARRAY_BUFFER, color_vbo)
    glEnableVertexAttribArray(6)
    glVertexAttribPointer(6, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))  # color
    glVertexAttribDivisor(6, 1) # 1 color
    glBindVertexArray(0)


def setup_ground_vao(vao, vbo):
    """
    Set up VAO for ground plane. Just position, no normal needed.
    """
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))  # 12 pos per vertex
    glBindVertexArray(0)
