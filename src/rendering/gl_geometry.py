import math, ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders as gl_shaders


def compile_shader_program(vert_src, frag_src):
    vertex_shader = gl_shaders.compileShader(vert_src, GL_VERTEX_SHADER)
    fragment_shader = gl_shaders.compileShader(frag_src, GL_FRAGMENT_SHADER)
    return gl_shaders.compileProgram(vertex_shader, fragment_shader)


def make_unit_cube():
    half_extent = 1.0
    faces = [
        (
            (0, 0, 1),
            [
                (-half_extent, -half_extent, half_extent),
                (half_extent, -half_extent, half_extent),
                (half_extent, half_extent, half_extent),
                (-half_extent, -half_extent, half_extent),
                (half_extent, half_extent, half_extent),
                (-half_extent, half_extent, half_extent),
            ],
        ),
        (
            (0, 0, -1),
            [
                (-half_extent, half_extent, -half_extent),
                (half_extent, half_extent, -half_extent),
                (half_extent, -half_extent, -half_extent),
                (-half_extent, half_extent, -half_extent),
                (half_extent, -half_extent, -half_extent),
                (-half_extent, -half_extent, -half_extent),
            ],
        ),
        (
            (0, 1, 0),
            [
                (-half_extent, half_extent, -half_extent),
                (-half_extent, half_extent, half_extent),
                (half_extent, half_extent, half_extent),
                (-half_extent, half_extent, -half_extent),
                (half_extent, half_extent, half_extent),
                (half_extent, half_extent, -half_extent),
            ],
        ),
        (
            (0, -1, 0),
            [
                (-half_extent, -half_extent, -half_extent),
                (half_extent, -half_extent, -half_extent),
                (half_extent, -half_extent, half_extent),
                (-half_extent, -half_extent, -half_extent),
                (half_extent, -half_extent, half_extent),
                (-half_extent, -half_extent, half_extent),
            ],
        ),
        (
            (1, 0, 0),
            [
                (half_extent, -half_extent, -half_extent),
                (half_extent, half_extent, -half_extent),
                (half_extent, half_extent, half_extent),
                (half_extent, -half_extent, -half_extent),
                (half_extent, half_extent, half_extent),
                (half_extent, -half_extent, half_extent),
            ],
        ),
        (
            (-1, 0, 0),
            [
                (-half_extent, -half_extent, -half_extent),
                (-half_extent, -half_extent, half_extent),
                (-half_extent, half_extent, half_extent),
                (-half_extent, -half_extent, -half_extent),
                (-half_extent, half_extent, half_extent),
                (-half_extent, half_extent, -half_extent),
            ],
        ),
    ]
    verts = []
    for normal, quad_verts in faces:
        for vertex in quad_verts:
            verts.extend(vertex)
            verts.extend(normal)
    return np.array(verts, dtype=np.float32), 36


def make_uv_sphere(slices=8, stacks=12):
    verts = []
    for i in range(stacks):
        theta0 = math.pi * i / stacks
        theta1 = math.pi * (i + 1) / stacks
        for j in range(slices):
            phi0 = 2.0 * math.pi * j / slices
            phi1 = 2.0 * math.pi * (j + 1) / slices
            p00 = (
                math.sin(theta0) * math.cos(phi0),
                math.sin(theta0) * math.sin(phi0),
                math.cos(theta0),
            )
            p10 = (
                math.sin(theta1) * math.cos(phi0),
                math.sin(theta1) * math.sin(phi0),
                math.cos(theta1),
            )
            p01 = (
                math.sin(theta0) * math.cos(phi1),
                math.sin(theta0) * math.sin(phi1),
                math.cos(theta0),
            )
            p11 = (
                math.sin(theta1) * math.cos(phi1),
                math.sin(theta1) * math.sin(phi1),
                math.cos(theta1),
            )
            for p in [p00, p10, p11, p00, p11, p01]:
                verts.extend(p)
                verts.extend(p)
    return np.array(verts, dtype=np.float32), stacks * slices * 6


def make_ground_quad(size=30.0):
    verts = np.array(
        [
            -size,
            -size,
            0,
            size,
            -size,
            0,
            size,
            size,
            0,
            -size,
            -size,
            0,
            size,
            size,
            0,
            -size,
            size,
            0,
        ],
        dtype=np.float32,
    )
    return verts, 6


def compute_vertex_normals(verts, indices):
    triangles = indices.reshape(-1, 3)
    v0, v1, v2 = verts[triangles[:, 0]], verts[triangles[:, 1]], verts[triangles[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(verts)
    np.add.at(normals, triangles[:, 0], face_normals)
    np.add.at(normals, triangles[:, 1], face_normals)
    np.add.at(normals, triangles[:, 2], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, 1e-8)
    return normals.astype(np.float32)


def create_vao():
    vao = GLuint(0)
    glGenVertexArrays(1, ctypes.byref(vao))
    return vao.value


def create_buffer(data=None, size=0, usage=GL_STATIC_DRAW):
    buf = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, buf)
    if data is not None:
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, usage)
    else:
        glBufferData(GL_ARRAY_BUFFER, size, None, usage)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    return buf
