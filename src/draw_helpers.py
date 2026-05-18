import math, ctypes
import numpy as np
from src.shaders import *
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL import shaders as gl_shaders


def perspective_matrix(fov_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at_matrix(eye, target, up):
    eye = np.asarray(eye, np.float32)
    target = np.asarray(target, np.float32)
    up = np.asarray(up, np.float32)
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def quat_to_mat3(q):
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def quat_to_mat4(q, pos):
    r = quat_to_mat3(q)
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = r
    m[:3, 3] = pos
    return m


def compile_shader_program(vert_src, frag_src):
    vs = gl_shaders.compileShader(vert_src, GL_VERTEX_SHADER)
    fs = gl_shaders.compileShader(frag_src, GL_FRAGMENT_SHADER)
    prog = gl_shaders.compileProgram(vs, fs)
    return prog


def make_unit_cube():
    h = 1.0
    faces = [
        (
            (0, 0, 1),
            [(-h, -h, h), (h, -h, h), (h, h, h), (-h, -h, h), (h, h, h), (-h, h, h)],
        ),
        (
            (0, 0, -1),
            [
                (-h, h, -h),
                (h, h, -h),
                (h, -h, -h),
                (-h, h, -h),
                (h, -h, -h),
                (-h, -h, -h),
            ],
        ),
        (
            (0, 1, 0),
            [(-h, h, -h), (-h, h, h), (h, h, h), (-h, h, -h), (h, h, h), (h, h, -h)],
        ),
        (
            (0, -1, 0),
            [
                (-h, -h, -h),
                (h, -h, -h),
                (h, -h, h),
                (-h, -h, -h),
                (h, -h, h),
                (-h, -h, h),
            ],
        ),
        (
            (1, 0, 0),
            [(h, -h, -h), (h, h, -h), (h, h, h), (h, -h, -h), (h, h, h), (h, -h, h)],
        ),
        (
            (-1, 0, 0),
            [
                (-h, -h, -h),
                (-h, -h, h),
                (-h, h, h),
                (-h, -h, -h),
                (-h, h, h),
                (-h, h, -h),
            ],
        ),
    ]
    verts = []
    for normal, quad_verts in faces:
        for v in quad_verts:
            verts.extend(v)
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
    s = size
    verts = np.array(
        [
            -s,
            -s,
            0,
            s,
            -s,
            0,
            s,
            s,
            0,
            -s,
            -s,
            0,
            s,
            s,
            0,
            -s,
            s,
            0,
        ],
        dtype=np.float32,
    )
    return verts, 6


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


def setup_mesh_vao_gpu(vao, data_vbo, binding_vbo, ebo):
    """Mesh VAO for full GPU deformation.
    data_vbo  : interleaved [localOffset(3), restNormal(3)] — static, set once.
    binding_vbo: float per vertex primary-voxel index       — static, set once."""
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, data_vbo)
    stride = 6 * 4
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(
        0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0)
    )  # aLocalOffset
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(
        1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12)
    )  # aRestNormal
    glBindBuffer(GL_ARRAY_BUFFER, binding_vbo)
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 4, ctypes.c_void_p(0))  # aBinding
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBindVertexArray(0)


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
        loc = 2 + i
        glEnableVertexAttribArray(loc)
        glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(i * 16))
        glVertexAttribDivisor(loc, 1)


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


def setup_ground_vao(vao, vbo):
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
    glBindVertexArray(0)


def compute_vertex_normals(verts, indices):
    idx = indices.reshape(-1, 3)
    v0 = verts[idx[:, 0]]
    v1 = verts[idx[:, 1]]
    v2 = verts[idx[:, 2]]

    face_normals = np.cross(v1 - v0, v2 - v0)

    normals = np.zeros_like(verts)
    np.add.at(normals, idx[:, 0], face_normals)
    np.add.at(normals, idx[:, 1], face_normals)
    np.add.at(normals, idx[:, 2], face_normals)

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norms, 1e-8)
    return normals.astype(np.float32)


def batch_quat_to_mat3(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.empty((len(q), 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (y2 + z2)
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = 1 - 2 * (x2 + z2)
    R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = 1 - 2 * (x2 + y2)
    return R
