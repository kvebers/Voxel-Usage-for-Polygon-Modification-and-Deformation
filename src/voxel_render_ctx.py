import numpy as np
import moderngl


VERT_SRC = """
#version 430

in vec3 in_position;
uniform mat4 u_mvp;

void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

FRAG_SRC = """
#version 430

uniform int u_res;
uniform int u_words_z;
uniform int u_width;

layout(std430, binding = 0) buffer VoxelBuf {
    uint voxels[];
};

out vec4 frag_color;

void main() {
    float depth = gl_FragCoord.z + 0.5 / float(u_res);
    int k = clamp(int(depth * float(u_res)), 0, u_res);
    ivec2 px  = ivec2(gl_FragCoord.xy);
    int   col = px.y * u_width + px.x;
    int   base = col * u_words_z;
    int k_word = k >> 5;
    int k_bit  = k & 31;
    for (int w = 0; w < k_word && w < u_words_z; w++) {
        atomicXor(voxels[base + w], 0xFFFFFFFFu);
    }
    if (k_word < u_words_z && k_bit > 0) {
        uint mask = (1u << k_bit) - 1u;
        atomicXor(voxels[base + k_word], mask);
    }
    frag_color = vec4(0.0);
}
"""


def unpack_voxel_bits(raw, resolution, n_words, word_axis, transpose):
    bits = np.arange(32, dtype=np.uint32)
    unpacked = ((raw[..., np.newaxis] >> bits) & 1).astype(np.uint8)
    ndim = raw.ndim
    other_axes = [a for a in range(ndim) if a != word_axis]
    perm = other_axes + [word_axis]
    perm_full = perm + [ndim]
    unpacked = unpacked.transpose(perm_full)
    new_shape = list(unpacked.shape[:-2]) + [n_words * 32]
    unpacked = unpacked.reshape(new_shape)
    slices = [slice(None)] * (len(new_shape) - 1) + [slice(0, resolution)]
    grid = unpacked[tuple(slices)]
    grid = grid.transpose(transpose).copy()
    return grid.astype(bool)


def _ortho_01(pad: float = 0.01) -> np.ndarray:
    m = np.zeros((4, 4), dtype=np.float32)
    lo, hi = -pad, 1.0 + pad
    for axis in range(3):
        m[axis, axis] = 2.0 / (hi - lo)
        m[axis, 3] = -(hi + lo) / (hi - lo)
    m[3, 3] = 1.0
    return m.T.copy()


_cache: dict[int, dict] = {}
_mvp = _ortho_01()


def _get_cache(ctx: moderngl.Context) -> dict:
    ctx_id = id(ctx)
    if ctx_id not in _cache:
        prog = ctx.program(vertex_shader=VERT_SRC, fragment_shader=FRAG_SRC)
        prog["u_mvp"].write(_mvp.tobytes())
        _cache[ctx_id] = {
            "prog": prog,
            "fbo": None,
            "fbo_res": 0,
            "dummy_tex": None,
            "ssbo": None,
            "ssbo_size": 0,
            "vbo": None,
            "vbo_size": 0,
        }
    return _cache[ctx_id]


def _update_vbo(ctx, c, transformed):
    vert_bytes = transformed.nbytes
    if c["vbo_size"] < vert_bytes:
        if c["vbo"] is not None:
            c["vbo"].release()
        c["vbo"] = ctx.buffer(reserve=vert_bytes)
        c["vbo_size"] = vert_bytes
    c["vbo"].write(transformed.tobytes())


def _update_fbo(ctx, c, res):
    if c["fbo_res"] != res:
        if c["fbo"] is not None:
            c["fbo"].release()
        if c["dummy_tex"] is not None:
            c["dummy_tex"].release()
        c["dummy_tex"] = ctx.texture((res, res), 4, dtype="f1")
        c["fbo"] = ctx.framebuffer(color_attachments=[c["dummy_tex"]])
        c["fbo_res"] = res


def _update_ssbo(ctx, c, ssbo_bytes):
    if c["ssbo_size"] < ssbo_bytes:
        if c["ssbo"] is not None:
            c["ssbo"].release()
        c["ssbo"] = ctx.buffer(reserve=ssbo_bytes)
        c["ssbo_size"] = ssbo_bytes
    c["ssbo"].clear()


def _render_and_read(ctx, c, prog, vao, res, words_z):
    ssbo_bytes = res * res * words_z * 4
    prog["u_res"].value = res
    prog["u_words_z"].value = words_z
    prog["u_width"].value = res
    c["ssbo"].bind_to_storage_buffer(0)
    c["fbo"].use()
    c["fbo"].clear()
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.CULL_FACE)
    vao.render(moderngl.TRIANGLES)
    ctx.memory_barrier()
    raw = np.frombuffer(c["ssbo"].read(size=ssbo_bytes), dtype=np.uint32)
    raw = raw.reshape(res, res, words_z)
    return unpack_voxel_bits(raw, resolution=res, n_words=words_z, word_axis=2, transpose=(1, 0, 2))
