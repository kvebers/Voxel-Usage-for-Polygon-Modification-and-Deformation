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
    float normalizedDepth = gl_FragCoord.z + 0.5 / float(u_res);
    int   depthBitIndex   = clamp(int(normalizedDepth * float(u_res)), 0, u_res);
    ivec2 pixelCoord      = ivec2(gl_FragCoord.xy);
    int   columnIndex     = pixelCoord.y * u_width + pixelCoord.x;
    int   voxelColumnBase = columnIndex * u_words_z;
    int   wordIndex       = depthBitIndex >> 5;
    int   bitIndex        = depthBitIndex & 31;
    for (int w = 0; w < wordIndex && w < u_words_z; w++) {
        atomicXor(voxels[voxelColumnBase + w], 0xFFFFFFFFu);
    }
    if (wordIndex < u_words_z && bitIndex > 0) {
        uint bitMask = (1u << bitIndex) - 1u;
        atomicXor(voxels[voxelColumnBase + wordIndex], bitMask);
    }
    frag_color = vec4(0.0);
}
"""


def unpack_voxel_bits(raw, resolution, n_words, word_axis, transpose):
    bit_positions = np.arange(32, dtype=np.uint32)
    unpacked = ((raw[..., np.newaxis] >> bit_positions) & 1).astype(np.uint8)
    ndim = raw.ndim
    other_axes = [a for a in range(ndim) if a != word_axis]
    axis_order = other_axes + [word_axis]
    full_axis_order = axis_order + [ndim]
    unpacked = unpacked.transpose(full_axis_order)
    new_shape = list(unpacked.shape[:-2]) + [n_words * 32]
    unpacked = unpacked.reshape(new_shape)
    crop_slices = [slice(None)] * (len(new_shape) - 1) + [slice(0, resolution)]
    grid = unpacked[tuple(crop_slices)]
    grid = grid.transpose(transpose).copy()
    return grid.astype(bool)


def make_ortho_matrix(pad=0.01):
    proj_matrix = np.zeros((4, 4), dtype=np.float32)
    lo, hi = -pad, 1.0 + pad
    for axis in range(3):
        proj_matrix[axis, axis] = 2.0 / (hi - lo)
        proj_matrix[axis, 3] = -(hi + lo) / (hi - lo)
    proj_matrix[3, 3] = 1.0
    return proj_matrix.T.copy()


_cache = {}
_ortho_mvp = make_ortho_matrix()


def get_cache(ctx):
    ctx_id = id(ctx)
    if ctx_id not in _cache:
        prog = ctx.program(vertex_shader=VERT_SRC, fragment_shader=FRAG_SRC)
        prog["u_mvp"].write(_ortho_mvp.tobytes())
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


def update_vbo(ctx, cache, transformed):
    vertex_data_bytes = transformed.nbytes
    if cache["vbo_size"] < vertex_data_bytes:
        if cache["vbo"] is not None:
            cache["vbo"].release()
        cache["vbo"] = ctx.buffer(reserve=vertex_data_bytes)
        cache["vbo_size"] = vertex_data_bytes
    cache["vbo"].write(transformed.tobytes())


def update_fbo(ctx, cache, res):
    if cache["fbo_res"] != res:
        if cache["fbo"] is not None:
            cache["fbo"].release()
        if cache["dummy_tex"] is not None:
            cache["dummy_tex"].release()
        cache["dummy_tex"] = ctx.texture((res, res), 4, dtype="f1")
        cache["fbo"] = ctx.framebuffer(color_attachments=[cache["dummy_tex"]])
        cache["fbo_res"] = res


def update_ssbo(ctx, cache, ssbo_bytes):
    if cache["ssbo_size"] < ssbo_bytes:
        if cache["ssbo"] is not None:
            cache["ssbo"].release()
        cache["ssbo"] = ctx.buffer(reserve=ssbo_bytes)
        cache["ssbo_size"] = ssbo_bytes
    cache["ssbo"].clear()


def render_and_read(ctx, cache, prog, vao, res, words_z):
    ssbo_bytes = res * res * words_z * 4
    prog["u_res"].value = res
    prog["u_words_z"].value = words_z
    prog["u_width"].value = res
    cache["ssbo"].bind_to_storage_buffer(0)
    cache["fbo"].use()
    cache["fbo"].clear()
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.CULL_FACE)
    vao.render(moderngl.TRIANGLES)
    ctx.memory_barrier()
    raw = np.frombuffer(cache["ssbo"].read(size=ssbo_bytes), dtype=np.uint32)
    raw = raw.reshape(res, res, words_z)
    return unpack_voxel_bits(raw, resolution=res, n_words=words_z, word_axis=2, transpose=(1, 0, 2))
