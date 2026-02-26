
import numpy as np
import moderngl



import numpy as np


def preprocess_vertices(
    vertex_data: np.ndarray,
    pad: float = 0.01,
) -> tuple[np.ndarray, int]:
    """Reshape, normalise vertices to [pad, 1-pad], return float32 + tri count.

    Parameters
    ----------
    vertex_data : array-like
        Flat or (N, 3) vertex positions.
    pad : float
        Margin inside [0, 1] to avoid boundary artefacts.

    Returns
    -------
    positions : np.ndarray, shape (V, 3), dtype float32
        Normalised vertex positions.
    num_tris : int
        Number of triangles (V // 3).
    """
    positions = np.asarray(vertex_data, dtype=np.float32).reshape(-1, 3)
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    extent = (hi - lo).max()
    if extent > 0:
        usable = 1.0 - 2.0 * pad          # e.g. 0.98
        positions = (positions - lo) / extent * usable + pad
    else:
        positions[:] = 0.5
    num_tris = len(positions) // 3
    return positions, num_tris


def unpack_voxel_bits(
    raw: np.ndarray,
    resolution: int,
    n_words: int,
    word_axis: int,
    transpose: tuple[int, ...],
) -> np.ndarray:
    """Vectorised bit-unpacking of a packed uint32 voxel buffer.

    Parameters
    ----------
    raw : np.ndarray, dtype uint32
        Packed voxel data.  Shape depends on the method:
        - compute shader : (words_x, res, res)
        - rasterisation  : (res, res, words_z)
    resolution : int
        Grid resolution.
    n_words : int
        Number of 32-bit words along the packed axis.
    word_axis : int
        Which axis of *raw* is the packed-word axis (0 or 2).
    transpose : tuple of int
        Axes permutation applied after unpacking to get the final
        (X, Y, Z) grid.

    Returns
    -------
    grid : np.ndarray, shape (res, res, res), dtype bool
    """
    bits = np.arange(32, dtype=np.uint32)

    # Expand raw along a new last axis for each of the 32 bits
    # Result shape: (*raw.shape, 32)
    unpacked = ((raw[..., np.newaxis] >> bits) & 1).astype(np.uint8)

    # Merge the word axis and the new bit axis into the voxel axis.
    # Move the word axis to be adjacent to the bit axis first.
    shape = list(raw.shape)
    ndim = raw.ndim  # 3

    # We want to combine axis `word_axis` (size n_words) with the last
    # axis (size 32) into a single axis of length n_words * 32, then
    # trim to `resolution`.
    #
    # Strategy: move word_axis to position -2 so that the last two axes
    # are (n_words, 32), then reshape to merge them.

    # Build the axes order: put word_axis second-to-last, keep others.
    other_axes = [a for a in range(ndim) if a != word_axis]
    perm = other_axes + [word_axis]          # e.g. [1,2,0] or [0,1,2]
    # unpacked has one extra trailing axis (bits), so extend perm
    perm_full = perm + [ndim]                # ndim is the bit axis index
    unpacked = unpacked.transpose(perm_full)

    # Now shape is (A, B, n_words, 32) where A, B are the two non-word dims.
    new_shape = list(unpacked.shape[:-2]) + [n_words * 32]
    unpacked = unpacked.reshape(new_shape)

    # Trim to resolution along the merged axis
    slices = [slice(None)] * (len(new_shape) - 1) + [slice(0, resolution)]
    grid = unpacked[tuple(slices)]

    # Apply the final transpose to get (X, Y, Z)
    grid = grid.transpose(transpose).copy()
    return grid.astype(bool)

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


def voxelize_gpu(
    vertex_data: np.ndarray,
    vertex_count: int | None = None,
    resolution: int = 64,
    *,
    ctx: moderngl.Context | None = None,
    prenormalized: bool = False,
) -> np.ndarray:
    own_ctx = ctx is None
    if own_ctx:
        ctx = moderngl.create_standalone_context()

    res = resolution
    w = h = res
    words_z = (res + 31) // 32

    if prenormalized:
        transformed = np.asarray(vertex_data, dtype=np.float32).reshape(-1, 3)
    else:
        transformed, _num_tris = preprocess_vertices(vertex_data, pad=0.01)

    c = _get_cache(ctx)
    prog = c["prog"]
    vert_bytes = transformed.nbytes
    if c["vbo_size"] < vert_bytes:
        if c["vbo"] is not None:
            c["vbo"].release()
        c["vbo"] = ctx.buffer(reserve=vert_bytes)
        c["vbo_size"] = vert_bytes
    c["vbo"].write(transformed.tobytes())
    vao = ctx.vertex_array(prog, [(c["vbo"], "3f", "in_position")])
    if c["fbo_res"] != res:
        if c["fbo"] is not None:
            c["fbo"].release()
        if c["dummy_tex"] is not None:
            c["dummy_tex"].release()
        c["dummy_tex"] = ctx.texture((w, h), 4, dtype="f1")
        c["fbo"] = ctx.framebuffer(color_attachments=[c["dummy_tex"]])
        c["fbo_res"] = res

    ssbo_bytes = w * h * words_z * 4
    if c["ssbo_size"] < ssbo_bytes:
        if c["ssbo"] is not None:
            c["ssbo"].release()
        c["ssbo"] = ctx.buffer(reserve=ssbo_bytes)
        c["ssbo_size"] = ssbo_bytes
    c["ssbo"].clear()

    prog["u_res"].value = res
    prog["u_words_z"].value = words_z
    prog["u_width"].value = w

    c["ssbo"].bind_to_storage_buffer(0)
    c["fbo"].use()
    c["fbo"].clear()
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.CULL_FACE)
    vao.render(moderngl.TRIANGLES)
    ctx.memory_barrier()

    raw = np.frombuffer(c["ssbo"].read(size=ssbo_bytes), dtype=np.uint32)
    raw = raw.reshape(h, w, words_z)

    grid = unpack_voxel_bits(
        raw,
        resolution=res,
        n_words=words_z,
        word_axis=2,
        transpose=(1, 0, 2),
    )

    vao.release()

    if own_ctx:
        ctx.release()

    return grid