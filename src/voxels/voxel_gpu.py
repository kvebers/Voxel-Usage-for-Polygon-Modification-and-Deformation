import numpy as np
import moderngl
from src.rendering.voxel_render_ctx import (
    get_cache,
    update_vbo,
    update_fbo,
    update_ssbo,
    render_and_read,
)


def preprocess_vertices(vertex_data, pad=0.01):
    positions = np.asarray(vertex_data, dtype=np.float32).reshape(-1, 3)
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    extent = (hi - lo).max()
    if extent > 0:
        usable = 1.0 - 2.0 * pad
        positions = (positions - lo) / extent * usable + pad
    else:
        positions[:] = 0.5
    num_tris = len(positions) // 3
    return positions, num_tris


def voxelize_gpu(vertex_data, resolution=64, *, ctx=None, prenormalized=False):
    own_ctx = ctx is None
    if own_ctx:
        ctx = moderngl.create_standalone_context()
    words_z = (resolution + 31) // 32
    if prenormalized:
        transformed = np.asarray(vertex_data, dtype=np.float32).reshape(-1, 3)
    else:
        transformed, _num_tris = preprocess_vertices(vertex_data, pad=0.01)
    cache = get_cache(ctx)
    prog = cache["prog"]
    update_vbo(ctx, cache, transformed)
    vao = ctx.vertex_array(prog, [(cache["vbo"], "3f", "in_position")])
    update_fbo(ctx, cache, resolution)
    update_ssbo(ctx, cache, resolution * resolution * words_z * 4)
    grid = render_and_read(ctx, cache, prog, vao, resolution, words_z)
    vao.release()
    if own_ctx:
        ctx.release()
    return grid
