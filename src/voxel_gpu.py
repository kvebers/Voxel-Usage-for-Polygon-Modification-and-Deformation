import numpy as np
import moderngl
from src.voxel_repair import repair_isolated_voxels, make_hollow
from src.voxel_topology import greedy_merge_grid, build_block_topology
from src.voxel_render_ctx import _get_cache, _update_vbo, _update_fbo, _update_ssbo, _render_and_read


def preprocess_vertices(vertex_data: np.ndarray, pad: float = 0.01) -> tuple[np.ndarray, int]:
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


def voxelize_gpu(
    vertex_data: np.ndarray,
    resolution: int = 64,
    *,
    ctx: moderngl.Context | None = None,
    prenormalized: bool = False,
) -> np.ndarray:
    own_ctx = ctx is None
    if own_ctx:
        ctx = moderngl.create_standalone_context()
    res = resolution
    words_z = (res + 31) // 32
    if prenormalized:
        transformed = np.asarray(vertex_data, dtype=np.float32).reshape(-1, 3)
    else:
        transformed, _num_tris = preprocess_vertices(vertex_data, pad=0.01)
    c = _get_cache(ctx)
    prog = c["prog"]
    _update_vbo(ctx, c, transformed)
    vao = ctx.vertex_array(prog, [(c["vbo"], "3f", "in_position")])
    _update_fbo(ctx, c, res)
    _update_ssbo(ctx, c, res * res * words_z * 4)
    grid = _render_and_read(ctx, c, prog, vao, res, words_z)
    vao.release()
    if own_ctx:
        ctx.release()
    return grid
