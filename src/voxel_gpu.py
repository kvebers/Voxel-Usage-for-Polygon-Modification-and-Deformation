import numpy as np
import moderngl
from scipy.ndimage import label
from scipy.spatial import KDTree
from src.constants import FACE_DEFS


def preprocess_vertices(
    vertex_data: np.ndarray, pad: float = 0.01
) -> tuple[np.ndarray, int]:
    positions = np.asarray(vertex_data, dtype=np.float32).reshape(-1, 3)
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    extent = (hi - lo).max()
    if extent > 0:
        usable = 1.0 - 2.0 * pad  # e.g. 0.98
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
    bits = np.arange(32, dtype=np.uint32)
    unpacked = ((raw[..., np.newaxis] >> bits) & 1).astype(np.uint8)
    shape = list(raw.shape)
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


def _make_struct6() -> tuple[np.ndarray, np.ndarray]:
    struct6 = np.zeros((3, 3, 3), dtype=bool)
    for idx in [
        (1, 1, 1),
        (0, 1, 1),
        (2, 1, 1),
        (1, 0, 1),
        (1, 2, 1),
        (1, 1, 0),
        (1, 1, 2),
    ]:
        struct6[idx] = True
    kernel = struct6.copy()
    kernel[1, 1, 1] = False
    return struct6, kernel


def _find_closest_gap(labeled, main_label, main_coords):
    main_tree = KDTree(main_coords)
    n = int(labeled.max())
    best_dist = np.inf
    best_src = best_dst = None
    for comp in range(1, n + 1):
        if comp == main_label:
            continue
        comp_coords = np.argwhere(labeled == comp)
        dists, idxs = main_tree.query(comp_coords)
        i = int(dists.argmin())
        if dists[i] < best_dist:
            best_dist = dists[i]
            best_src = comp_coords[i]
            best_dst = main_coords[idxs[i]]
    return best_src, best_dst


def repair_isolated_voxels(active: np.ndarray, pool: np.ndarray) -> np.ndarray:
    struct6, _ = _make_struct6()
    result = active.astype(bool).copy()

    while True:
        labeled, n = label(result, structure=struct6)
        if n <= 1:
            break
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        main_label = int(sizes.argmax())
        main_coords = np.argwhere(labeled == main_label)
        best_src, best_dst = _find_closest_gap(labeled, main_label, main_coords)
        pos = list(best_src)
        for axis in range(3):
            while pos[axis] != best_dst[axis]:
                pos[axis] += 1 if best_dst[axis] > pos[axis] else -1
                result[pos[0], pos[1], pos[2]] = True

    return result


def make_hollow(grid: np.ndarray, min_thickness: int = 2) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    struct6, _ = _make_struct6()
    deep_interior = binary_erosion(grid, structure=struct6, iterations=min_thickness)
    shell = (grid & ~deep_interior).copy()
    return repair_isolated_voxels(shell, grid)


def greedy_merge_grid(grid: np.ndarray) -> list:
    remaining = grid.astype(bool).copy()
    blocks = []
    for x0, y0, z0 in np.argwhere(remaining):
        x0, y0, z0 = int(x0), int(y0), int(z0)
        if not remaining[x0, y0, z0]:
            continue
        x1 = x0
        while x1 + 1 < grid.shape[0] and remaining[x1 + 1, y0, z0]:
            x1 += 1
        y1 = y0
        while y1 + 1 < grid.shape[1] and remaining[x0 : x1 + 1, y1 + 1, z0].all():
            y1 += 1
        z1 = z0
        while (
            z1 + 1 < grid.shape[2] and remaining[x0 : x1 + 1, y0 : y1 + 1, z1 + 1].all()
        ):
            z1 += 1
        blocks.append((x0, y0, z0, x1, y1, z1))
        remaining[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] = False
    return blocks


def _build_block_geometry(blocks: list, grid_shape: tuple):
    voxel_to_block = np.full(grid_shape, -1, dtype=np.int32)
    for bi, (x0, y0, z0, x1, y1, z1) in enumerate(blocks):
        voxel_to_block[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] = bi
    block_centers = [
        ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0)
        for x0, y0, z0, x1, y1, z1 in blocks
    ]
    block_halves_voxel = [
        ((x1 - x0 + 1) / 2.0, (y1 - y0 + 1) / 2.0, (z1 - z0 + 1) / 2.0)
        for x0, y0, z0, x1, y1, z1 in blocks
    ]
    return voxel_to_block, block_centers, block_halves_voxel


def _scan_block_faces(blocks, voxel_to_block, grid_shape):
    pair_joints: dict = {}
    for bi, blk in enumerate(blocks):
        for (dx, dy, dz), face_fn in FACE_DEFS:
            fx0, fx1, fy0, fy1, fz0, fz1 = face_fn(blk)
            fxs, fys, fzs = np.meshgrid(
                np.arange(fx0, fx1 + 1),
                np.arange(fy0, fy1 + 1),
                np.arange(fz0, fz1 + 1),
                indexing="ij",
            )
            fxs, fys, fzs = fxs.ravel(), fys.ravel(), fzs.ravel()
            axs, ays, azs = fxs + dx, fys + dy, fzs + dz
            in_bounds = (
                (0 <= axs)
                & (axs < grid_shape[0])
                & (0 <= ays)
                & (ays < grid_shape[1])
                & (0 <= azs)
                & (azs < grid_shape[2])
            )
            axs, ays, azs = axs[in_bounds], ays[in_bounds], azs[in_bounds]
            fxs, fys, fzs = fxs[in_bounds], fys[in_bounds], fzs[in_bounds]
            others = voxel_to_block[axs, ays, azs]
            valid = (others >= 0) & (others != bi)
            for fx, fy, fz, other in zip(
                fxs[valid], fys[valid], fzs[valid], others[valid]
            ):
                key = (min(bi, int(other)), max(bi, int(other)))
                jx = float(fx) + (1 + dx) / 2
                jy = float(fy) + (1 + dy) / 2
                jz = float(fz) + (1 + dz) / 2
                pair_joints.setdefault(key, set()).add((jx, jy, jz))
    return pair_joints


def _build_joint_offsets(pair_joints, block_centers):
    neighbor_pairs = []
    joint_voxel_offsets = []
    for (a, b), pts in pair_joints.items():
        cx_a = block_centers[a][0] + 0.5
        cy_a = block_centers[a][1] + 0.5
        cz_a = block_centers[a][2] + 0.5
        cx_b = block_centers[b][0] + 0.5
        cy_b = block_centers[b][1] + 0.5
        cz_b = block_centers[b][2] + 0.5
        for jx, jy, jz in pts:
            neighbor_pairs.append((a, b))
            joint_voxel_offsets.append(
                (
                    (jx - cx_a, jy - cy_a, jz - cz_a),
                    (jx - cx_b, jy - cy_b, jz - cz_b),
                )
            )
    return neighbor_pairs, joint_voxel_offsets


def _enumerate_joint_contacts(blocks, voxel_to_block, block_centers, grid_shape):
    pair_joints = _scan_block_faces(blocks, voxel_to_block, grid_shape)
    return _build_joint_offsets(pair_joints, block_centers)


def build_block_topology(blocks: list, grid_shape: tuple):
    voxel_to_block, block_centers, block_halves_voxel = _build_block_geometry(
        blocks, grid_shape
    )
    neighbor_pairs, joint_voxel_offsets = _enumerate_joint_contacts(
        blocks, voxel_to_block, block_centers, grid_shape
    )
    return block_centers, block_halves_voxel, neighbor_pairs, joint_voxel_offsets


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
    return unpack_voxel_bits(
        raw, resolution=res, n_words=words_z, word_axis=2, transpose=(1, 0, 2)
    )


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
