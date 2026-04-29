import numpy as np


def _parse_vertices(v_parts):
    if not v_parts:
        return np.empty((0, 3), dtype=np.float32)
    raw = np.fromstring(" ".join(v_parts), dtype=np.float32, sep=" ")
    n_col = round(len(raw) / len(v_parts))
    return raw.reshape(len(v_parts), n_col)[:, :3].copy()


def _parse_faces_fast(f_parts, n_verts):
    raw_f = np.fromstring(" ".join(f_parts), dtype=np.int32, sep=" ")
    neg = raw_f < 0
    raw_f[neg] += n_verts
    raw_f[~neg] -= 1
    n_per = round(len(raw_f) / len(f_parts))
    if n_per == 3:
        return raw_f
    if n_per == 4:
        faces = raw_f.reshape(-1, 4)
        tris = np.empty((len(faces) * 2, 3), dtype=np.int32)
        tris[0::2] = faces[:, [0, 1, 2]]
        tris[1::2] = faces[:, [0, 2, 3]]
        return tris.reshape(-1)
    faces = raw_f.reshape(-1, n_per)
    n_tris = n_per - 2
    tris = np.empty((len(faces) * n_tris, 3), dtype=np.int32)
    for i in range(n_tris):
        tris[i::n_tris, 0] = faces[:, 0]
        tris[i::n_tris, 1] = faces[:, i + 1]
        tris[i::n_tris, 2] = faces[:, i + 2]
    return tris.reshape(-1)


def _parse_faces_slow(f_parts, f_vcounts):
    idx_list = []
    for line, vc in zip(f_parts, f_vcounts):
        parts = line.split()
        fv = [int(p.split("/")[0]) for p in parts]
        fv = [i - 1 if i > 0 else vc + i for i in fv]
        for i in range(1, len(fv) - 1):
            idx_list.extend([fv[0], fv[i], fv[i + 1]])
    return np.array(idx_list, dtype=np.int32)


def _normalize_vertices(vertices):
    if len(vertices):
        vertices -= vertices.mean(axis=0)
        max_dist = np.linalg.norm(vertices, axis=1).max()
        if max_dist > 0:
            vertices /= max_dist
    return vertices


def _filter_degenerate_faces(vertices, indices):
    if len(indices) < 3:
        return indices
    tri = indices.reshape(-1, 3)
    i0, i1, i2 = tri[:, 0], tri[:, 1], tri[:, 2]
    no_dup = ~((i0 == i1) | (i1 == i2) | (i0 == i2))
    v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
    cross = np.cross(v1 - v0, v2 - v0)
    no_degen = np.einsum("ij,ij->i", cross, cross) >= 1e-16
    return tri[no_dup & no_degen].reshape(-1)


def load_obj(path: str):
    with open(path, "r") as f:
        content = f.read()
    v_parts = []
    f_parts = []
    f_vcounts = []

    for line in content.splitlines():
        if line.startswith("v "):
            v_parts.append(line[2:])
        elif line.startswith("f "):
            f_parts.append(line[2:])
            f_vcounts.append(len(v_parts))

    vertices = _parse_vertices(v_parts)
    n_verts = len(v_parts)

    if not f_parts:
        indices = np.empty(0, dtype=np.int32)
    elif "/" not in f_parts[0] and (not f_vcounts or f_vcounts[0] == n_verts):
        indices = _parse_faces_fast(f_parts, n_verts)
    else:
        indices = _parse_faces_slow(f_parts, f_vcounts)

    vertices = _normalize_vertices(vertices)
    indices = _filter_degenerate_faces(vertices, indices)
    return vertices, indices.astype(np.int32)
