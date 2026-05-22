import numpy as np
from scipy.ndimage import label, binary_erosion
from scipy.spatial import KDTree


def _make_struct6():
    struct6 = np.zeros((3, 3, 3), dtype=bool)
    for idx in [(1,1,1),(0,1,1),(2,1,1),(1,0,1),(1,2,1),(1,1,0),(1,1,2)]:
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
    struct6, _ = _make_struct6()
    deep_interior = binary_erosion(grid, structure=struct6, iterations=min_thickness)
    shell = (grid & ~deep_interior).copy()
    return repair_isolated_voxels(shell, grid)
