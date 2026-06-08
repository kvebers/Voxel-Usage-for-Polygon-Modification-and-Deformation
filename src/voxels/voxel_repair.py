import numpy as np
from scipy.ndimage import label, binary_erosion
from scipy.spatial import KDTree


def poly_struct():
    """
    Create 3x3 grid
    """
    face_struct = np.zeros((3, 3, 3), dtype=bool)
    for index in [(1, 1, 1), (0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)]:
        face_struct[index] = True
    kernel = face_struct.copy()
    kernel[1, 1, 1] = False
    return face_struct, kernel


def seek_gaps(labeled, main_label, main_coords):
    """
    Find closes point between 2 islands and returns it
    """
    main_tree = KDTree(main_coords)
    component_count = int(labeled.max())
    best_dist = np.inf
    best_src = best_dst = None
    for component_label in range(1, component_count + 1):  # TODO optimize
        if component_label == main_label:
            continue
        comp_coords = np.argwhere(labeled == component_label)
        dists, indexs = main_tree.query(comp_coords)
        min_dist_index = int(dists.argmin())
        if dists[min_dist_index] < best_dist:
            best_dist = dists[min_dist_index]
            best_src = comp_coords[min_dist_index]
            best_dst = main_coords[indexs[min_dist_index]]
    return best_src, best_dst


def repair_isolated_voxels(active, pool):
    face_struct, _ = poly_struct()
    result = active.astype(bool).copy()
    while True:
        labeled, component_count = label(result, structure=face_struct)
        if component_count <= 1:
            # if 1 island
            break
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        main_label = int(sizes.argmax())
        main_coords = np.argwhere(labeled == main_label)
        best_src, best_dst = seek_gaps(labeled, main_label, main_coords)
        pos = list(best_src)
        # goes trough the best position and makes a line in that direction
        for axis in range(3):
            while pos[axis] != best_dst[axis]:
                pos[axis] += 1 if best_dst[axis] > pos[axis] else -1
                result[pos[0], pos[1], pos[2]] = True
    return result


def make_hollow(grid, min_thickness=2):
    face_struct, _ = poly_struct()
    # make the grid more shallow by thickness of 2 to ensure that connections are still there, less job later
    mid_section = binary_erosion(grid, structure=face_struct, iterations=min_thickness)
    shell = (grid & ~mid_section).copy()
    return repair_isolated_voxels(shell, grid)  # fix the disconnected parts
