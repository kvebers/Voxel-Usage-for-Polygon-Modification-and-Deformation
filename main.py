import sys, time, math
import numpy as np
import warp as wp
import newton
from voxel_gpu import voxelize_gpu

wp.init()
device = wp.get_cuda_device()

RESOLUTION = 32


def load_obj(path):
    vertices = []
    indices = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                face_verts = []
                for p in parts[1:]:
                    face_verts.append(int(p.split("/")[0]) - 1)
                for i in range(1, len(face_verts) - 1):
                    indices.extend([face_verts[0], face_verts[i], face_verts[i + 1]])

    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.int32)

    clean_indices = []
    for i in range(0, len(indices), 3):
        i0, i1, i2 = indices[i], indices[i + 1], indices[i + 2]
        if i0 == i1 or i1 == i2 or i0 == i2:
            continue
        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
        if np.linalg.norm(np.cross(v1 - v0, v2 - v0)) < 1e-8:
            continue
        clean_indices.extend([i0, i1, i2])

    return vertices, np.array(clean_indices, dtype=np.int32)


def voxelize_and_build_topology(mesh_verts, mesh_indices, resolution=32):
    tri_verts = mesh_verts[mesh_indices].reshape(-1, 3)
    grid = voxelize_gpu(tri_verts, resolution=resolution)

    filled = np.argwhere(grid > 0)
    coords = [(int(ix), int(iy), int(iz)) for ix, iy, iz in filled]
    coord_to_idx = {c: i for i, c in enumerate(coords)}

    neighbor_pairs = []
    for ix, iy, iz in coords:
        for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            nb = (ix + dx, iy + dy, iz + dz)
            if nb in coord_to_idx:
                neighbor_pairs.append((coord_to_idx[(ix, iy, iz)], coord_to_idx[nb]))

    print(f"Voxelized: {len(coords)} filled, {len(neighbor_pairs)} neighbor pairs")
    return grid, coords, coord_to_idx, neighbor_pairs


def break_stressed_joints(model, state, scene, break_threshold=0.5):
    """Check joint stress and break overstressed joints."""
    transforms = state.body_q.numpy()
    vbs = scene["voxel_body_start"]
    neighbor_pairs = scene["neighbor_pairs"]
    positions = scene["positions"]  # original positions
    
    # Get per-joint stiffness array
    ke = model.joint_target_ke.numpy()
    
    broken_count = 0
    for joint_idx, (ia, ib) in enumerate(neighbor_pairs):
        # Skip already broken joints
        if ke[joint_idx] == 0.0:
            continue
            
        body_a = vbs + ia
        body_b = vbs + ib
        
        # Current positions
        pos_a = transforms[body_a][:3]
        pos_b = transforms[body_b][:3]
        
        # Original distance
        orig_a = positions[ia]
        orig_b = positions[ib]
        orig_dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(orig_a, orig_b)))
        
        # Current distance
        curr_dist = np.linalg.norm(pos_a - pos_b)
        
        # Strain = how much the distance changed
        strain = abs(curr_dist - orig_dist) / max(orig_dist, 1e-6)
        
        if strain > break_threshold:
            ke[joint_idx] = 0.0
            broken_count += 1
    
    if broken_count > 0:
        model.joint_target_ke = wp.array(ke, dtype=model.joint_target_ke.dtype, device=device)
        print(f"  Broke {broken_count} joints (total broken: {np.sum(ke == 0.0)})")
    
    return broken_count


def build_scene(mesh_verts, mesh_indices, resolution=32):
    print("=== Building scene ===")

    t0 = time.time()
    grid, coords, coord_to_idx, neighbor_pairs = voxelize_and_build_topology(
        mesh_verts, mesh_indices, resolution
    )
    print(f"  Topology: {time.time() - t0:.3f}s")

    # Voxel world positions
    pad = 0.01
    vmin = mesh_verts.min(axis=0)
    vmax = mesh_verts.max(axis=0)
    extent = (vmax - vmin).max()
    usable = 1.0 - 2.0 * pad
    voxel_size = (1.0 / resolution) / usable * extent
    half = voxel_size * 0.5

    positions = []
    for ix, iy, iz in coords:
        nx = (ix + 0.5) / resolution
        ny = (iy + 0.5) / resolution
        nz = (iz + 0.5) / resolution
        px = (nx - pad) / usable * extent + vmin[0]
        py = (ny - pad) / usable * extent + vmin[1]
        pz = (nz - pad) / usable * extent + vmin[2]
        # Rotate 90 deg around X: (x, y, z) -> (x, -z, y)
        rx, ry, rz = px, -pz, py
        positions.append((rx, ry, rz + 1.5))

    # Build model
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    # Voxel bodies with explicit mass
# Voxel bodies — let shape density handle mass
# Voxel bodies — let shape density handle mass
    voxel_body_start = builder.body_count
    voxel_cfg = newton.ModelBuilder.ShapeConfig(density=20000.0)  # lighter than default

    for px, py, pz in positions:
        builder.add_body(
            xform=wp.transform(p=wp.vec3(px, py, pz), q=wp.quat_identity()),
        )
        builder.add_shape_box(body=builder.body_count - 1, hx=half, hy=half, hz=half, cfg=voxel_cfg)
    # Fixed joints between neighbors with collision_filter_parent
    joint_count = 0
    for ia, ib in neighbor_pairs:
        body_a = voxel_body_start + ia
        body_b = voxel_body_start + ib

        # Compute offset from A to B
        pa = positions[ia]
        pb = positions[ib]
        dx = (pb[0] - pa[0]) * 0.5
        dy = (pb[1] - pa[1]) * 0.5
        dz = (pb[2] - pa[2]) * 0.5

        builder.add_joint_fixed(
            body_a, body_b,
            parent_xform=wp.transform(
                p=wp.vec3(dx, dy, dz), q=wp.quat_identity()),
            child_xform=wp.transform(
                p=wp.vec3(-dx, -dy, -dz), q=wp.quat_identity()),
            collision_filter_parent=True,
        )
        joint_count += 1

    print(f"  {joint_count} fixed joints added")

    # Color for VBD
    builder.color()

    # Finalize
    model = builder.finalize(device=device)

    # Initialize state from joint configuration
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    control = model.control()
    contacts = model.contacts()

    # Solver with appropriate buffer size
    n_bodies = model.body_count
    buf_size = max(256, n_bodies * 2)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=10,
        rigid_body_contact_buffer_size=buf_size,
        rigid_contact_k_start=1e2,
        rigid_avbd_beta=1e3,
        rigid_joint_linear_ke=1e6,    # linear joint stiffness (lower = weaker)
        rigid_joint_angular_ke=1e6,   # angular joint stiffness (lower = weaker)
        rigid_joint_linear_kd=10.0,   # linear damping
        rigid_joint_angular_kd=10.0,  # angular damping
    )

    print(f"  Total bodies: {model.body_count}")
    print(f"  Contact buffer: {buf_size}")
    print("=== Scene ready ===\n")

    return {
        "model": model,
        "state_0": state_0,
        "state_1": state_1,
        "control": control,
        "contacts": contacts,
        "solver": solver,
        "voxel_body_start": voxel_body_start,
        "voxel_count": len(coords),
        "positions": positions,
        "neighbor_pairs": neighbor_pairs,
        "coords": coords,
        "coord_to_idx": coord_to_idx,
        "voxel_size": voxel_size,
    }


def main():
    vertices, indices = load_obj("obj/teapot.obj")
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    centered_verts = vertices - center

    scene = build_scene(centered_verts, indices, RESOLUTION)

    model = scene["model"]
    state_0 = scene["state_0"]
    state_1 = scene["state_1"]
    control = scene["control"]
    contacts = scene["contacts"]
    solver = scene["solver"]

    renderer = newton.viewer.ViewerGL(width=1280, height=720)
    renderer.set_model(model)

    simulating = False
    s_was_pressed = False

    fps = 60
    frame_dt = 1.0 / fps
    sim_substeps = 4
    sim_dt = frame_dt / sim_substeps
    sim_time = 0.0

    while renderer.is_running():
        s_pressed = renderer.is_key_down('c')

        if s_pressed and not s_was_pressed:
            simulating = not simulating
            print(f"Physics {'started' if simulating else 'stopped'}")

        s_was_pressed = s_pressed

        if simulating:
            for _ in range(sim_substeps):
                state_0.clear_forces()
                contacts.clear()
                model.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, sim_dt)
                state_0, state_1 = state_1, state_0
            sim_time += frame_dt
            
            # Check and break stressed joints every frame
            break_stressed_joints(model, state_0, scene, break_threshold=0.1)
        renderer.begin_frame(sim_time)
        renderer.log_state(state_0)
        renderer.end_frame()

    renderer.close()


if __name__ == "__main__":
    main()