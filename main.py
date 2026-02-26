import sys, time, math, json, argparse
import numpy as np
import warp as wp
import newton
from voxel_gpu import voxelize_gpu

wp.init()
device = wp.get_cuda_device()


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


def voxelize_mesh(vertices, indices, resolution=32):
    tri_verts = vertices[indices].reshape(-1, 3)
    grid = voxelize_gpu(tri_verts, resolution=resolution)
    return grid


HIDDEN_POS = wp.vec3(0.0, -1000.0, 0.0)

def build_combined_scene(mesh_verts, mesh_indices, resolution=32):
    """Build a single scene with both the mesh body and all voxel bodies."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    # --- Mesh body ---
    mesh = newton.Mesh(
        vertices=mesh_verts.tolist(),
        indices=mesh_indices.tolist(),
    )

    mesh_xform = wp.transform(
        p=wp.vec3(0.0, 0.0, 1.5),
        q=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.radians(90.0)),
    )

    builder.add_body(xform=mesh_xform)
    mesh_body = builder.body_count - 1
    builder.add_shape_mesh(body=mesh_body, mesh=mesh)
    grid = voxelize_mesh(mesh_verts, mesh_indices, resolution)
    pad = 0.01
    vmin = mesh_verts.min(axis=0)
    vmax = mesh_verts.max(axis=0)
    extent = (vmax - vmin).max() 
    usable = 1.0 - 2.0 * pad
    voxel_size_norm = 1.0 / resolution
    half_world = (voxel_size_norm / usable * extent) * 0.5

    filled = np.argwhere(grid > 0)
    print(f"Voxelized: {len(filled)} filled voxels out of {resolution}^3")

    voxel_bodies = []
    voxel_positions = []

    for ix, iy, iz in filled:
        nx = (ix + 0.5) / resolution
        ny = (iy + 0.5) / resolution
        nz = (iz + 0.5) / resolution
        px = (nx - pad) / usable * extent + vmin[0]
        py = (ny - pad) / usable * extent + vmin[1]
        pz = (nz - pad) / usable * extent + vmin[2]
        rx, ry, rz = px, -pz, py
        real_pos = wp.vec3(rx, ry, rz + 1.5)
        builder.add_body(
            xform=wp.transform(p=HIDDEN_POS, q=wp.quat_identity()),
        )
        body = builder.body_count - 1
        builder.add_shape_box(
            body=body,
            hx=float(half_world),
            hy=float(half_world),
            hz=float(half_world),
        )
        voxel_bodies.append(body)
        voxel_positions.append(real_pos)

    return builder, mesh_body, voxel_bodies, voxel_positions, mesh_xform


def set_body_position(state, body_index, pos, rot=wp.quat_identity()):
    """Move a single body by writing into the state transform array."""
    transforms = state.body_q.numpy()
    # body_q stores 7 floats per body: px, py, pz, qx, qy, qz, qw
    transforms[body_index] = [pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]]
    state.body_q = wp.array(transforms, dtype=wp.transform, device=device)


def show_mesh_hide_voxels(state, mesh_body, mesh_xform, voxel_bodies):
    transforms = state.body_q.numpy()

    # Show mesh at its real position
    p = mesh_xform.p
    q = mesh_xform.q
    transforms[mesh_body] = [p[0], p[1], p[2], q[0], q[1], q[2], q[3]]

    # Hide all voxels
    hp = HIDDEN_POS
    qi = wp.quat_identity()
    for b in voxel_bodies:
        transforms[b] = [hp[0], hp[1], hp[2], qi[0], qi[1], qi[2], qi[3]]

    state.body_q = wp.array(transforms, dtype=wp.transform, device=device)


def show_voxels_hide_mesh(state, mesh_body, voxel_bodies, voxel_positions):
    transforms = state.body_q.numpy()

    # Hide mesh
    hp = HIDDEN_POS
    qi = wp.quat_identity()
    transforms[mesh_body] = [hp[0], hp[1], hp[2], qi[0], qi[1], qi[2], qi[3]]

    # Show voxels at their real positions
    for b, pos in zip(voxel_bodies, voxel_positions):
        transforms[b] = [pos[0], pos[1], pos[2], qi[0], qi[1], qi[2], qi[3]]

    state.body_q = wp.array(transforms, dtype=wp.transform, device=device)


def main():
    # Load mesh
    vertices, indices = load_obj("obj/teapot.obj")
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) * 0.5
    centered_verts = vertices - center

    # Build combined scene (one model, one set_model call)
    builder, mesh_body, voxel_bodies, voxel_positions, mesh_xform = build_combined_scene(
        centered_verts, indices, resolution=32
    )

    model = builder.finalize(device=device)
    state = model.state()

    # Set initial state: show mesh, hide voxels
    show_mesh_hide_voxels(state, mesh_body, mesh_xform, voxel_bodies)

    renderer = newton.viewer.ViewerGL(width=1280, height=720)
    renderer.set_model(model)  # called only once

    showing_voxels = False
    x_was_pressed = False

    print("Press 'X' to toggle between Mesh and Voxels.")

    while renderer.is_running():
        x_pressed = renderer.is_key_down('x')

        if x_pressed and not x_was_pressed:
            showing_voxels = not showing_voxels

            if showing_voxels:
                print("Switching to Voxel View...")
                show_voxels_hide_mesh(state, mesh_body, voxel_bodies, voxel_positions)
            else:
                print("Switching to Mesh View...")
                show_mesh_hide_voxels(state, mesh_body, mesh_xform, voxel_bodies)

        x_was_pressed = x_pressed

        renderer.begin_frame(0.0)
        renderer.log_state(state)
        renderer.end_frame()

    renderer.close()


if __name__ == "__main__":
    main()