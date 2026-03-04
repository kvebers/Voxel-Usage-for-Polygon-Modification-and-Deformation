import math
import os
import subprocess
import time
from types import SimpleNamespace
import numpy as np
import warp as wp
import newton
from pygame.locals import *
import pygame
from OpenGL.GL import *
import moderngl
from src.shaders import *
from src.voxel_gpu import (
    voxelize_gpu,
    make_hollow,
    repair_isolated_voxels,
    greedy_merge_grid,
    build_block_topology,
)
from src.draw_helpers import *
from src.load_obj import load_obj
from src.renderer import Renderer
from src.joints import JointBreaker
from src.config import load_config
from src.mesh_split import MeshSplitter
from src.force_modes import ForceApplier
import argparse
from src.profiler import Profiler, NullProfiler


class VideoRecorder:
    def __init__(self, path, width, height, fps):
        self._path = path
        self._width = width
        self._height = height
        self._fps = fps
        self._proc = None
        self.active = False

    def start(self):
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self._width}x{self._height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(self._fps),
            "-i",
            "pipe:0",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            self._path,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.active = True
        print(f"[recorder] Recording → {self._path}")

    def write_frame(self, width, height):
        buf = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
        self._proc.stdin.write(np.flipud(frame).tobytes())

    def stop(self):
        if self._proc:
            self._proc.stdin.close()
            self._proc.wait()
            self._proc = None
        self.active = False
        print(f"[recorder] Saved {self._path}")


wp.init()
device = wp.get_cuda_device()


def voxelize_and_build_topology(
    mesh_verts,
    mesh_indices,
    resolution=32,
    fill_mode=True,
    ensure_connected=False,
    greedy_merge=False,
    ctx=None,
):
    tri_verts = mesh_verts[mesh_indices].reshape(-1, 3)
    grid_filled = voxelize_gpu(tri_verts, resolution=resolution, ctx=ctx)
    if fill_mode:
        grid_hollow = None
        active_grid = grid_filled
    else:
        grid_hollow = make_hollow(grid_filled)
        active_grid = grid_hollow
    if ensure_connected and fill_mode:
        active_grid = repair_isolated_voxels(active_grid, grid_filled)

    if greedy_merge:
        blocks = greedy_merge_grid(active_grid)
        block_centers, block_halves_voxel, neighbor_pairs, joint_voxel_offsets = (
            build_block_topology(blocks, active_grid.shape)
        )
        coords = [tuple(float(v) for v in c) for c in block_centers]
        coord_to_idx = {}
        return (
            grid_filled,
            grid_hollow,
            active_grid,
            coords,
            coord_to_idx,
            neighbor_pairs,
            block_halves_voxel,
            joint_voxel_offsets,
        )

    filled = np.argwhere(active_grid > 0)
    coords = [(int(ix), int(iy), int(iz)) for ix, iy, iz in filled]
    coord_to_idx = {c: i for i, c in enumerate(coords)}
    neighbor_pairs = []
    for ix, iy, iz in coords:
        for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            nb = (ix + dx, iy + dy, iz + dz)
            if nb in coord_to_idx:
                neighbor_pairs.append((coord_to_idx[(ix, iy, iz)], coord_to_idx[nb]))
    return (
        grid_filled,
        grid_hollow,
        active_grid,
        coords,
        coord_to_idx,
        neighbor_pairs,
        None,
        None,
    )


def bind_vertices_to_voxels(
    mesh_verts, coords, coord_to_idx, grid_min, voxel_size, resolution
):
    from scipy.spatial import KDTree

    coords_arr = np.array(coords, dtype=np.float32)
    voxel_centers = grid_min + (coords_arr + 0.5) * voxel_size
    _, nearest = KDTree(voxel_centers).query(mesh_verts)
    bindings = nearest.astype(np.int32)
    offsets = (mesh_verts - voxel_centers[bindings]).astype(np.float32)
    return bindings, offsets


def _compute_voxel_geometry(mesh_verts, resolution, cfg):
    pad = cfg.voxels.padding
    vmin = mesh_verts.min(axis=0)
    vmax = mesh_verts.max(axis=0)
    extent = (vmax - vmin).max()
    usable = 1.0 - 2.0 * pad
    voxel_size = (1.0 / resolution) / usable * extent
    half = voxel_size * 0.5
    grid_min = vmin - pad * extent / usable
    return extent, usable, voxel_size, half, grid_min


def _compute_voxel_positions(
    coords, resolution, cfg, mesh_verts, half, world_offset=(0.0, 0.0, 0.0)
):
    pad = cfg.voxels.padding
    vmin = mesh_verts.min(axis=0)
    extent = (mesh_verts.max(axis=0) - vmin).max()
    usable = 1.0 - 2.0 * pad
    coords_arr = np.array(coords, dtype=np.float64)
    if len(coords_arr) == 0:
        raise ValueError(f"Voxel Cord Err")
    n = (coords_arr + 0.5) / resolution
    p = (n - pad) / usable * extent + vmin
    raw = np.empty((len(coords_arr), 3), dtype=np.float64)
    raw[:, 0] = p[:, 0]
    raw[:, 1] = -p[:, 2]
    raw[:, 2] = p[:, 1]
    gnd = cfg.ground
    ground_top = gnd.position[2] + gnd.half_extents[2]
    z_offset = ground_top - raw[:, 2].min() + half
    ox, oy, oz = world_offset
    raw[:, 0] += ox
    raw[:, 1] += oy
    raw[:, 2] += z_offset + oz
    return raw


def _add_ground(builder, cfg):
    gnd = cfg.ground
    gx, gy, gz = gnd.position
    builder.add_body(xform=wp.transform(p=wp.vec3(gx, gy, gz), q=wp.quat_identity()))
    hx, hy, hz = gnd.half_extents
    ground_cfg = newton.ModelBuilder.ShapeConfig(
        density=gnd.density, ke=gnd.ke, kd=gnd.kd, kf=gnd.kf, mu=gnd.mu
    )
    builder.add_shape_box(
        body=builder.body_count - 1, hx=hx, hy=hy, hz=hz, cfg=ground_cfg
    )


def _add_voxel_bodies(builder, positions, half, cfg, block_halves_world=None):
    voxel_body_start = builder.body_count
    voxel_cfg = newton.ModelBuilder.ShapeConfig(
        density=cfg.voxels.density, kf=1e4, mu=1.0
    )
    q = wp.quat_identity()
    body_idx = voxel_body_start
    if block_halves_world is not None:
        for i, pos in enumerate(positions):
            builder.add_body(
                xform=wp.transform(
                    p=wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])), q=q
                )
            )
            hx, hy, hz = block_halves_world[i]
            builder.add_shape_box(body=body_idx, hx=hx, hy=hy, hz=hz, cfg=voxel_cfg)
            body_idx += 1
    else:
        for pos in positions:
            builder.add_body(
                xform=wp.transform(
                    p=wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])), q=q
                )
            )
            builder.add_shape_box(
                body=body_idx, hx=half, hy=half, hz=half, cfg=voxel_cfg
            )
            body_idx += 1
    return voxel_body_start


def _add_ball(builder, positions, half, extent, cfg):
    bcfg = cfg.ball
    if not getattr(bcfg, "enabled", True):
        return [], 0.0
    count = max(1, int(getattr(bcfg, "count", 1)))
    ball_radius = extent * bcfg.radius_factor
    max_z = float(positions[:, 2].max())
    ball_spawn_z = max_z + half + ball_radius + extent * bcfg.height_factor
    ball_shape_cfg = newton.ModelBuilder.ShapeConfig(
        density=bcfg.density, ke=bcfg.ke, kd=bcfg.kd, kf=bcfg.kf, mu=bcfg.mu
    )
    spacing = ball_radius * 3.0
    x_start = -(count - 1) * spacing * 0.5
    ball_bodies = []
    for i in range(count):
        x = x_start + i * spacing
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(x, 0.0, ball_spawn_z), q=wp.quat_identity())
        )
        builder.add_shape_sphere(body, radius=ball_radius, cfg=ball_shape_cfg)
        ball_bodies.append(body)
    return ball_bodies, ball_radius


def _add_joints(
    builder, neighbor_pairs, positions, voxel_body_start, joint_world_offsets=None
):
    if not neighbor_pairs:
        return
    q = wp.quat_identity()
    pairs = np.asarray(neighbor_pairs, dtype=np.int32)  # (N, 2)
    n = len(pairs)
    if joint_world_offsets is not None:
        wo = np.asarray(joint_world_offsets, dtype=np.float64)  # (N, 2, 3)
        oa_all = wo[:, 0]
        ob_all = wo[:, 1]
    else:
        oa_all = (positions[pairs[:, 1]] - positions[pairs[:, 0]]) * 0.5  # (N, 3)
        ob_all = -oa_all
    for ji in range(n):
        ba = voxel_body_start + int(pairs[ji, 0])
        bb = voxel_body_start + int(pairs[ji, 1])
        ox_a, oy_a, oz_a = (
            float(oa_all[ji, 0]),
            float(oa_all[ji, 1]),
            float(oa_all[ji, 2]),
        )
        ox_b, oy_b, oz_b = (
            float(ob_all[ji, 0]),
            float(ob_all[ji, 1]),
            float(ob_all[ji, 2]),
        )
        builder.add_joint_fixed(
            ba,
            bb,
            parent_xform=wp.transform(p=wp.vec3(ox_a, oy_a, oz_a), q=q),
            child_xform=wp.transform(p=wp.vec3(ox_b, oy_b, oz_b), q=q),
        )


def _create_solver(model, cfg):
    scfg = cfg.solver
    return newton.solvers.SolverVBD(
        model,
        iterations=scfg.iterations,
        rigid_body_contact_buffer_size=scfg.rigid_body_contact_buffer_size,
        rigid_contact_k_start=scfg.rigid_contact_k_start,
        rigid_avbd_beta=scfg.rigid_avbd_beta,
        rigid_joint_linear_ke=scfg.rigid_joint_linear_ke,
        rigid_joint_angular_ke=scfg.rigid_joint_angular_ke,
        rigid_joint_linear_kd=scfg.rigid_joint_linear_kd,
        rigid_joint_angular_kd=scfg.rigid_joint_angular_kd,
    )


def get_object_configs(cfg):
    if hasattr(cfg, "objects") and cfg.objects:
        objs = []
        for o in cfg.objects:
            objs.append(
                SimpleNamespace(
                    path=o.path,
                    offset=list(o.offset) if hasattr(o, "offset") else [0.0, 0.0, 0.0],
                    color=list(o.color)
                    if hasattr(o, "color")
                    else list(cfg.render.mesh_color),
                    resolution=o.resolution
                    if hasattr(o, "resolution")
                    else cfg.mesh.resolution,
                    rot=list(o.rot) if hasattr(o, "rot") else None,
                    scale=float(o.scale) if hasattr(o, "scale") else 1.0,
                )
            )
        return objs
    return [
        SimpleNamespace(
            path=cfg.mesh.path,
            offset=[0.0, 0.0, 0.0],
            color=list(cfg.render.mesh_color),
            resolution=cfg.mesh.resolution,
        )
    ]


def load_and_voxelize_one(cfg, obj_cfg, ctx=None):
    fill_mode = bool(getattr(cfg.voxels, "fill_mode", True))
    ensure_connected = bool(getattr(cfg.voxels, "ensure_connected", False))
    greedy_merge = bool(getattr(cfg.voxels, "greedy_merge", False))
    vertices, indices = load_obj(obj_cfg.path)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    centered_verts = vertices - center
    if obj_cfg.rot is not None:
        centered_verts = (np.array(obj_cfg.rot, dtype=np.float32) @ centered_verts.T).T
    if obj_cfg.scale != 1.0:
        centered_verts = centered_verts * obj_cfg.scale
    (
        _,
        _,
        _,
        coords,
        coord_to_idx,
        neighbor_pairs,
        block_halves_voxel,
        joint_voxel_offsets,
    ) = voxelize_and_build_topology(
        centered_verts,
        indices,
        obj_cfg.resolution,
        fill_mode=fill_mode,
        ensure_connected=ensure_connected,
        greedy_merge=greedy_merge,
        ctx=ctx,
    )
    return {
        "centered_verts": centered_verts,
        "indices": indices,
        "coords": coords,
        "coord_to_idx": coord_to_idx,
        "neighbor_pairs": neighbor_pairs,
        "block_halves_voxel": block_halves_voxel,
        "joint_voxel_offsets": joint_voxel_offsets,
        "resolution": obj_cfg.resolution,
        "offset": obj_cfg.offset,
        "color": obj_cfg.color,
    }


def build_scene_multi(all_obj_data, cfg):
    builder = newton.ModelBuilder()
    _add_ground(builder, cfg)

    per_obj = []
    joint_offset = 0
    first_extent = None
    all_positions_list = []

    for obj_data in all_obj_data:
        mesh_verts = obj_data["centered_verts"]
        coords = obj_data["coords"]
        coord_to_idx = obj_data["coord_to_idx"]
        neighbor_pairs = obj_data["neighbor_pairs"]
        block_halves_voxel = obj_data["block_halves_voxel"]
        joint_voxel_offsets = obj_data["joint_voxel_offsets"]
        resolution = obj_data["resolution"]
        world_offset = tuple(obj_data["offset"])

        extent, _, voxel_size, half, grid_min = _compute_voxel_geometry(
            mesh_verts, resolution, cfg
        )
        if first_extent is None:
            first_extent = (extent, half)

        positions = _compute_voxel_positions(
            coords, resolution, cfg, mesh_verts, half, world_offset=world_offset
        )
        bindings, offsets = bind_vertices_to_voxels(
            mesh_verts, coords, coord_to_idx, grid_min, voxel_size, resolution
        )
        block_halves_world = None
        joint_world_offsets = None
        if block_halves_voxel is not None:
            block_halves_world = [
                (hx * voxel_size, hz * voxel_size, hy * voxel_size)
                for hx, hy, hz in block_halves_voxel
            ]
        if joint_voxel_offsets is not None:
            joint_world_offsets = [
                (
                    (oa[0] * voxel_size, -oa[2] * voxel_size, oa[1] * voxel_size),
                    (ob[0] * voxel_size, -ob[2] * voxel_size, ob[1] * voxel_size),
                )
                for oa, ob in joint_voxel_offsets
            ]

        voxel_body_start = _add_voxel_bodies(
            builder, positions, half, cfg, block_halves_world=block_halves_world
        )
        print(f"[build_scene] Object {len(per_obj) + 1}: {len(positions)} voxels")
        if cfg.joints.enabled:
            _add_joints(
                builder,
                neighbor_pairs,
                positions,
                voxel_body_start,
                joint_world_offsets=joint_world_offsets,
            )

        all_positions_list.append(positions)
        per_obj.append(
            {
                "voxel_body_start": voxel_body_start,
                "voxel_count": len(coords),
                "positions": positions,
                "neighbor_pairs": neighbor_pairs,
                "bindings": bindings,
                "offsets": offsets,
                "half": half,
                "block_halves_world": block_halves_world,
                "joint_start": joint_offset,
            }
        )
        joint_offset += len(neighbor_pairs)

    extent0, half0 = first_extent
    all_positions = np.concatenate(all_positions_list, axis=0)
    ball_bodies, ball_radius = _add_ball(builder, all_positions, half0, extent0, cfg)

    builder.color()
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    solver = _create_solver(model, cfg)

    for obj in per_obj:
        obj["solver"] = solver

    return {
        "model": model,
        "state_0": state_0,
        "state_1": state_1,
        "control": model.control(),
        "contacts": model.contacts(),
        "solver": solver,
        "ball_bodies": ball_bodies,
        "ball_radius": ball_radius,
        "objects": per_obj,
    }


def create_joint_breaker(obj_scene, model, cfg):
    jcfg = cfg.joints
    return JointBreaker(
        obj_scene,
        model,
        linear_break_force=jcfg.linear_break_force,
        angular_break_torque=jcfg.angular_break_torque,
        damage_rate=jcfg.damage_rate,
        heal_rate=jcfg.heal_rate,
        instant_break_force=jcfg.instant_break_force,
        instant_break_torque=jcfg.instant_break_torque,
        max_breaks_per_step=jcfg.max_breaks_per_step,
    )


def create_mesh_splitter(indices, obj_scene, centered_verts, neighbor_pairs, cfg):
    return MeshSplitter(
        indices,
        obj_scene["bindings"],
        obj_scene["offsets"],
        obj_scene["voxel_count"],
        neighbor_pairs,
        obj_scene["positions"],
        obj_scene["half"],
        centered_verts,
        cfg,
    )


def init_window(cfg):
    wcfg = cfg.window
    width, height = wcfg.width, wcfg.height
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
    )
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | RESIZABLE)
    pygame.display.set_caption(wcfg.title)
    glEnable(GL_DEPTH_TEST)
    cc = wcfg.clear_color
    glClearColor(cc[0], cc[1], cc[2], cc[3])
    return width, height


def init_camera(cfg):
    ccfg = cfg.camera
    pos = getattr(ccfg, "position", None)
    return {
        "dist": ccfg.distance,
        "yaw": ccfg.yaw,
        "pitch": ccfg.pitch,
        "target": np.array(ccfg.target, dtype=np.float32),
        "fov": ccfg.fov,
        "near": ccfg.near,
        "far": ccfg.far,
        "position": np.array(pos, dtype=np.float32) if pos is not None else None,
    }


def init_sim_state(cfg):
    fps = cfg.simulation.fps
    frame_dt = 1.0 / fps
    fm_cfg = getattr(cfg, "force_modes", None)
    strength = fm_cfg.strength if fm_cfg is not None else 5000.0
    return {
        "simulating": not cfg.simulation.start_paused,
        "render_mode": cfg.render.default_mode,
        "fps": fps,
        "frame_dt": frame_dt,
        "substeps": cfg.simulation.substeps,
        "sim_dt": frame_dt / cfg.simulation.substeps,
        "sim_time": 0.0,
        "frame_count": 0,
        "force_mode": 0,
        "force_strength": strength,
    }


def handle_events(cam, sim, width, height):
    running = True
    mouse_dragging = getattr(handle_events, "_dragging", False)

    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
            elif event.key == K_c:
                sim["simulating"] = not sim["simulating"]
                print(f"Physics {'started' if sim['simulating'] else 'stopped'}")
            elif event.key == K_v:
                sim["render_mode"] = "voxel" if sim["render_mode"] == "mesh" else "mesh"
                print(f"Render mode: {sim['render_mode']}")
            elif event.key in (K_0, K_1, K_2, K_3, K_4):
                new_mode = event.key - K_0
                sim["force_mode"] = 0 if sim["force_mode"] == new_mode else new_mode
            elif event.key == K_EQUALS or event.key == K_PLUS:
                sim["force_strength"] *= 2.0
                print(f"Force strength: {sim['force_strength']:.1f} N")
            elif event.key == K_MINUS:
                sim["force_strength"] = max(1.0, sim["force_strength"] / 2.0)
                print(f"Force strength: {sim['force_strength']:.1f} N")
        elif event.type == VIDEORESIZE:
            width, height = event.w, event.h
            pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | RESIZABLE)
            glViewport(0, 0, width, height)
        elif event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_dragging = True
            elif event.button == 4:
                cam["dist"] = max(1.0, cam["dist"] - 0.5)
            elif event.button == 5:
                cam["dist"] += 0.5
        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:
                mouse_dragging = False
        elif event.type == MOUSEMOTION:
            if mouse_dragging:
                dx, dy = event.rel
                cam["yaw"] += dx * 0.3
                cam["pitch"] = max(-89, min(89, cam["pitch"] - dy * 0.3))

    handle_events._dragging = mouse_dragging
    return running, width, height


def pan_camera(cam):
    keys = pygame.key.get_pressed()
    if not any([keys[K_w], keys[K_s], keys[K_a], keys[K_d], keys[K_q], keys[K_e]]):
        return
    speed = cam["dist"] * 0.02
    yaw_r = math.radians(cam["yaw"])
    fwd = np.array([-math.cos(yaw_r), -math.sin(yaw_r), 0.0], np.float32)
    right = np.array([-math.sin(yaw_r), math.cos(yaw_r), 0.0], np.float32)
    up = np.array([0.0, 0.0, 1.0], np.float32)
    if keys[K_w]:
        cam["target"] += fwd * speed
    if keys[K_s]:
        cam["target"] -= fwd * speed
    if keys[K_d]:
        cam["target"] += right * speed
    if keys[K_a]:
        cam["target"] -= right * speed
    if keys[K_e]:
        cam["target"] += up * speed
    if keys[K_q]:
        cam["target"] -= up * speed


def step_simulation(
    scene, sim, joint_breakers, mesh_splitters, force_appliers, profiler=None
):
    state_0, state_1 = scene["state_0"], scene["state_1"]
    model = scene["model"]
    control, contacts, solver = scene["control"], scene["contacts"], scene["solver"]
    force_mode = sim["force_mode"]
    force_strength = sim["force_strength"]

    t_forces = t_collision = t_solver = t_joints = 0.0
    timing = profiler is not None

    for _ in range(sim["substeps"]):
        state_0.clear_forces()
        if timing:
            _t = time.perf_counter()
        for fa, jb in zip(force_appliers, joint_breakers):
            fa.apply_elastic(state_0, device, broken=jb.broken)
            if force_mode != 0:
                fa.apply(state_0, force_mode, force_strength, device)
        if timing:
            wp.synchronize()
            t_forces += time.perf_counter() - _t

        if timing:
            _t = time.perf_counter()
        contacts.clear()
        model.collide(state_0, contacts)
        if timing:
            wp.synchronize()
            t_collision += time.perf_counter() - _t

        if timing:
            _t = time.perf_counter()
        solver.step(state_0, state_1, control, contacts, sim["sim_dt"])
        if timing:
            wp.synchronize()
            t_solver += time.perf_counter() - _t

        state_0, state_1 = state_1, state_0

        if timing:
            _t = time.perf_counter()
        body_torques = (
            solver.body_torques.numpy()
        )  # one GPU read shared across all JointBreakers
        for jb in joint_breakers:
            jb.update(model, sim["sim_dt"], device, body_torques=body_torques)
        if timing:
            t_joints += time.perf_counter() - _t

    if profiler is not None:
        profiler.record("forces", t_forces)
        profiler.record("collision", t_collision)
        profiler.record("solver", t_solver)
        profiler.record("joints", t_joints)
        profiler.count("substeps", sim["substeps"])
        profiler.count(
            "active_joints", sum(int(np.sum(~jb.broken)) for jb in joint_breakers)
        )
        profiler.count(
            "broken_joints", sum(int(np.sum(jb.broken)) for jb in joint_breakers)
        )

    scene["state_0"], scene["state_1"] = state_0, state_1
    sim["sim_time"] += sim["frame_dt"]
    sim["frame_count"] += 1

    for jb, ms in zip(joint_breakers, mesh_splitters):
        if np.any(jb.broken):
            ms.set_broken(jb.broken)


def compute_view_projection(cam, width, height):
    aspect = width / max(height, 1)
    proj = perspective_matrix(cam["fov"], aspect, cam["near"], cam["far"])
    if cam.get("position") is not None:
        eye = cam["position"]
    else:
        yaw_r = math.radians(cam["yaw"])
        pitch_r = math.radians(cam["pitch"])
        eye = np.array(
            [
                cam["target"][0] + cam["dist"] * math.cos(pitch_r) * math.cos(yaw_r),
                cam["target"][1] + cam["dist"] * math.cos(pitch_r) * math.sin(yaw_r),
                cam["target"][2] + cam["dist"] * math.sin(pitch_r),
            ],
            dtype=np.float32,
        )
    view = look_at_matrix(eye, cam["target"], np.array([0, 0, 1], np.float32))
    return proj, view, eye


def _recover_nan_bodies(scene, transforms):
    finite = np.isfinite(transforms).all(axis=1)
    if finite.all():
        scene["_last_valid_body_q"] = transforms.copy()
        return False

    last_good = scene.get("_last_valid_body_q")
    if last_good is None:
        return False

    nan_idx = np.where(~finite)[0]
    transforms[nan_idx] = last_good[nan_idx]
    st0 = scene["state_0"]
    st0.body_q = wp.array(transforms, dtype=st0.body_q.dtype, device=device)
    qd0 = st0.body_qd.numpy()
    qd0[nan_idx] = 0.0
    st0.body_qd = wp.array(qd0, dtype=st0.body_qd.dtype, device=device)
    st1 = scene["state_1"]
    q1 = st1.body_q.numpy()
    q1[nan_idx] = last_good[nan_idx]
    st1.body_q = wp.array(q1, dtype=st1.body_q.dtype, device=device)
    qd1 = st1.body_qd.numpy()
    qd1[nan_idx] = 0.0
    st1.body_qd = wp.array(qd1, dtype=st1.body_qd.dtype, device=device)
    return True


def render_frame(renderer, scene, sim, obj_list, proj, view, eye, cfg, profiler=None):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    renderer.draw_ground(proj, view)
    transforms = scene["state_0"].body_q.numpy()
    _recover_nan_bodies(scene, transforms)
    t_deform = 0.0
    n_verts_deformed = 0
    n_faces_rendered = 0
    for obj in obj_list:
        obj_scene = obj["scene"]
        obj_renderer = obj["obj_renderer"]
        color = tuple(obj["color"])
        ms = obj["mesh_splitter"]
        jb = obj["joint_breaker"]
        vbs = obj_scene["voxel_body_start"]
        nv = obj_scene["voxel_count"]
        if sim["render_mode"] == "mesh":
            _t = time.perf_counter()
            ms.deform_split_mesh(transforms, vbs, nv)
            t_deform += time.perf_counter() - _t
            n_verts_deformed += ms.n_split_verts
            n_faces_rendered += ms.current_index_count // 3
            obj_renderer.update_voxel_transforms(ms.last_voxel_slice)
            obj_renderer.draw_mesh_mode(
                proj, view, eye, color, index_count=ms.current_index_count
            )
        else:
            obj_renderer.update_voxel_instances(transforms, vbs)
            obj_renderer.update_voxel_colors(jb.get_voxel_colors())
            obj_renderer.draw_voxels(proj, view, eye)

    if profiler is not None:
        profiler.record("mesh_deform", t_deform)
        profiler.count("verts_deformed", n_verts_deformed)
        profiler.count("faces_rendered", n_faces_rendered)

    renderer.update_ball_instances(
        transforms, scene["ball_bodies"], scene["ball_radius"]
    )
    renderer.draw_ball(
        proj, view, eye, tuple(cfg.ball.color), count=len(scene["ball_bodies"])
    )


def main(scene_file: str = "scene.json", profile: bool = False, record: str = None):
    setup = Profiler() if profile else NullProfiler()
    profiler = Profiler() if profile else NullProfiler()
    with setup.section("load_config"):
        cfg = load_config(scene_file)
        obj_cfgs = get_object_configs(cfg)
    with setup.section("voxelize"):
        vox_ctx = moderngl.create_standalone_context()
        try:
            all_obj_data = [
                load_and_voxelize_one(cfg, oc, ctx=vox_ctx) for oc in obj_cfgs
            ]
        finally:
            vox_ctx.release()
    with setup.section("build_scene"):
        scene = build_scene_multi(all_obj_data, cfg)
    obj_list = []
    joint_breakers = []
    mesh_splitters = []
    force_appliers = []

    with setup.section("create_components"):
        for i, (obj_data, obj_scene) in enumerate(zip(all_obj_data, scene["objects"])):
            jb = create_joint_breaker(obj_scene, scene["model"], cfg)
            ms = create_mesh_splitter(
                obj_data["indices"],
                obj_scene,
                obj_data["centered_verts"],
                obj_scene["neighbor_pairs"],
                cfg,
            )
            ms.set_broken(jb.broken)
            el_cfg = getattr(cfg, "elasticity", None)
            _half = obj_scene["half"]
            _voxel_mass = cfg.voxels.density * 8.0 * _half**3
            fa = ForceApplier(
                obj_scene["positions"],
                obj_scene["voxel_body_start"],
                obj_scene["voxel_count"],
                stiffness=el_cfg.stiffness if el_cfg else 0.0,
                damping=el_cfg.damping if el_cfg else 0.0,
                voxel_mass=_voxel_mass,
                neighbor_pairs=obj_scene["neighbor_pairs"],
            )
            obj_list.append(
                {
                    "scene": obj_scene,
                    "mesh_splitter": ms,
                    "joint_breaker": jb,
                    "color": obj_data["color"],
                }
            )
            joint_breakers.append(jb)
            mesh_splitters.append(ms)
            force_appliers.append(fa)

    with setup.section("init_renderer"):
        width, height = init_window(cfg)
        renderer = Renderer(
            scene["ball_radius"], max_balls=max(1, len(scene["ball_bodies"]))
        )

        for obj in obj_list:
            obj_scene = obj["scene"]
            i = obj_list.index(obj)
            obj_renderer = renderer.create_object_renderer(
                all_obj_data[i]["indices"],
                obj_scene["voxel_count"],
                obj_scene["half"],
                block_halves=obj_scene["block_halves_world"],
            )
            obj_renderer.setup_gpu_deform(obj["mesh_splitter"])
            obj["obj_renderer"] = obj_renderer

    cam = init_camera(cfg)
    sim = init_sim_state(cfg)
    clock = pygame.time.Clock()
    recorder = VideoRecorder(record, width, height, sim["fps"]) if record else None
    was_simulating = sim["simulating"]
    running = True
    while running:
        with profiler.section("frame"):
            with profiler.section("events"):
                running, width, height = handle_events(cam, sim, width, height)
                pan_camera(cam)
            if (
                recorder
                and not recorder.active
                and sim["simulating"]
                and not was_simulating
            ):
                recorder.start()
            elif (
                recorder
                and recorder.active
                and not sim["simulating"]
                and was_simulating
            ):
                recorder.stop()
            was_simulating = sim["simulating"]

            if sim["simulating"]:
                with profiler.section("simulation"):
                    step_simulation(
                        scene,
                        sim,
                        joint_breakers,
                        mesh_splitters,
                        force_appliers,
                        profiler,
                    )

            proj, view, eye = compute_view_projection(cam, width, height)

            with profiler.section("render"):
                render_frame(
                    renderer, scene, sim, obj_list, proj, view, eye, cfg, profiler
                )
            if recorder and recorder.active:
                recorder.write_frame(width, height)
            pygame.display.flip()

        clock.tick(sim["fps"])
    if recorder and recorder.active:
        recorder.stop()
    setup.save_summary_csv("profile_setup.csv")
    profiler.save_csv("profile_frames.csv")
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scene",
        nargs="?",
        default="scene.jsonc",
        help="Scene config file to load (default: scene.jsonc)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect timing and operation counts; write CSVs on exit.",
    )
    parser.add_argument(
        "--record",
        nargs="?",
        const=True,
        metavar="OUTPUT.mp4",
        help="Record to video when play is pressed (default: <scene>.mp4)",
    )
    args = parser.parse_args()
    if args.record is True:
        record_path = os.path.splitext(os.path.basename(args.scene))[0] + ".mp4"
    else:
        record_path = args.record
    main(scene_file=args.scene, profile=args.profile, record=record_path)
