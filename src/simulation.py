import time
import numpy as np
import warp as wp


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


def _record_substep_stats(profiler, t_forces, t_collision, t_solver, t_joints, sim, joint_breakers):
    profiler.record("forces", t_forces)
    profiler.record("collision", t_collision)
    profiler.record("solver", t_solver)
    profiler.record("joints", t_joints)
    profiler.count("substeps", sim["substeps"])
    profiler.count("active_joints", sum(int(np.sum(~jb.broken)) for jb in joint_breakers))
    profiler.count("broken_joints", sum(int(np.sum(jb.broken)) for jb in joint_breakers))


def _sync_mesh_splitters(joint_breakers, mesh_splitters):
    for jb, ms in zip(joint_breakers, mesh_splitters):
        if np.any(jb.broken):
            ms.set_broken(jb.broken)


def step_simulation(scene, sim, joint_breakers, mesh_splitters, force_appliers, profiler=None):
    state_0, state_1 = scene["state_0"], scene["state_1"]
    model = scene["model"]
    control, contacts, solver = scene["control"], scene["contacts"], scene["solver"]
    force_mode = sim["force_mode"]
    force_strength = sim["force_strength"]
    device = wp.get_cuda_device()

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
        body_torques = solver.body_torques.numpy()
        for jb in joint_breakers:
            jb.update(model, sim["sim_dt"], device, body_torques=body_torques)
        if timing:
            t_joints += time.perf_counter() - _t

    if profiler is not None:
        _record_substep_stats(profiler, t_forces, t_collision, t_solver, t_joints, sim, joint_breakers)
    scene["state_0"], scene["state_1"] = state_0, state_1
    sim["sim_time"] += sim["frame_dt"]
    sim["frame_count"] += 1
    _sync_mesh_splitters(joint_breakers, mesh_splitters)
