import numpy as np
import warp as wp


def init_sim_state(config):
    fps = config.simulation.fps
    frame_dt = 1.0 / fps
    return {
        "simulating": not config.simulation.start_paused,
        "render_mode": config.render.default_mode,
        "fps": fps,
        "frame_dt": frame_dt,
        "sim_dt": frame_dt,
        "sim_time": 0.0,
        "frame_count": 0,
    }


def assisted_mesh_splitters(joint_breakers, mesh_splitters):
    for joint_breaker, mesh_splitter in zip(joint_breakers, mesh_splitters):
        if np.any(joint_breaker.broken):
            mesh_splitter.set_broken(joint_breaker.broken)


def step_simulation(scene, sim, joint_breakers, mesh_splitters, force_appliers):
    state_current, state_next = scene["state_current"], scene["state_next"]
    model = scene["model"]
    control, contacts, solver = (
        scene["control"],
        scene["contacts"],
        scene["solver"],
    )
    device = wp.get_cuda_device()
    state_current.clear_forces()
    for force_applier, joint_breaker in zip(force_appliers, joint_breakers):
        force_applier.apply_elastic(state_current, device, broken=joint_breaker.broken)
    contacts.clear()
    model.collide(state_current, contacts)
    solver.step(state_current, state_next, control, contacts, sim["sim_dt"])
    state_current, state_next = state_next, state_current
    body_torques = solver.body_torques.numpy()
    for joint_breaker in joint_breakers:
        joint_breaker.update(model, sim["sim_dt"], device, body_torques=body_torques)
    scene["state_current"], scene["state_next"] = state_current, state_next
    sim["sim_time"] += sim["frame_dt"]
    sim["frame_count"] += 1
    assisted_mesh_splitters(joint_breakers, mesh_splitters)
