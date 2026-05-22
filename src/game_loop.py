import time
import numpy as np
import pygame
import moderngl
from OpenGL.GL import *
from src.config import load_config
from src.scene_setup import get_object_configs, load_and_voxelize_one
from src.physics_world import build_scene_multi
from src.simulation import step_simulation
from src.camera import compute_view_projection, movement
from src.input_handler import handle_events


def _render_single_object(obj, transforms, sim, proj, view, eye):
    obj_scene = obj["scene"]
    obj_renderer = obj["obj_renderer"]
    ms = obj["mesh_splitter"]
    vbs = obj_scene["voxel_body_start"]
    nv = obj_scene["voxel_count"]
    if sim["render_mode"] == "mesh":
        _t = time.perf_counter()
        ms.deform_split_mesh(transforms, vbs, nv)
        t_deform = time.perf_counter() - _t
        obj_renderer.update_voxel_transforms(ms.last_voxel_slice)
        obj_renderer.draw_mesh_mode(
            proj, view, eye, tuple(obj["color"]), index_count=ms.current_index_count
        )
        return t_deform, ms.n_split_verts, ms.current_index_count // 3
    else:
        obj_renderer.update_voxel_instances(transforms, vbs)
        obj_renderer.update_voxel_colors(obj["joint_breaker"].get_voxel_colors())
        obj_renderer.draw_voxels(proj, view, eye)
        return 0.0, 0, 0


def render_frame(renderer, scene, sim, obj_list, proj, view, eye, cfg, profiler=None):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    renderer.draw_ground(proj, view)
    transforms = scene["state_0"].body_q.numpy()
    t_deform = 0.0
    n_verts_deformed = 0
    n_faces_rendered = 0
    for obj in obj_list:
        td, nv, nf = _render_single_object(obj, transforms, sim, proj, view, eye)
        t_deform += td
        n_verts_deformed += nv
        n_faces_rendered += nf
    if profiler is not None:
        profiler.record("mesh_deform", t_deform)
        profiler.count("verts_deformed", n_verts_deformed)
        profiler.count("faces_rendered", n_faces_rendered)
    renderer.update_ball_instances(transforms, scene["ball_bodies"], scene["ball_radius"])
    renderer.draw_ball(proj, view, eye, tuple(cfg.ball.color), count=len(scene["ball_bodies"]))


def _load_and_build_scene(scene_file, setup):
    with setup.section("load_config"):
        cfg = load_config(scene_file)
        obj_cfgs = get_object_configs(cfg)
    with setup.section("voxelize"):
        vox_ctx = moderngl.create_standalone_context()
        try:
            all_obj_data = [load_and_voxelize_one(cfg, oc, ctx=vox_ctx) for oc in obj_cfgs]
        finally:
            vox_ctx.release()
    with setup.section("build_scene"):
        scene = build_scene_multi(all_obj_data, cfg)
    return cfg, all_obj_data, scene


def _update_recorder(recorder, sim, was_simulating):
    if recorder and not recorder.active and sim["simulating"] and not was_simulating:
        recorder.start()
    elif recorder and recorder.active and not sim["simulating"] and was_simulating:
        recorder.stop()


def _run_game_loop(
    scene, sim, cam, cfg, obj_list, joint_breakers, mesh_splitters,
    force_appliers, renderer, profiler, recorder, width, height,
):
    clock = pygame.time.Clock()
    was_simulating = sim["simulating"]
    running = True
    while running:
        with profiler.section("frame"):
            with profiler.section("events"):
                running, width, height = handle_events(cam, sim, width, height)
                movement(cam)
            _update_recorder(recorder, sim, was_simulating)
            was_simulating = sim["simulating"]
            if sim["simulating"]:
                with profiler.section("simulation"):
                    step_simulation(scene, sim, joint_breakers, mesh_splitters, force_appliers, profiler)
            proj, view, eye = compute_view_projection(cam, width, height)
            with profiler.section("render"):
                render_frame(renderer, scene, sim, obj_list, proj, view, eye, cfg, profiler)
            if recorder and recorder.active:
                recorder.write_frame(width, height)
            pygame.display.flip()
        clock.tick(sim["fps"])
    if recorder and recorder.active:
        recorder.stop()
