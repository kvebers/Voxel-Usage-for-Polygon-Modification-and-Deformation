import numpy as np
import pygame
import moderngl
from OpenGL.GL import *
from src.config import load_config
from src.scene_setup import get_object_configs, load_and_voxelize_one
from src.physics.physics_world import build_scene_multi
from src.physics.simulation import step_simulation
from src.scene.camera import compute_view_projection, movement
from src.scene.input_handler import handle_events


def render_object(obj, transforms, sim, proj, view, eye):
    obj_scene = obj["scene"]
    obj_renderer = obj["obj_renderer"]
    mesh_splitter = obj["mesh_splitter"]
    voxel_body_start = obj_scene["voxel_body_start"]
    voxel_count = obj_scene["voxel_count"]
    if sim["render_mode"] == "mesh":
        mesh_splitter.deform_split_mesh(transforms, voxel_body_start, voxel_count)
        obj_renderer.update_voxel_transforms(mesh_splitter.last_voxel_slice)
        obj_renderer.draw_mesh_mode(proj, view, eye, tuple(obj["color"]), index_count=mesh_splitter.current_index_count)
    else:
        obj_renderer.update_voxel_instances(transforms, voxel_body_start)
        obj_renderer.update_voxel_colors(obj["joint_breaker"].get_voxel_colors())
        obj_renderer.draw_voxels(proj, view, eye)


def render_frame(renderer, scene, sim, obj_list, proj, view, eye):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    renderer.draw_ground(proj, view)
    transforms = scene["state_current"].body_q.numpy()
    for obj in obj_list:
        render_object(obj, transforms, sim, proj, view, eye)
    for body, radius, ball_config in zip(scene["ball_bodies"], scene["ball_radii"], scene["ball_cfgs"]):
        if not getattr(ball_config, "visible", True):
            continue
        renderer.update_ball_single(transforms, body, radius)
        renderer.draw_ball(proj, view, eye, tuple(ball_config.color))


def load_and_build_scene(scene_file):
    cfg = load_config(scene_file)
    obj_configs = get_object_configs(cfg)
    vox_ctx = moderngl.create_standalone_context()
    try:
        all_obj_data = [load_and_voxelize_one(cfg, oc, ctx=vox_ctx) for oc in obj_configs]
    finally:
        vox_ctx.release()
    scene = build_scene_multi(all_obj_data, cfg)
    return cfg, all_obj_data, scene


def update_recording(recorder, sim, was_simulating):
    if recorder and not recorder.active and sim["simulating"] and not was_simulating:
        recorder.start()
    elif recorder and recorder.active and not sim["simulating"] and was_simulating:
        recorder.stop()


def run_simulation_loop(scene, sim, camera, cfg, obj_list, joint_breakers, mesh_splitters, force_appliers, renderer, recorder, width, height):
    clock = pygame.time.Clock()
    was_simulating = sim["simulating"]
    running = True
    while running:
        running, width, height = handle_events(camera, sim, width, height)
        movement(camera)
        update_recording(recorder, sim, was_simulating)
        was_simulating = sim["simulating"]
        if sim["simulating"]:
            step_simulation(scene, sim, joint_breakers, mesh_splitters, force_appliers)
        proj, view, eye = compute_view_projection(camera, width, height)
        render_frame(renderer, scene, sim, obj_list, proj, view, eye)
        if recorder and recorder.active:
            recorder.write_frame(width, height)
        pygame.display.flip()
        clock.tick(sim["fps"])
    if recorder and recorder.active:
        recorder.stop()
