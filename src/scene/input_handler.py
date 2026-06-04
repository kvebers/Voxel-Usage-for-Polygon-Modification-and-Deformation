import pygame
from pygame.locals import *
from OpenGL.GL import *


def init_window(cfg):
    window_config = cfg.window
    width, height = window_config.width, window_config.height
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | RESIZABLE)
    pygame.display.set_caption(window_config.title)
    glEnable(GL_DEPTH_TEST)
    clear_color = window_config.clear_color
    glClearColor(clear_color[0], clear_color[1], clear_color[2], clear_color[3])
    return width, height


def handle_key_event(event, sim, camera=None, shooter=None, scene=None):
    if event.key == K_ESCAPE:
        return False
    elif event.key == K_c:
        sim["simulating"] = not sim["simulating"]
    elif event.key == K_v:
        cycle = {"mesh": "combined", "voxel": "mesh", "combined": "voxel"}
        sim["render_mode"] = cycle.get(sim["render_mode"], "mesh")
    elif event.key == K_SPACE:
        if shooter is not None and camera is not None and scene is not None:
            shooter.shoot(camera, scene)
    return True


def handle_mouse_event(event, cam, dragging):
    if event.type == MOUSEBUTTONDOWN:
        if event.button == 1:
            return True
        elif event.button == 4:
            cam["speed"] = min(10.0, cam["speed"] * 1.5)
        elif event.button == 5:
            cam["speed"] = max(0.01, cam["speed"] / 1.5)
    elif event.type == MOUSEBUTTONUP:
        if event.button == 1:
            return False
    elif event.type == MOUSEMOTION and dragging:
        mouse_dx, mouse_dy = event.rel
        sensitivity = cam["mouse_sensitivity"]
        cam["yaw"] += mouse_dx * sensitivity
        cam["pitch"] = max(-89, min(89, cam["pitch"] - mouse_dy * sensitivity))
    return dragging


def handle_events(cam, sim, width, height, shooter=None, scene=None):
    running = True
    mouse_dragging = getattr(handle_events, "_dragging", False)
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if not handle_key_event(event, sim, camera=cam, shooter=shooter, scene=scene):
                running = False
        elif event.type == VIDEORESIZE:
            width, height = event.w, event.h
            pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | RESIZABLE)
            glViewport(0, 0, width, height)
        elif event.type in (MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION):
            mouse_dragging = handle_mouse_event(event, cam, mouse_dragging)
    handle_events._dragging = mouse_dragging
    return running, width, height
