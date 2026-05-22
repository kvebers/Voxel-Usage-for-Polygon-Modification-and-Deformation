import pygame
from pygame.locals import *
from OpenGL.GL import *


def init_window(cfg):
    wcfg = cfg.window
    width, height = wcfg.width, wcfg.height
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | RESIZABLE)
    pygame.display.set_caption(wcfg.title)
    glEnable(GL_DEPTH_TEST)
    cc = wcfg.clear_color
    glClearColor(cc[0], cc[1], cc[2], cc[3])
    return width, height


def _handle_key_event(event, sim):
    if event.key == K_ESCAPE:
        return False
    elif event.key == K_c:
        sim["simulating"] = not sim["simulating"]
        print(f"Physics {'started' if sim['simulating'] else 'stopped'}")
    elif event.key == K_v:
        sim["render_mode"] = "voxel" if sim["render_mode"] == "mesh" else "mesh"
        print(f"Render mode: {sim['render_mode']}")
    elif event.key in (K_0, K_1, K_2, K_3, K_4):
        new_mode = event.key - K_0
        sim["force_mode"] = 0 if sim["force_mode"] == new_mode else new_mode
    elif event.key in (K_EQUALS, K_PLUS):
        sim["force_strength"] *= 2.0
        print(f"Force strength: {sim['force_strength']:.1f} N")
    elif event.key == K_MINUS:
        sim["force_strength"] = max(1.0, sim["force_strength"] / 2.0)
        print(f"Force strength: {sim['force_strength']:.1f} N")
    return True


def _handle_mouse_event(event, cam, dragging):
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
        dx, dy = event.rel
        sens = cam["mouse_sensitivity"]
        cam["yaw"] += dx * sens
        cam["pitch"] = max(-89, min(89, cam["pitch"] - dy * sens))
    return dragging


def handle_events(cam, sim, width, height):
    running = True
    mouse_dragging = getattr(handle_events, "_dragging", False)

    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if not _handle_key_event(event, sim):
                running = False
        elif event.type == VIDEORESIZE:
            width, height = event.w, event.h
            pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | RESIZABLE)
            glViewport(0, 0, width, height)
        elif event.type in (MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION):
            mouse_dragging = _handle_mouse_event(event, cam, mouse_dragging)

    handle_events._dragging = mouse_dragging
    return running, width, height
