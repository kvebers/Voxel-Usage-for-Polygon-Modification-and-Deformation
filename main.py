import time
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from src import load_config, interpolate_camera, draw_floor, Shader, render_objects, init_objects


width, height = 800, 600


def gameLoop(config, cam_keys, ground_y, duration, scene_objects, shader):
    glViewport(0, 0, width, height)
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.1, 0.1, 0.1, 1.0)

    clock    = pygame.time.Clock()
    sim_time = 0.0
    running  = True

    while running:
        frame_start = time.perf_counter()

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                running = False

        dt = clock.tick(120) / 1000.0
        dt = min(dt, 0.1)
        sim_time += dt

        if duration > 0.0 and sim_time >= duration:
            running = False
            continue

        eye, target, fov = interpolate_camera(cam_keys, sim_time)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(fov, width / max(height, 1), 0.1, 200.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(*eye, *target, 0, 1, 0)

        draw_floor(ground_y)
        render_objects(scene_objects, shader)
        frame_time = time.perf_counter() - frame_start
        real_fps = 1.0 / max(frame_time, 1e-9)
        pygame.display.set_caption(f"AVBD — {real_fps:.0f} FPS | t={sim_time:.1f}s")
        pygame.display.flip()

    pygame.quit()


def main():
    config   = load_config("scene.json")
    cam_keys = sorted(config['cameras'], key=lambda c: c['time'])
    ground_y = config.get('simulation', {}).get('avbd', {}).get('groundY', 0.0)
    duration = config.get('simulation', {}).get('duration', 0.0)

    pygame.init()
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("AVBD")

    shader = Shader()
    shader.compile()

    scene_objects = init_objects(config)

    gameLoop(config, cam_keys, ground_y, duration, scene_objects, shader)


if __name__ == "__main__":
    main()