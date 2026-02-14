import json
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


def load_config(path="scene.json"):
    with open(path, 'r') as f:
        return json.load(f)

def interpolate_camera(cam_keys, t):
    if not cam_keys:
        return [20, 6, 0], [0, 0, 0], 45.0
    if t <= cam_keys[0]['time']:
        c = cam_keys[0]
        return c['position'], c['target'], c['fov']
    if t >= cam_keys[-1]['time']:
        c = cam_keys[-1]
        return c['position'], c['target'], c['fov']

    for i in range(len(cam_keys) - 1):
        c0, c1 = cam_keys[i], cam_keys[i + 1]
        if c0['time'] <= t <= c1['time']:
            f = (t - c0['time']) / max(c1['time'] - c0['time'], 1e-6)
            pos = [c0['position'][j] + f * (c1['position'][j] - c0['position'][j]) for j in range(3)]
            tgt = [c0['target'][j]   + f * (c1['target'][j]   - c0['target'][j])   for j in range(3)]
            fov = c0['fov'] + f * (c1['fov'] - c0['fov'])
            return pos, tgt, fov

    c = cam_keys[-1]
    return c['position'], c['target'], c['fov']

def draw_floor(y=0.0, size=40.0):
    glColor3f(0.18, 0.18, 0.20)
    glBegin(GL_QUADS)
    glVertex3f(-size, y, -size)
    glVertex3f( size, y, -size)
    glVertex3f( size, y,  size)
    glVertex3f(-size, y,  size)
    glEnd()

def main():
    config   = load_config("scene.json")
    cam_keys = sorted(config['cameras'], key=lambda c: c['time'])
    ground_y = config.get('simulation', {}).get('avbd', {}).get('groundY', 0.0)
    duration = config.get('simulation', {}).get('duration', 0.0)

    width, height = 800, 600

    pygame.init()
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("AVBD")

    glViewport(0, 0, width, height)
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.1, 0.1, 0.1, 1.0)

    clock    = pygame.time.Clock()
    sim_time = 0.0
    running  = True

    while running:
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
        gluLookAt(eye[0], eye[1], eye[2],
                  target[0], target[1], target[2],
                  0, 1, 0)
        draw_floor(ground_y)

        fps = clock.get_fps()
        pygame.display.set_caption(f"AVBD — {fps:.0f} FPS | t={sim_time:.1f}s")
        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main()