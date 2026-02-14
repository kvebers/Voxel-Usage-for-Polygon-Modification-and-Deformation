import ctypes
import numpy as np
from OpenGL.GL import *


def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == 'v':
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif p[0] == 'f':
                face = [int(s.split('/')[0]) - 1 for s in p[1:]]
                for i in range(1, len(face) - 1):
                    faces.append((face[0], face[i], face[i + 1]))

    data = []
    for f in faces:
        for vi in f:
            data.extend(verts[vi])
    return np.array(data, dtype=np.float32), len(faces) * 3


class ObjModel:
    def __init__(self, path):
        data, self.count = load_obj(path)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))
        glDrawArrays(GL_TRIANGLES, 0, self.count)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)


def init_objects(config):
    objects = []
    for o in config.get('objects', []):
        objects.append({'mesh': ObjModel(o['file']), 'config': o})
    return objects


def render_objects(objects, shader):
    shader.use()
    for obj in objects:
        cfg = obj['config']
        pos = cfg.get('position', [0, 0, 0])
        rot = cfg.get('rotation', [0, 0, 0])
        scale = cfg.get('scale', [1, 1, 1])
        color = cfg.get('color', [0.8, 0.8, 0.8])

        shader.set_color(*color)

        glPushMatrix()
        glTranslatef(*pos)
        glRotatef(rot[0], 1, 0, 0)
        glRotatef(rot[1], 0, 1, 0)
        glRotatef(rot[2], 0, 0, 1)
        glScalef(*scale)
        obj['mesh'].draw()
        glPopMatrix()

    shader.stop()