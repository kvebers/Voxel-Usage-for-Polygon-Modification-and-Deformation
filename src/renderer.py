from OpenGL.GL import *


def draw_floor(y=0.0, size=40.0):
    glColor3f(0.18, 0.18, 0.20)
    glBegin(GL_QUADS)
    glVertex3f(-size, y, -size)
    glVertex3f( size, y, -size)
    glVertex3f( size, y,  size)
    glVertex3f(-size, y,  size)
    glEnd()