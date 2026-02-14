from OpenGL.GL import *
from OpenGL.GL import shaders

VERTEX_SHADER = """
#version 120
void main() {
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
"""

FRAGMENT_SHADER = """
#version 120
uniform vec3 color;
void main() {
    gl_FragColor = vec4(color, 1.0);
}
"""

class Shader:
    def __init__(self):
        self.program = None
    
    def compile(self):
        vertex = shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
        fragment = shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        self.program = shaders.compileProgram(vertex, fragment)
    
    def use(self):
        glUseProgram(self.program)
    
    def set_color(self, r, g, b):
        loc = glGetUniformLocation(self.program, "color")
        glUniform3f(loc, r, g, b)
    
    def stop(self):
        glUseProgram(0)