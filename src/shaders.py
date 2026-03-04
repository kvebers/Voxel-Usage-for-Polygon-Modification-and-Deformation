VERT_MESH = """
#version 330 core
layout(location = 0) in vec3 aLocalOffset; // BASE_ROT * rest_offset from voxel centre (static)
layout(location = 1) in vec3 aRestNormal;  // rest-pose world-space normal (static)
layout(location = 2) in float aBinding;    // primary voxel index (static)

uniform samplerBuffer uVoxelPos;   // current voxel world positions  xyz  (unit 0, RGB32F)
uniform samplerBuffer uVoxelQuat;  // current voxel quaternions      xyzw (unit 1, RGBA32F)

uniform mat4 uProj;
uniform mat4 uView;

out vec3 vWorldPos;
out vec3 vNormal;

// Rotate v by unit quaternion q = (x,y,z,w).
vec3 quatRotate(vec4 q, vec3 v) {
    vec3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

void main() {
    int  bi   = int(aBinding);
    vec3 pos  = texelFetch(uVoxelPos,  bi).xyz;
    vec4 quat = texelFetch(uVoxelQuat, bi);

    vec4 world = vec4(pos + quatRotate(quat, aLocalOffset), 1.0);
    vWorldPos  = world.xyz;
    vNormal    = normalize(quatRotate(quat, aRestNormal));
    gl_Position = uProj * uView * world;
}
"""

FRAG_BLINN = """
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;

uniform vec3 uColor;
uniform vec3 uLightDir;
uniform vec3 uCamPos;
uniform float uAmbient;

out vec4 FragColor;

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightDir);
    float diff = max(dot(N, L), 0.0);

    vec3 V = normalize(uCamPos - vWorldPos);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 64.0);

    vec3 col = uColor * (uAmbient + diff * 0.7) + vec3(1.0) * spec * 0.3;
    FragColor = vec4(col, 1.0);
}
"""


VERT_INSTANCED_COLOR = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec4 aModel0;
layout(location = 3) in vec4 aModel1;
layout(location = 4) in vec4 aModel2;
layout(location = 5) in vec4 aModel3;

layout(location = 6) in vec3 aInstColor;

uniform mat4 uProj;
uniform mat4 uView;

out vec3 vWorldPos;
out vec3 vNormal;
out vec3 vColor;

void main() {
    mat4 model = mat4(aModel0, aModel1, aModel2, aModel3);
    mat3 normalMat = mat3(model);
    vec4 world = model * vec4(aPos, 1.0);
    vWorldPos = world.xyz;
    vNormal = normalize(normalMat * aNormal);
    vColor = aInstColor;
    gl_Position = uProj * uView * world;
}
"""

FRAG_BLINN_INSTANCED = """
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec3 vColor;

uniform vec3 uLightDir;
uniform vec3 uCamPos;
uniform float uAmbient;

out vec4 FragColor;

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightDir);
    float diff = max(dot(N, L), 0.0);

    vec3 V = normalize(uCamPos - vWorldPos);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 64.0);

    vec3 col = vColor * (uAmbient + diff * 0.7) + vec3(1.0) * spec * 0.3;
    FragColor = vec4(col, 1.0);
}
"""


VERT_INSTANCED = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec4 aModel0;
layout(location = 3) in vec4 aModel1;
layout(location = 4) in vec4 aModel2;
layout(location = 5) in vec4 aModel3;

uniform mat4 uProj;
uniform mat4 uView;

out vec3 vWorldPos;
out vec3 vNormal;

void main() {
    mat4 model = mat4(aModel0, aModel1, aModel2, aModel3);
    mat3 normalMat = mat3(model);
    vec4 world = model * vec4(aPos, 1.0);
    vWorldPos = world.xyz;
    vNormal = normalize(normalMat * aNormal);
    gl_Position = uProj * uView * world;
}
"""

VERT_GROUND = """
#version 330 core
layout(location = 0) in vec3 aPos;

uniform mat4 uProj;
uniform mat4 uView;

out vec3 vWorldPos;

void main() {
    vWorldPos = aPos;
    gl_Position = uProj * uView * vec4(aPos, 1.0);
}
"""

FRAG_GROUND = """
#version 330 core
in vec3 vWorldPos;
out vec4 FragColor;

void main() {
    float fx = floor(vWorldPos.x);
    float fy = floor(vWorldPos.y);
    float checker = mod(fx + fy, 2.0);
    vec3 dark  = vec3(0.18, 0.20, 0.22);
    vec3 light = vec3(0.25, 0.27, 0.30);
    vec3 col = mix(dark, light, checker);
    FragColor = vec4(col, 1.0);
}
"""
