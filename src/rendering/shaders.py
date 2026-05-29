VERT_MESH = """
#version 330 core
layout(location = 0) in vec3 aRestPos;
layout(location = 1) in vec3 aRestNormal;
layout(location = 2) in vec4 aBlendIndices;
layout(location = 3) in vec4 aBlendWeights;

uniform samplerBuffer uVoxelSkinT;
uniform samplerBuffer uVoxelQuat;

uniform mat4 uProj;
uniform mat4 uView;

out vec3 vWorldPosition;
out vec3 vSurfaceNormal;

vec3 quatRotate(vec4 q, vec3 v) {
    vec3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

void applyBone(int vi, float w, inout vec3 pos, inout vec3 norm) {
    vec4 rot   = texelFetch(uVoxelQuat,  vi);
    vec3 skinT = texelFetch(uVoxelSkinT, vi).xyz;
    pos  += w * (skinT + quatRotate(rot, aRestPos));
    norm += w * quatRotate(rot, aRestNormal);
}

void main() {
    vec3 worldPos  = vec3(0.0);
    vec3 worldNorm = vec3(0.0);
    applyBone(int(aBlendIndices.x), aBlendWeights.x, worldPos, worldNorm);
    applyBone(int(aBlendIndices.y), aBlendWeights.y, worldPos, worldNorm);
    applyBone(int(aBlendIndices.z), aBlendWeights.z, worldPos, worldNorm);
    applyBone(int(aBlendIndices.w), aBlendWeights.w, worldPos, worldNorm);
    vWorldPosition = worldPos;
    vSurfaceNormal = normalize(worldNorm);
    gl_Position    = uProj * uView * vec4(worldPos, 1.0);
}
"""

FRAG_BLINN = """
#version 330 core
in vec3 vWorldPosition;
in vec3 vSurfaceNormal;

uniform vec3 uColor;
uniform vec3 uLightDir;
uniform vec3 uCamPos;
uniform float uAmbient;

out vec4 FragColor;

void main() {
    vec3  surfaceNormal = normalize(vSurfaceNormal);
    vec3  lightDirection = normalize(uLightDir);
    float diffuseFactor = max(dot(surfaceNormal, lightDirection), 0.0);

    vec3  viewDirection = normalize(uCamPos - vWorldPosition);
    vec3  halfVector = normalize(lightDirection + viewDirection);
    float specularFactor = pow(max(dot(surfaceNormal, halfVector), 0.0), 64.0);

    vec3 finalColor = uColor * (uAmbient + diffuseFactor * 0.7) + vec3(1.0) * specularFactor * 0.3;
    FragColor = vec4(finalColor, 1.0);
}
"""


VERT_INSTANCED_COLOR = """
#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aSurfaceNormal;
layout(location = 2) in vec4 aModelColumn0;
layout(location = 3) in vec4 aModelColumn1;
layout(location = 4) in vec4 aModelColumn2;
layout(location = 5) in vec4 aModelColumn3;

layout(location = 6) in vec3 aInstanceColor;

uniform mat4 uProj;
uniform mat4 uView;

out vec3 vWorldPosition;
out vec3 vSurfaceNormal;
out vec3 vInstanceColor;

void main() {
    mat4 modelMatrix = mat4(aModelColumn0, aModelColumn1, aModelColumn2, aModelColumn3);
    mat3 normalMatrix = mat3(modelMatrix);
    vec4 worldPosition = modelMatrix * vec4(aPosition, 1.0);
    vWorldPosition = worldPosition.xyz;
    vSurfaceNormal = normalize(normalMatrix * aSurfaceNormal);
    vInstanceColor = aInstanceColor;
    gl_Position = uProj * uView * worldPosition;
}
"""

FRAG_BLINN_INSTANCED = """
#version 330 core
in vec3 vWorldPosition;
in vec3 vSurfaceNormal;
in vec3 vInstanceColor;

uniform vec3 uLightDir;
uniform vec3 uCamPos;
uniform float uAmbient;
uniform float uAlpha;

out vec4 FragColor;

void main() {
    vec3  surfaceNormal = normalize(vSurfaceNormal);
    vec3  lightDirection = normalize(uLightDir);
    float diffuseFactor = max(dot(surfaceNormal, lightDirection), 0.0);
    vec3  viewDirection = normalize(uCamPos - vWorldPosition);
    vec3  halfVector = normalize(lightDirection + viewDirection);
    float specularFactor = pow(max(dot(surfaceNormal, halfVector), 0.0), 64.0);
    vec3 finalColor = vInstanceColor * (uAmbient + diffuseFactor * 0.7) + vec3(1.0) * specularFactor * 0.3;
    FragColor = vec4(finalColor, uAlpha);
}
"""


VERT_INSTANCED = """
#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aSurfaceNormal;
layout(location = 2) in vec4 aModelColumn0;
layout(location = 3) in vec4 aModelColumn1;
layout(location = 4) in vec4 aModelColumn2;
layout(location = 5) in vec4 aModelColumn3;

uniform mat4 uProj;
uniform mat4 uView;

out vec3 vWorldPosition;
out vec3 vSurfaceNormal;

void main() {
    mat4 modelMatrix = mat4(aModelColumn0, aModelColumn1, aModelColumn2, aModelColumn3);
    mat3 normalMatrix = mat3(modelMatrix);
    vec4 worldPosition = modelMatrix * vec4(aPosition, 1.0);
    vWorldPosition = worldPosition.xyz;
    vSurfaceNormal = normalize(normalMatrix * aSurfaceNormal);
    gl_Position = uProj * uView * worldPosition;
}
"""

VERT_WALL = """
#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;

uniform mat4 uProj;
uniform mat4 uView;

out vec3 vWorldPosition;
out vec3 vSurfaceNormal;

void main() {
    vWorldPosition = aPosition;
    vSurfaceNormal = aNormal;
    gl_Position = uProj * uView * vec4(aPosition, 1.0);
}
"""

VERT_GROUND = """
#version 330 core
layout(location = 0) in vec3 aPosition;

uniform mat4 uProj;
uniform mat4 uView;

out vec3 vWorldPosition;

void main() {
    vWorldPosition = aPosition;
    gl_Position = uProj * uView * vec4(aPosition, 1.0);
}
"""

FRAG_GROUND = """
#version 330 core
in vec3 vWorldPosition;
out vec4 FragColor;

void main() {
    float flooredX = floor(vWorldPosition.x);
    float flooredY = floor(vWorldPosition.y);
    float checkerPattern = mod(flooredX + flooredY, 2.0);
    vec3 darkTileColor = vec3(0.18, 0.20, 0.22);
    vec3 lightTileColor = vec3(0.25, 0.27, 0.30);
    vec3 finalColor = mix(darkTileColor, lightTileColor, checkerPattern);
    FragColor = vec4(finalColor, 1.0);
}
"""
