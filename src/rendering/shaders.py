VERT_MESH = """
#version 330 core
layout(location = 0) in vec3 aLocalOffset;
layout(location = 1) in vec3 aRestNormal;
layout(location = 2) in float aVoxelBinding;

uniform samplerBuffer uVoxelPos;
uniform samplerBuffer uVoxelQuat;

uniform mat4 uProj;
uniform mat4 uView;

out vec3 vWorldPosition;
out vec3 vSurfaceNormal;

vec3 quatRotate(vec4 quaternion, vec3 inputVector) {
    vec3 crossTerm = 2.0 * cross(quaternion.xyz, inputVector);
    return inputVector + quaternion.w * crossTerm + cross(quaternion.xyz, crossTerm);
}

void main() {
    int  voxelIndex = int(aVoxelBinding);
    vec3 voxelPosition = texelFetch(uVoxelPos,  voxelIndex).xyz;
    vec4 voxelRotation = texelFetch(uVoxelQuat, voxelIndex);

    vec4 worldPosition = vec4(voxelPosition + quatRotate(voxelRotation, aLocalOffset), 1.0);
    vWorldPosition = worldPosition.xyz;
    vSurfaceNormal = normalize(quatRotate(voxelRotation, aRestNormal));
    gl_Position = uProj * uView * worldPosition;
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

out vec4 FragColor;

void main() {
    vec3  surfaceNormal = normalize(vSurfaceNormal);
    vec3  lightDirection = normalize(uLightDir);
    float diffuseFactor = max(dot(surfaceNormal, lightDirection), 0.0);
    vec3  viewDirection = normalize(uCamPos - vWorldPosition);
    vec3  halfVector = normalize(lightDirection + viewDirection);
    float specularFactor = pow(max(dot(surfaceNormal, halfVector), 0.0), 64.0);
    vec3 finalColor = vInstanceColor * (uAmbient + diffuseFactor * 0.7) + vec3(1.0) * specularFactor * 0.3;
    FragColor = vec4(finalColor, 1.0);
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
