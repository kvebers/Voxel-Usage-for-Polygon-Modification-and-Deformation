VERT_MESH = """
#version 330 core
// vertex shader for deformable mesh bound to voxels
layout(location = 0) in vec3 aLocalOffset; // vertex offset from voxel center
layout(location = 1) in vec3 aRestNormal; // normal direction at rest pose
layout(location = 2) in float aVoxelBinding; // which voxel this vertex follows

uniform samplerBuffer uVoxelPos; // GPU buffer: position of each voxel
uniform samplerBuffer uVoxelQuat; // GPU buffer: rotation of each voxel

uniform mat4 uProj; // perspective matrix
uniform mat4 uView; // camera matrix

out vec3 vWorldPosition; // pass world pos to fragment shader
out vec3 vSurfaceNormal; // pass rotated normal to fragment shader

// rotate a vector by a quaternion
vec3 quatRotate(vec4 quaternion, vec3 inputVector) {
    vec3 crossTerm = 2.0 * cross(quaternion.xyz, inputVector);
    return inputVector + quaternion.w * crossTerm + cross(quaternion.xyz, crossTerm);
}

void main() {
    int  voxelIndex = int(aVoxelBinding);  // which voxel owns this vertex
    vec3 voxelPosition = texelFetch(uVoxelPos,  voxelIndex).xyz;  // read voxel position from GPU buffer
    vec4 voxelRotation = texelFetch(uVoxelQuat, voxelIndex);  // read voxel rotation from GPU buffer

    vec4 worldPosition = vec4(voxelPosition + quatRotate(voxelRotation, aLocalOffset), 1.0); // rotate offset then add to voxel position
    vWorldPosition = worldPosition.xyz;
    vSurfaceNormal = normalize(quatRotate(voxelRotation, aRestNormal)); // rotate normal with voxel
    gl_Position = uProj * uView * worldPosition; // project to screen
}
"""

FRAG_BLINN = """
#version 330 core
// solid mesh
in vec3 vWorldPosition;
in vec3 vSurfaceNormal;

uniform vec3 uColor; // object color
uniform vec3 uLightDir; // direction light comes from
uniform vec3 uCamPos; // camera position for specular
uniform float uAmbient; // minimum brightness

out vec4 FragColor;

void main() {
    vec3  surfaceNormal   = normalize(vSurfaceNormal);
    vec3  lightDirection  = normalize(uLightDir);
    float diffuseFactor   = max(dot(surfaceNormal, lightDirection), 0.0); // how much light hits surface

    vec3  viewDirection   = normalize(uCamPos - vWorldPosition);
    vec3  halfVector      = normalize(lightDirection + viewDirection);
    float specularFactor  = pow(max(dot(surfaceNormal, halfVector), 0.0), 64.0); // shiny highlight

    vec3 finalColor = uColor * (uAmbient + diffuseFactor * 0.7) + vec3(1.0) * specularFactor * 0.3;
    FragColor = vec4(finalColor, 1.0);
}
"""


VERT_INSTANCED_COLOR = """
#version 330 core
// vertex shader for instanced rendering where each instance has its own color
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aSurfaceNormal;
// model matrix passed as 4 columns because GPU attributes max 4 floats each
layout(location = 2) in vec4 aModelColumn0;
layout(location = 3) in vec4 aModelColumn1;
layout(location = 4) in vec4 aModelColumn2;
layout(location = 5) in vec4 aModelColumn3;

layout(location = 6) in vec3 aInstanceColor; // per-instance color (health color)

uniform mat4 uProj;
uniform mat4 uView;

out vec3 vWorldPosition;
out vec3 vSurfaceNormal;
out vec3 vInstanceColor;

void main() {
    mat4 modelMatrix  = mat4(aModelColumn0, aModelColumn1, aModelColumn2, aModelColumn3); // rebuild model matrix from columns
    mat3 normalMatrix = mat3(modelMatrix); // 3x3 for rotating normals (no translation)
    vec4 worldPosition = modelMatrix * vec4(aPosition, 1.0);  // move vertex to world space
    vWorldPosition = worldPosition.xyz;
    vSurfaceNormal = normalize(normalMatrix * aSurfaceNormal); // rotate normal with model
    vInstanceColor = aInstanceColor;
    gl_Position = uProj * uView * worldPosition; // project to screen
}
"""

FRAG_BLINN_INSTANCED = """
#version 330 core
// blinn-phong lighting using per-instance color
in vec3 vWorldPosition;
in vec3 vSurfaceNormal;
in vec3 vInstanceColor; // color passed from vertex shader

uniform vec3 uLightDir;
uniform vec3 uCamPos;
uniform float uAmbient;

out vec4 FragColor;

void main() {
    vec3  surfaceNormal  = normalize(vSurfaceNormal);
    vec3  lightDirection = normalize(uLightDir);
    float diffuseFactor  = max(dot(surfaceNormal, lightDirection), 0.0); // how much light hits surface
    vec3  viewDirection  = normalize(uCamPos - vWorldPosition);
    vec3  halfVector     = normalize(lightDirection + viewDirection);
    float specularFactor = pow(max(dot(surfaceNormal, halfVector), 0.0), 64.0); // shiny highlight
    vec3 finalColor = vInstanceColor * (uAmbient + diffuseFactor * 0.7) + vec3(1.0) * specularFactor * 0.3;
    FragColor = vec4(finalColor, 1.0);
}
"""


VERT_INSTANCED = """
#version 330 core
// vertex shader for instanced rendering with single uniform color
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aSurfaceNormal;
// model matrix split into 4 columns
layout(location = 2) in vec4 aModelColumn0;
layout(location = 3) in vec4 aModelColumn1;
layout(location = 4) in vec4 aModelColumn2;
layout(location = 5) in vec4 aModelColumn3;

uniform mat4 uProj;
uniform mat4 uView;

out vec3 vWorldPosition;
out vec3 vSurfaceNormal;

void main() {
    mat4 modelMatrix  = mat4(aModelColumn0, aModelColumn1, aModelColumn2, aModelColumn3); // rebuild model matrix
    mat3 normalMatrix = mat3(modelMatrix); // 3x3 for normals only
    vec4 worldPosition = modelMatrix * vec4(aPosition, 1.0);
    vWorldPosition = worldPosition.xyz;
    vSurfaceNormal = normalize(normalMatrix * aSurfaceNormal);
    gl_Position = uProj * uView * worldPosition;
}
"""

VERT_GROUND = """
#version 330 core
// simple pass-through vertex shader for the ground plane
layout(location = 0) in vec3 aPosition;

uniform mat4 uProj;
uniform mat4 uView;

out vec3 vWorldPosition;

void main() {
    vWorldPosition = aPosition; // pass raw position for checker pattern
    gl_Position = uProj * uView * vec4(aPosition, 1.0);
}
"""

FRAG_GROUND = """
#version 330 core
// draws checker tile pattern on the ground
in vec3 vWorldPosition;
out vec4 FragColor;

void main() {
    float flooredX = floor(vWorldPosition.x);
    float flooredY = floor(vWorldPosition.y);
    float checkerPattern = mod(flooredX + flooredY, 2.0); // alternates 0 and 1 per tile
    vec3 darkTileColor  = vec3(0.18, 0.20, 0.22);
    vec3 lightTileColor = vec3(0.25, 0.27, 0.30);
    vec3 finalColor = mix(darkTileColor, lightTileColor, checkerPattern); // pick dark or light tile
    FragColor = vec4(finalColor, 1.0);
}
"""
