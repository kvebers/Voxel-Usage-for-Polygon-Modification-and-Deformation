from src.physics.physics_world import create_joint_breaker, create_mesh_splitter
from src.scene.force_modes import ForceApplier
from src.rendering.renderer import Renderer
from src.scene.input_handler import init_window


def create_force_applier(obj_scene, cfg):
    elasticity_cfg = getattr(cfg, "elasticity", None)
    half = obj_scene["half"]
    voxel_mass = cfg.voxels.density * 8.0 * half**3
    return ForceApplier(
        obj_scene["positions"],
        obj_scene["voxel_body_start"],
        obj_scene["voxel_count"],
        stiffness=elasticity_cfg.stiffness if elasticity_cfg else 0.0,
        damping=elasticity_cfg.damping if elasticity_cfg else 0.0,
        voxel_mass=voxel_mass,
        neighbor_pairs=obj_scene["neighbor_pairs"],
    )


def create_components(all_obj_data, scene, cfg):
    obj_list = []
    joint_breakers = []
    mesh_splitters = []
    force_appliers = []
    for obj_data, obj_scene in zip(all_obj_data, scene["objects"]):
        joint_breaker = create_joint_breaker(obj_scene, scene["model"], cfg)
        mesh_splitter = create_mesh_splitter(obj_data["indices"], obj_scene, obj_data["centered_verts"], obj_scene["neighbor_pairs"])
        if len(joint_breaker.broken) > 0:
            mesh_splitter.set_broken(joint_breaker.broken)
        obj_list.append({"scene": obj_scene, "mesh_splitter": mesh_splitter, "joint_breaker": joint_breaker, "color": obj_data["color"]})
        joint_breakers.append(joint_breaker)
        mesh_splitters.append(mesh_splitter)
        force_appliers.append(create_force_applier(obj_scene, cfg))
    return obj_list, joint_breakers, mesh_splitters, force_appliers


def init_renderer(scene, obj_list, all_obj_data, cfg):
    width, height = init_window(cfg)
    renderer = Renderer()
    renderer.setup_walls(cfg)
    for i, obj in enumerate(obj_list):
        obj_scene = obj["scene"]
        obj_renderer = renderer.create_object_renderer(
            all_obj_data[i]["indices"], obj_scene["voxel_count"], obj_scene["half"], block_halves=obj_scene["block_halves_world"]
        )
        obj_renderer.setup_gpu_deform(obj["mesh_splitter"])
        obj["obj_renderer"] = obj_renderer
    return renderer, width, height
