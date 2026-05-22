from src.physics_world import create_joint_breaker, create_mesh_splitter
from src.force_modes import ForceApplier
from src.renderer import Renderer
from src.input_handler import init_window


def _create_force_applier(obj_scene, cfg):
    el_cfg = getattr(cfg, "elasticity", None)
    half = obj_scene["half"]
    voxel_mass = cfg.voxels.density * 8.0 * half**3
    return ForceApplier(
        obj_scene["positions"],
        obj_scene["voxel_body_start"],
        obj_scene["voxel_count"],
        stiffness=el_cfg.stiffness if el_cfg else 0.0,
        damping=el_cfg.damping if el_cfg else 0.0,
        voxel_mass=voxel_mass,
        neighbor_pairs=obj_scene["neighbor_pairs"],
    )


def _create_per_object_components(all_obj_data, scene, cfg):
    obj_list = []
    joint_breakers = []
    mesh_splitters = []
    force_appliers = []
    for obj_data, obj_scene in zip(all_obj_data, scene["objects"]):
        jb = create_joint_breaker(obj_scene, scene["model"], cfg)
        ms = create_mesh_splitter(
            obj_data["indices"],
            obj_scene,
            obj_data["centered_verts"],
            obj_scene["neighbor_pairs"],
            cfg,
        )
        if len(jb.broken) > 0:
            ms.set_broken(jb.broken)
        obj_list.append({
            "scene": obj_scene,
            "mesh_splitter": ms,
            "joint_breaker": jb,
            "color": obj_data["color"],
        })
        joint_breakers.append(jb)
        mesh_splitters.append(ms)
        force_appliers.append(_create_force_applier(obj_scene, cfg))
    return obj_list, joint_breakers, mesh_splitters, force_appliers


def _init_renderer(scene, obj_list, all_obj_data, cfg):
    width, height = init_window(cfg)
    renderer = Renderer(scene["ball_radius"], max_balls=max(1, len(scene["ball_bodies"])))
    for i, obj in enumerate(obj_list):
        obj_scene = obj["scene"]
        obj_renderer = renderer.create_object_renderer(
            all_obj_data[i]["indices"],
            obj_scene["voxel_count"],
            obj_scene["half"],
            block_halves=obj_scene["block_halves_world"],
        )
        obj_renderer.setup_gpu_deform(obj["mesh_splitter"])
        obj["obj_renderer"] = obj_renderer
    return renderer, width, height
