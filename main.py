import warp as wp
import pygame
from src.scene.camera import init_camera
from src.physics.simulation import init_sim_state
from src.scene.component_factory import create_components, init_renderer
from src.game_loop import load_and_build_scene, run_simulation_loop
from src.utils.video_recorder import VideoRecorder
from src.utils.parse_args import parse_args

wp.init()


def main(scene_file="scene.json", record=None):
    cfg, all_obj_data, scene = load_and_build_scene(scene_file)
    obj_list, joint_breakers, mesh_splitters, force_appliers = create_components(all_obj_data, scene, cfg)
    renderer, width, height = init_renderer(scene, obj_list, all_obj_data, cfg)
    camera = init_camera(cfg)
    sim_state = init_sim_state(cfg)
    recorder = VideoRecorder(record, width, height, sim_state["fps"]) if record else None
    run_simulation_loop(scene, sim_state, camera, cfg, obj_list, joint_breakers, mesh_splitters, force_appliers, renderer, recorder, width, height)
    pygame.quit()


if __name__ == "__main__":
    scene_file, record = parse_args()
    main(scene_file=scene_file, record=record)
