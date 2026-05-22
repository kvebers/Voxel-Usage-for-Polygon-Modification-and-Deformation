import os
import subprocess
import argparse
import numpy as np
import warp as wp
from OpenGL.GL import *
from src.camera import init_camera
from src.simulation import init_sim_state
from src.component_factory import _create_per_object_components, _init_renderer
from src.game_loop import _load_and_build_scene, _run_game_loop
from src.profiler import Profiler, NullProfiler


class VideoRecorder:
    def __init__(self, path, width, height, fps):
        self._path = path
        self._width = width
        self._height = height
        self._fps = fps
        self._proc = None
        self.active = False

    def start(self):
        cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{self._width}x{self._height}", "-pix_fmt", "rgb24",
            "-r", str(self._fps), "-i", "pipe:0",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            self._path,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.active = True
        print(f"[recorder] Recording → {self._path}")

    def write_frame(self, width, height):
        buf = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
        self._proc.stdin.write(np.flipud(frame).tobytes())

    def stop(self):
        if self._proc:
            self._proc.stdin.close()
            self._proc.wait()
            self._proc = None
        self.active = False
        print(f"[recorder] Saved {self._path}")


wp.init()


def main(scene_file: str = "scene.json", profile: bool = False, record: str = None):
    setup = Profiler() if profile else NullProfiler()
    profiler = Profiler() if profile else NullProfiler()
    cfg, all_obj_data, scene = _load_and_build_scene(scene_file, setup)
    with setup.section("create_components"):
        obj_list, joint_breakers, mesh_splitters, force_appliers = (
            _create_per_object_components(all_obj_data, scene, cfg)
        )
    with setup.section("init_renderer"):
        renderer, width, height = _init_renderer(scene, obj_list, all_obj_data, cfg)
    cam = init_camera(cfg)
    sim = init_sim_state(cfg)
    recorder = VideoRecorder(record, width, height, sim["fps"]) if record else None
    _run_game_loop(
        scene, sim, cam, cfg, obj_list, joint_breakers, mesh_splitters,
        force_appliers, renderer, profiler, recorder, width, height,
    )
    setup.save_summary_csv("profile_setup.csv")
    profiler.save_csv("profile_frames.csv")
    import pygame
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", nargs="?", default="scene.jsonc", help="Scene config")
    parser.add_argument("--profile", action="store_true", help="Collect data")
    parser.add_argument("--record", nargs="?", const=True, metavar="OUTPUT.mp4", help="Record video")
    args = parser.parse_args()
    if args.record is True:
        record_path = os.path.splitext(os.path.basename(args.scene))[0] + ".mp4"
    else:
        record_path = args.record
    main(scene_file=args.scene, profile=args.profile, record=record_path)
