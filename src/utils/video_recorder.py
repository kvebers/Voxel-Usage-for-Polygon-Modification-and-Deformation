import subprocess
import numpy as np
from OpenGL.GL import *


class VideoRecorder:
    def __init__(self, path, width, height, fps):
        self._path = path
        self._width = width
        self._height = height
        self._fps = fps
        self.proc = None
        self.active = False

    def start(self):
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self._width}x{self._height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(self._fps),
            "-i",
            "pipe:0",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            self._path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.active = True

    def write_frame(self, width, height):
        pixel_buffer = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        frame = np.frombuffer(pixel_buffer, dtype=np.uint8).reshape(height, width, 3)
        self.proc.stdin.write(np.flipud(frame).tobytes())

    def stop(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
            self.proc = None
        self.active = False
