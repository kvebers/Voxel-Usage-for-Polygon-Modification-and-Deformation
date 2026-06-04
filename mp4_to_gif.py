import sys
import os
from pathlib import Path


def mp4_to_gif(input_path: str, output_path: str = None, fps: int = None, scale: float = 1.0):
    try:
        from moviepy import VideoFileClip
    except ImportError:
        print("moviepy not installed. Run: pip install moviepy")
        sys.exit(1)

    input_path = Path(input_path)
    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    if output_path is None:
        output_path = input_path.with_suffix(".gif")
    else:
        output_path = Path(output_path)

    with VideoFileClip(str(input_path)) as clip:
        # GIF format stores delays in centiseconds; 50fps is the highest browsers honour
        effective_fps = min(fps if fps is not None else clip.fps, 50)
        print(f"Source fps: {clip.fps:.2f} -> GIF fps: {effective_fps} (GIF max is 50)")
        print(f"Converting {input_path} -> {output_path} (scale={scale})")

        if scale != 1.0:
            new_width = int(clip.w * scale)
            clip = clip.resized(width=new_width)
        clip.write_gif(str(output_path), fps=effective_fps)

    print(f"Done: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mp4_to_gif.py <input.mp4> [output.gif] [fps] [scale]")
        print("  fps:   frames per second for the GIF (default: 10)")
        print("  scale: resize factor, e.g. 0.5 for half size (default: 1.0)")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    fps_arg = int(sys.argv[3]) if len(sys.argv) > 3 else None
    scale_arg = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    mp4_to_gif(input_file, output_file, fps=fps_arg, scale=scale_arg)
