"""Render an animated VRM adapter output to H.264 MP4.

Requires Blender 5.1+ and an ffmpeg executable on PATH (or --ffmpeg).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--blender", default=os.environ.get("BLENDER_EXE", str(DEFAULT_BLENDER)))
    parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG_EXE", "ffmpeg"))
    args = parser.parse_args()

    blend, profile, output = Path(args.blend).resolve(), Path(args.profile).resolve(), Path(args.out).resolve()
    blender = Path(args.blender)
    if not blender.exists():
        raise SystemExit(f"Blender executable not found: {blender}")
    ffmpeg = shutil.which(args.ffmpeg) or (str(Path(args.ffmpeg)) if Path(args.ffmpeg).exists() else None)
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found; install it, pass --ffmpeg, or set FFMPEG_EXE")
    frames_dir = output.parent / f"{output.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    render_command = [
        str(blender), "--background", str(blend), "--python",
        str(REPO/"pipeline"/"render_vrm_frames.py"), "--",
        "--profile", str(profile), "--frames-dir", str(frames_dir),
        "--width", str(args.width), "--height", str(args.height),
    ]
    subprocess.run(render_command, cwd=REPO, check=True)
    first_frame = frames_dir / "frame_0001.png"
    if not first_frame.exists():
        raise SystemExit(
            f"Blender did not produce {first_frame}; inspect the Blender traceback above "
            "(Blender may return exit code 0 after a Python exception)."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    encode_command = [
        ffmpeg, "-y", "-framerate", str(args.fps),
        "-i", str(frames_dir/"frame_%04d.png"), "-c:v", "libx264",
        "-preset", "medium", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]
    subprocess.run(encode_command, cwd=REPO, check=True)
    print(f"VRM_VIDEO={output}")


if __name__ == "__main__":
    main()
