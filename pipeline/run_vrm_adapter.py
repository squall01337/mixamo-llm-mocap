"""Host-side command for the native VRM adapter.

Runs Blender in background mode with ``apply_vrm_fk.py`` so users do not need
to assemble Blender's argument separator by hand.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--vrm", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--profile-out")
    parser.add_argument("--blend-out")
    parser.add_argument("--action-name")
    parser.add_argument("--qa-frames")
    parser.add_argument("--no-stills", action="store_true")
    parser.add_argument("--blender", default=os.environ.get("BLENDER_EXE", str(DEFAULT_BLENDER)))
    args = parser.parse_args()

    blender = Path(args.blender)
    if not blender.exists():
        raise SystemExit(
            f"Blender executable not found: {blender}\n"
            "Pass --blender /path/to/blender or set BLENDER_EXE."
        )
    command = [
        str(blender), "--background", "--python", str(REPO / "pipeline" / "apply_vrm_fk.py"), "--",
        "--spec", args.spec, "--vrm", args.vrm, "--out-dir", args.out_dir,
    ]
    for flag, value in (
        ("--profile-out", args.profile_out), ("--blend-out", args.blend_out),
        ("--action-name", args.action_name), ("--qa-frames", args.qa_frames),
    ):
        if value:
            command.extend((flag, value))
    if args.no_stills:
        command.append("--no-stills")
    completed = subprocess.run(command, cwd=REPO, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
