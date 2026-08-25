"""Blender-side full-body frame renderer used by render_vrm_video.py."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    if args.frame_start is not None:
        scene.frame_start = args.frame_start
    if args.frame_end is not None:
        scene.frame_end = args.frame_end
    arm = bpy.data.objects.get(profile["armature"])
    if arm is None or arm.type != "ARMATURE":
        raise RuntimeError(f"profile armature {profile['armature']!r} is not in the scene")
    mapping = profile["humanoid_bones"]
    rest_heads = [Vector(value) for value in profile["rest_heads_world"].values()]
    rest_min_x, rest_max_x = min(p.x for p in rest_heads), max(p.x for p in rest_heads)
    rest_min_z, rest_max_z = min(p.z for p in rest_heads), max(p.z for p in rest_heads)
    bone_height = max(0.5, rest_max_z-rest_min_z)
    bone_width = max(0.5, rest_max_x-rest_min_x)
    # The camera is fixed, so frame it from the complete animated trajectory.
    # Rest-pose bounds alone clip root travel, jumps and extended limbs.
    heads = []
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        heads.extend(
            arm.matrix_world @ arm.pose.bones[bone].matrix.to_translation()
            for bone in mapping.values()
        )
    min_x, max_x = min(p.x for p in heads), max(p.x for p in heads)
    min_y = min(p.y for p in heads)
    min_z, max_z = min(p.z for p in heads), max(p.z for p in heads)
    # Bone heads stop at the skull base, wrist and ankle. Add asymmetric
    # anatomical margins for hair/head, soles and fingertips. Avoid raw mesh
    # bounds here: tails, skirts and spring-bone accessories can make a normal
    # humanoid tiny in frame.
    min_z -= 0.08 * bone_height
    max_z += 0.18 * bone_height
    min_x -= 0.08 * bone_width
    max_x += 0.08 * bone_width
    character_height = max_z-min_z
    character_width = max_x-min_x
    aspect = args.width / args.height
    # Blender's orthographic scale is horizontal in a wide render. Ensure both
    # the vertical body height and horizontal T-pose width fit with 15% margin.
    ortho_scale = max(character_width, character_height * aspect) * 1.05

    camera_data = bpy.data.cameras.new("VRM_Video_Camera")
    camera = bpy.data.objects.new("VRM_Video_Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = ortho_scale
    camera.location = Vector((0.5*(min_x+max_x), min_y-max(4.0, 2.0*character_height), 0.5*(min_z+max_z)))
    camera.rotation_euler = (math.radians(90), 0, 0)

    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(frames_dir / "frame_")
    scene.render.film_transparent = False
    bpy.ops.render.render(animation=True)
    print(f"VRM_VIDEO_FRAMES={frames_dir}")


if __name__ == "__main__":
    main()
