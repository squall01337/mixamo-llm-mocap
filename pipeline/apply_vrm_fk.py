"""Reusable native VRM 1.0 adapter: semantic lifted motion -> Blender action.

This adapter consumes the pipeline's semantic joint-position stream directly.
It never reads or converts a Mixamo Blender action.  The current lift output is
accepted as the transitional stream because its joint fields are semantic; all
target proportions, rest directions, bone names, and ground measurements come
from the selected VRM.

Usage (Blender 5.1+):

  blender --background --python pipeline/apply_vrm_fk.py -- ^
    --spec action_specs/baseline_dance.json ^
    --vrm "assets/vrm/My Avatar.vrm" ^
    --out-dir clips/baseline_dance_vrm

Optional: --profile-out, --blend-out, --action-name, --qa-frames.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

REPO = Path(__file__).resolve().parents[1]

REQUIRED_ROLES = {
    "hips", "spine", "chest", "neck", "head",
    "leftUpperArm", "leftLowerArm", "leftHand",
    "rightUpperArm", "rightLowerArm", "rightHand",
    "leftUpperLeg", "leftLowerLeg", "leftFoot",
    "rightUpperLeg", "rightLowerLeg", "rightFoot",
}

SEMANTIC_CHILDREN = {
    "spine": "chest", "chest": "upperChest", "upperChest": "neck", "neck": "head",
    "leftShoulder": "leftUpperArm", "leftUpperArm": "leftLowerArm", "leftLowerArm": "leftHand",
    "rightShoulder": "rightUpperArm", "rightUpperArm": "rightLowerArm", "rightLowerArm": "rightHand",
    "leftUpperLeg": "leftLowerLeg", "leftLowerLeg": "leftFoot", "leftFoot": "leftToes",
    "rightUpperLeg": "rightLowerLeg", "rightLowerLeg": "rightFoot", "rightFoot": "rightToes",
}

# VRM role, semantic stream segment start, semantic stream segment end.
CHAINS = [
    ("spine", "spine", "spine1"), ("chest", "spine1", "spine2"),
    ("upperChest", "spine2", "neck"), ("neck", "neck", "head"),
    ("leftShoulder", "l_shoulder", "l_arm"),
    ("leftUpperArm", "l_arm", "l_elbow"), ("leftLowerArm", "l_elbow", "l_wrist"),
    ("rightShoulder", "r_shoulder", "r_arm"),
    ("rightUpperArm", "r_arm", "r_elbow"), ("rightLowerArm", "r_elbow", "r_wrist"),
    ("leftUpperLeg", "l_upleg", "l_knee"), ("leftLowerLeg", "l_knee", "l_ankle"),
    ("leftFoot", "l_ankle", "l_foot"),
    ("rightUpperLeg", "r_upleg", "r_knee"), ("rightLowerLeg", "r_knee", "r_ankle"),
    ("rightFoot", "r_ankle", "r_foot"),
]


def rpath(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def glb_json(path: Path) -> dict:
    with path.open("rb") as file:
        magic, version, _length = struct.unpack("<4sII", file.read(12))
        if magic != b"glTF" or version != 2:
            raise RuntimeError(f"not a GLB 2 file: {path}")
        size, kind = struct.unpack("<II", file.read(8))
        if kind != 0x4E4F534A:
            raise RuntimeError("first GLB chunk is not JSON")
        return json.loads(file.read(size).decode("utf-8").rstrip(" \t\r\n\0"))


def humanoid_mapping(doc: dict) -> dict[str, str]:
    extension = doc.get("extensions", {}).get("VRMC_vrm")
    if not extension:
        raise RuntimeError("VRM 1.0 VRMC_vrm metadata not found (VRM 0.x is not supported yet)")
    nodes = doc["nodes"]
    mapping = {
        role: nodes[int(value["node"])].get("name", f"node_{value['node']}")
        for role, value in extension["humanoid"]["humanBones"].items()
    }
    missing = sorted(REQUIRED_ROLES - mapping.keys())
    if missing:
        raise RuntimeError(f"VRM is missing required humanoid roles: {missing}")
    return mapping


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def world_head(arm, bone: str) -> Vector:
    return arm.matrix_world @ arm.pose.bones[bone].matrix.to_translation()


def reset_pose(arm) -> None:
    for pose_bone in arm.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.rotation_quaternion = Quaternion()
        pose_bone.location = Vector()
        pose_bone.scale = Vector((1, 1, 1))
    bpy.context.view_layer.update()


def measure_profile(arm, mapping: dict[str, str], vrm: Path) -> tuple[dict, dict[str, Vector]]:
    reset_pose(arm)
    missing = {role: name for role, name in mapping.items() if name not in arm.pose.bones}
    if missing:
        raise RuntimeError(f"VRM metadata nodes did not import as pose bones: {missing}")
    rest_directions = {}
    for role, child_role in SEMANTIC_CHILDREN.items():
        if role in mapping and child_role in mapping:
            rest_directions[mapping[role]] = (
                arm.pose.bones[mapping[child_role]].head - arm.pose.bones[mapping[role]].head
            )
    rest_heads = {role: list(world_head(arm, bone)) for role, bone in mapping.items()}
    left_ground = world_head(arm, mapping["leftFoot"]).z
    right_ground = world_head(arm, mapping["rightFoot"]).z
    profile = {
        "format": "vrm-rig-profile-0.2",
        "source": str(vrm),
        "armature": arm.name,
        "humanoid_bones": mapping,
        "rest_heads_world": rest_heads,
        "rest_directions_armature": {bone: list(direction) for bone, direction in rest_directions.items()},
        "rest_hips": list(world_head(arm, mapping["hips"])),
        "ground": {
            "leftFoot": left_ground, "rightFoot": right_ground,
            "policy": "semantic foot-bone head at authored rest height",
        },
    }
    return profile, rest_directions


def aim(arm, bone: str, target_world: Vector, rest_directions: dict[str, Vector]) -> None:
    pose_bone = arm.pose.bones[bone]
    pose_bone.rotation_mode = "QUATERNION"
    pose_bone.rotation_quaternion = Quaternion()
    pose_bone.location = Vector()
    bpy.context.view_layer.update()
    head = pose_bone.matrix.to_translation()
    direction = arm.matrix_world.inverted() @ target_world - head
    if direction.length < 1e-7:
        return
    parent_deform = (
        pose_bone.parent.matrix @ pose_bone.parent.bone.matrix_local.inverted()
        if pose_bone.parent else Matrix.Identity(4)
    )
    rest_direction = rest_directions.get(bone, pose_bone.bone.vector).normalized()
    current_rest_direction = (parent_deform.to_3x3() @ rest_direction).normalized()
    delta = current_rest_direction.rotation_difference(direction.normalized())
    matrix = delta.to_matrix().to_4x4() @ pose_bone.matrix
    matrix.translation = head
    pose_bone.matrix = matrix
    bpy.context.view_layer.update()


def set_hips(arm, bone: str, frame: dict, scale: float,
             stream_rest_hips: Vector, rig_rest_hips: Vector) -> None:
    pose_bone = arm.pose.bones[bone]
    pose_bone.rotation_mode = "QUATERNION"
    pose_bone.rotation_quaternion = Quaternion()
    pose_bone.location = Vector()
    bpy.context.view_layer.update()
    bx, by, bz = Vector(frame["basis_x"]), Vector(frame["basis_y"]), Vector(frame["basis_z"])
    world_basis = Matrix(((bx.x, by.x, bz.x), (bx.y, by.y, bz.y), (bx.z, by.z, bz.z)))
    arm_basis = arm.matrix_world.to_3x3().inverted() @ world_basis @ arm.matrix_world.to_3x3()
    matrix = arm_basis.to_4x4() @ pose_bone.bone.matrix_local
    matrix.translation = pose_bone.bone.matrix_local.translation
    pose_bone.matrix = matrix
    bpy.context.view_layer.update()
    rotation = pose_bone.rotation_quaternion.copy()
    desired = rig_rest_hips + (Vector(frame["hips"]) - stream_rest_hips) * scale
    delta = arm.matrix_world.to_3x3().inverted() @ (desired - world_head(arm, bone))
    pose_bone.location += delta
    pose_bone.rotation_quaternion = rotation
    bpy.context.view_layer.update()


def shift_hips_world(arm, hips_bone: str, delta_world: Vector) -> None:
    pose_bone = arm.pose.bones[hips_bone]
    matrix = pose_bone.matrix.copy()
    matrix.translation += arm.matrix_world.to_3x3().inverted() @ delta_world
    pose_bone.matrix = matrix
    bpy.context.view_layer.update()


def plant_hips(arm, hips_bone: str, foot_bone: str, ground_z: float) -> None:
    shift_hips_world(arm, hips_bone, Vector((0, 0, ground_z - world_head(arm, foot_bone).z)))


def flatten_support_foot(arm, foot_bone: str, source_heading: Vector,
                         rest_directions: dict[str, Vector]) -> None:
    rest = rest_directions[foot_bone]
    horizontal_length = math.sqrt(max(0.0, rest.length_squared - rest.z * rest.z))
    heading = Vector((source_heading.x, source_heading.y, 0.0))
    if heading.length < 1e-6:
        heading = Vector((rest.x, rest.y, 0.0))
    if heading.length < 1e-6:
        return
    heading.normalize()
    ankle = world_head(arm, foot_bone)
    target = ankle + heading * horizontal_length + Vector((0, 0, rest.z))
    aim(arm, foot_bone, target, rest_directions)


def ensure_action(arm, name: str, frame_count: int, fps: int):
    if arm.animation_data is None:
        arm.animation_data_create()
    existing = bpy.data.actions.get(name)
    if existing:
        bpy.data.actions.remove(existing)
    action = bpy.data.actions.new(name)
    arm.animation_data.action = action
    scene = bpy.context.scene
    scene.render.fps = fps
    scene.frame_start, scene.frame_end = 1, frame_count
    return action


def key_pose(arm, hips_bone: str, frame: int) -> None:
    for pose_bone in arm.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.keyframe_insert("rotation_quaternion", frame=frame)
        if pose_bone.name == hips_bone:
            pose_bone.keyframe_insert("location", frame=frame)


def set_linear(action) -> None:
    try:
        for layer in action.layers:
            for strip in layer.strips:
                for slot in action.slots:
                    bag = strip.channelbag(slot)
                    if bag:
                        for curve in bag.fcurves:
                            for key in curve.keyframe_points:
                                key.interpolation = "LINEAR"
    except Exception:
        for curve in getattr(action, "fcurves", []):
            for key in curve.keyframe_points:
                key.interpolation = "LINEAR"


def render_stills(scene, arm, hips_bone: str, frames: list[int], out_dir: Path) -> list[str]:
    pose_dir = out_dir / "poses"
    pose_dir.mkdir(parents=True, exist_ok=True)
    camera_data = bpy.data.cameras.new("VRM_QA_Camera")
    camera = bpy.data.objects.new("VRM_QA_Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.render.resolution_x = scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 2.25
    written = []
    for frame in frames:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        hips = world_head(arm, hips_bone)
        for view, location, rotation in (
            ("front", (hips.x, hips.y - 4.0, 1.0), (math.radians(90), 0, 0)),
            ("side", (hips.x + 4.0, hips.y, 1.0), (math.radians(90), 0, math.radians(90))),
        ):
            camera.location, camera.rotation_euler = location, rotation
            destination = pose_dir / f"{view}_{frame:04d}.png"
            scene.render.filepath = str(destination)
            bpy.ops.render.render(write_still=True)
            written.append(str(destination))
    return written


def validate_contacts(scene, arm, mapping: dict[str, str], frames: list[dict],
                      left_ground: float, right_ground: float) -> dict:
    errors = {"left": [], "right": [], "both_lower": []}
    positions = {"left": [], "right": []}
    for frame in frames:
        if float(frame.get("rest_amount", 0.0)) >= 0.65:
            continue
        scene.frame_set(int(frame["frame"]))
        left = world_head(arm, mapping["leftFoot"])
        right = world_head(arm, mapping["rightFoot"])
        support = frame.get("plant", "both")
        if support == "left":
            errors["left"].append(abs(left.z - left_ground)); positions["left"].append(left.xy)
        elif support == "right":
            errors["right"].append(abs(right.z - right_ground)); positions["right"].append(right.xy)
        elif support == "both":
            errors["both_lower"].append(min(abs(left.z-left_ground), abs(right.z-right_ground)))
    result = {key: {"frames": len(values), "max_ground_error_m": max(values, default=0.0)}
              for key, values in errors.items()}
    for side, values in positions.items():
        anchor = values[0] if values else Vector()
        result[side]["max_xy_wander_m"] = max(((value-anchor).length for value in values), default=0.0)
    return result


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--vrm", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--profile-out")
    parser.add_argument("--blend-out")
    parser.add_argument("--action-name")
    parser.add_argument("--qa-frames", help="comma-separated destination frames")
    parser.add_argument("--no-stills", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    spec_path, vrm, out_dir = rpath(args.spec), rpath(args.vrm), rpath(args.out_dir)
    if not spec_path.exists() or not vrm.exists():
        raise SystemExit(f"missing input: spec={spec_path.exists()} vrm={vrm.exists()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    stream_path = rpath(spec["joints_out"])
    payload = json.loads(stream_path.read_text(encoding="utf-8"))
    frames = payload["frames"]
    if not frames:
        raise RuntimeError("motion stream has no frames")

    mapping = humanoid_mapping(glb_json(vrm))
    clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(vrm))
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"expected one armature, found {[obj.name for obj in armatures]}")
    arm = armatures[0]
    profile, rest_directions = measure_profile(arm, mapping, vrm)
    profile_path = rpath(args.profile_out) if args.profile_out else out_dir / "vrm_rig_profile.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    hips_bone = mapping["hips"]
    rig_rest_hips = Vector(profile["rest_hips"])
    stream_rest_frame = max(frames, key=lambda frame: float(frame.get("rest_amount", 0.0)))
    stream_rest_hips = Vector(stream_rest_frame["hips"])
    if abs(stream_rest_hips.z) < 1e-6:
        raise RuntimeError("motion stream rest hips height is zero")
    scale = rig_rest_hips.z / stream_rest_hips.z
    left_ground = float(profile["ground"]["leftFoot"])
    right_ground = float(profile["ground"]["rightFoot"])
    action_name = args.action_name or f"{spec.get('action_name', spec.get('name', 'Motion'))}_VRM"
    action = ensure_action(arm, action_name, len(frames), int(round(payload.get("dst_fps", 30))))
    chains = [(mapping[role], start, end) for role, start, end in CHAINS if role in mapping]

    active_lock_side, foot_anchor = None, None
    for frame in frames:
        reset_pose(arm)
        rest = float(frame.get("rest_amount", 0.0))
        if rest < 0.65:
            set_hips(arm, hips_bone, frame, scale, stream_rest_hips, rig_rest_hips)
            for bone, start_key, end_key in chains:
                direction = Vector(frame[end_key]) - Vector(frame[start_key])
                if direction.length > 1e-7:
                    aim(arm, bone, world_head(arm, bone) + direction.normalized(), rest_directions)
            support = frame.get("plant", "both")
            if support == "left":
                plant_hips(arm, hips_bone, mapping["leftFoot"], left_ground)
                flatten_support_foot(arm, mapping["leftFoot"], Vector(frame["l_toe"])-Vector(frame["l_foot"]), rest_directions)
            elif support == "right":
                plant_hips(arm, hips_bone, mapping["rightFoot"], right_ground)
                flatten_support_foot(arm, mapping["rightFoot"], Vector(frame["r_toe"])-Vector(frame["r_foot"]), rest_directions)
            elif support == "both":
                left_error = abs(world_head(arm, mapping["leftFoot"]).z-left_ground)
                right_error = abs(world_head(arm, mapping["rightFoot"]).z-right_ground)
                side = "left" if left_error <= right_error else "right"
                foot = mapping[f"{side}Foot"]
                ground = left_ground if side == "left" else right_ground
                prefix = "l" if side == "left" else "r"
                plant_hips(arm, hips_bone, foot, ground)
                flatten_support_foot(arm, foot, Vector(frame[f"{prefix}_toe"])-Vector(frame[f"{prefix}_foot"]), rest_directions)
            if support in ("left", "right"):
                foot = mapping[f"{support}Foot"]
                position = world_head(arm, foot)
                if active_lock_side != support or foot_anchor is None:
                    active_lock_side, foot_anchor = support, position.copy()
                else:
                    shift_hips_world(arm, hips_bone, Vector((foot_anchor.x-position.x, foot_anchor.y-position.y, 0)))
            else:
                active_lock_side, foot_anchor = None, None
        key_pose(arm, hips_bone, int(frame["frame"]))
    set_linear(action)

    scene = bpy.context.scene
    contact_qa = validate_contacts(scene, arm, mapping, frames, left_ground, right_ground)
    if args.qa_frames:
        qa_frames = [int(value) for value in args.qa_frames.split(",")]
    else:
        src_fps, dst_fps = float(spec.get("src_fps", 24)), float(payload.get("dst_fps", 30))
        qa_frames = sorted({max(1, min(len(frames), round((src-1)*dst_fps/src_fps+1)))
                            for src in spec.get("qa_src_frames", [1, len(frames)])})
    stills = [] if args.no_stills else render_stills(scene, arm, hips_bone, qa_frames, out_dir)
    blend_path = rpath(args.blend_out) if args.blend_out else out_dir / f"{spec.get('name', 'motion')}_vrm_animated.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    result = {
        "status": "ok", "adapter": "native-vrm-1.0", "vrm": str(vrm),
        "motion_stream": str(stream_path), "profile": str(profile_path),
        "blend": str(blend_path), "action": action_name, "frames": len(frames),
        "scale": scale, "contact_qa": contact_qa, "qa_stills": stills,
    }
    result_path = out_dir / "vrm_adapter_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("VRM_ADAPTER_RESULT", json.dumps(result))


if __name__ == "__main__":
    main()
