"""Generic FK apply: lifted joints -> live Y Bot action. Runs inside
Blender 5.1 (via the official Blender MCP, or pipeline/blender_exec.py).

Per-frame the joints file carries the support foot ("plant": left /
right / both / none), the fist amount, the rest-blend amount and the
estimator's absolute pelvis height. This module aims each mixamorig
bone at its lifted joint (armature space, FK only — no constraints, no
IK), then resolves the hip height:

  left/right  binary-search hips Y so that foot lands at GROUND_Z,
              then flatten it (Z-only, keeps estimator XZ + heading)
  both        plant the lower foot, flatten both if near ground
  none        airborne: integrate the estimator's pelvis arc from the
              last applied height (continuous takeoff, ballistic flight)
  rest>=0.65  exact rest height (start/end T-pose)

Usage inside Blender:
    import apply_mixamo_fk
    result = apply_mixamo_fk.run("action_specs/<motion>.json")
Then, separately (keeps the long keying pass in one socket request):
    apply_mixamo_fk.dump_curves("action_specs/<motion>.json")
    apply_mixamo_fk.run_stills_render("action_specs/<motion>.json", [1, ...])
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
from mathutils import Quaternion, Vector

REPO = Path(__file__).resolve().parents[1]
GROUND_Z = 0.105      # Y Bot rest ankle height (flat-foot contact)
HIP_HEIGHT = 0.99792  # Y Bot rest hip height
FPS = 30


def use_profile(path=None) -> str:
    """Adopt one character's measured ground/hip heights.

    Solo scenes fall back to rig_profile.json at the repo root; a
    multi-character scene names a profile per character in its spec
    (`rig_profile`), because two characters in one file do NOT share a
    hip height and planting the Ninja at the Y Bot's ground offset
    buries its feet.
    """
    global GROUND_Z, HIP_HEIGHT
    p = (Path(path) if Path(path).is_absolute() else REPO / path) if path else (REPO / "rig_profile.json")
    if not p.exists():
        if path:
            raise RuntimeError(f"rig profile not found: {p}")
        return "built-in Y Bot"
    _p = json.loads(p.read_text(encoding="utf-8"))
    GROUND_Z = float(_p["ground_z"])
    HIP_HEIGHT = float(_p["hip_height"])
    return p.name


use_profile()

AIM = [
    ("mixamorig:Spine", "spine1"),
    ("mixamorig:Spine1", "spine2"),
    ("mixamorig:Spine2", "neck"),
    ("mixamorig:Neck", "head"),
    ("mixamorig:Head", "head"),
    ("mixamorig:LeftShoulder", "l_arm"),
    ("mixamorig:LeftArm", "l_elbow"),
    ("mixamorig:LeftForeArm", "l_wrist"),
    ("mixamorig:LeftHand", "l_hand"),
    ("mixamorig:RightShoulder", "r_arm"),
    ("mixamorig:RightArm", "r_elbow"),
    ("mixamorig:RightForeArm", "r_wrist"),
    ("mixamorig:RightHand", "r_hand"),
    ("mixamorig:LeftUpLeg", "l_knee"),
    ("mixamorig:LeftLeg", "l_ankle"),
    ("mixamorig:LeftFoot", "l_foot"),
    ("mixamorig:LeftToeBase", "l_toe"),
    ("mixamorig:RightUpLeg", "r_knee"),
    ("mixamorig:RightLeg", "r_ankle"),
    ("mixamorig:RightFoot", "r_foot"),
    ("mixamorig:RightToeBase", "r_toe"),
]


def _fist_quats() -> dict:
    """Original procedural fist (no third-party motion data).

    Finger curl on this rig is a positive rotation about the bone-local
    X axis for both hands (80 deg base, 95 deg mid, 50 deg tip). Thumb
    bones bend on a DIFFERENT axis mix: the fold that wraps the thumb
    across the curled fingers is mostly negative-X plus a y/z opposition
    component that mirrors between hands. Positive-X on a thumb
    hyperextends it outward — the original bug this replaces. Values
    validated visually on the rig (palm/top/front close-up renders).
    """
    seg1 = (0.766044, 0.642788, 0.0, 0.0)   # 80 deg about X
    seg2 = (0.67559, 0.737277, 0.0, 0.0)    # 95 deg about X
    seg3 = (0.906308, 0.422618, 0.0, 0.0)   # 50 deg about X
    out = {}
    for side, zs in (("Left", -1.0), ("Right", 1.0)):
        out[f"mixamorig:{side}HandThumb1"] = (0.961, 0.069, 0.165 * zs, 0.207 * zs)
        out[f"mixamorig:{side}HandThumb2"] = (0.766, -0.399, 0.161 * zs, 0.476 * zs)
        out[f"mixamorig:{side}HandThumb3"] = (0.940, -0.210, 0.136 * zs, 0.231 * zs)
        for fn in ("Index", "Middle", "Ring", "Pinky"):
            out[f"mixamorig:{side}Hand{fn}1"] = seg1
            out[f"mixamorig:{side}Hand{fn}2"] = seg2
            out[f"mixamorig:{side}Hand{fn}3"] = seg3
    return out


FIST = _fist_quats()


def rpath(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO / p)


def get_armature(spec: dict):
    """The armature this spec animates. Multi-character scenes name it
    (`"armature": "Armature_YBot"`); solo scenes keep "Armature"."""
    name = spec.get("armature", "Armature")
    arm = bpy.data.objects.get(name)
    if arm is None or arm.type != "ARMATURE":
        have = [o.name for o in bpy.data.objects if o.type == "ARMATURE"]
        raise RuntimeError(f"armature {name!r} not in the scene (have: {have})")
    return arm


def bind_action(arm, spec: dict):
    """Bind this spec's action to its armature and return it.

    The live scene holds whatever action was applied LAST (docs/PITFALLS.md
    #16). Anything that reads the pose back — curves, stills, mesh contact —
    must bind the clip it is asked about first, or it silently measures
    another clip. Same binding as render_preview.py.
    """
    act = bpy.data.actions.get(spec["action_name"])
    if act is None:
        raise RuntimeError(f"action {spec['action_name']!r} not in the scene - run the apply first")
    if arm.animation_data is None:
        arm.animation_data_create()
    arm.animation_data.action = act
    try:
        for slot in act.slots:
            arm.animation_data.action_slot = slot
            break
    except Exception:
        pass
    return act


def v(seq) -> Vector:
    return Vector((float(seq[0]), float(seq[1]), float(seq[2])))


def world_loc(arm, name: str) -> Vector:
    pb = arm.pose.bones[name]
    return (arm.matrix_world @ pb.matrix).to_translation()


def reset_pose(arm) -> None:
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
        pb.location = Vector((0.0, 0.0, 0.0))
        pb.scale = Vector((1.0, 1.0, 1.0))
    bpy.context.view_layer.update()


def to_arm(arm, world_m: Vector) -> Vector:
    return arm.matrix_world.inverted() @ world_m


def dir_to_arm(arm, world_dir: Vector) -> Vector:
    return (arm.matrix_world.to_3x3().inverted() @ world_dir).normalized()


def aim_bone(arm, name: str, target_world: Vector) -> None:
    """Aim bone +Y at a world-meter target. All math in ARMATURE space
    (centimeters) — never build pose matrices through matrix_world."""
    pb = arm.pose.bones[name]
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
    if name != "mixamorig:Hips":
        pb.location = Vector((0.0, 0.0, 0.0))
    bpy.context.view_layer.update()
    head = pb.matrix.to_translation()
    target = to_arm(arm, target_world)
    direction = target - head
    if direction.length < 1e-5:
        return
    rot = pb.y_axis.normalized().rotation_difference(direction.normalized())
    mat = rot.to_matrix().to_4x4() @ pb.matrix.to_3x3().to_4x4()
    mat.translation = head
    pb.matrix = mat
    bpy.context.view_layer.update()
    q = pb.rotation_quaternion.copy()
    if name != "mixamorig:Hips":
        pb.location = Vector((0.0, 0.0, 0.0))
    pb.rotation_quaternion = q
    bpy.context.view_layer.update()


def set_hips(arm, basis_x, basis_y, basis_z, loc_world) -> None:
    """Hips rotation from the lifted basis; location in Mixamo pose-cm
    (x -> world X, y -> world Z height, z -> world -Y forward)."""
    pb = arm.pose.bones["mixamorig:Hips"]
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
    pb.location = Vector((0.0, 0.0, 0.0))
    bpy.context.view_layer.update()
    up_arm = dir_to_arm(arm, basis_z)
    left_arm = dir_to_arm(arm, basis_x)
    rot_y = pb.y_axis.normalized().rotation_difference(up_arm)
    mat = rot_y.to_matrix().to_4x4() @ pb.matrix.to_3x3().to_4x4()
    mat.translation = pb.matrix.to_translation()
    pb.matrix = mat
    bpy.context.view_layer.update()
    x_now = pb.x_axis.normalized()
    left_flat = left_arm - up_arm * left_arm.dot(up_arm)
    if left_flat.length > 1e-5:
        twist = x_now.rotation_difference(left_flat.normalized())
        axis, ang = twist.to_axis_angle()
        if axis.dot(pb.y_axis) < 0.0:
            ang = -ang
        twist_y = Quaternion(pb.y_axis.normalized(), ang)
        mat = twist_y.to_matrix().to_4x4() @ pb.matrix.to_3x3().to_4x4()
        mat.translation = pb.matrix.to_translation()
        pb.matrix = mat
        bpy.context.view_layer.update()
    q = pb.rotation_quaternion.copy()
    pb.location = Vector(
        (loc_world.x * 100.0, (loc_world.z - HIP_HEIGHT) * 100.0, -loc_world.y * 100.0)
    )
    pb.rotation_quaternion = q
    bpy.context.view_layer.update()


def plant_hips(arm, hint_y: float, loc_x: float, loc_z: float, foot: str) -> float:
    """Binary-search hips.location.y (cm) until *foot* lands at GROUND_Z."""
    hips = arm.pose.bones["mixamorig:Hips"]
    lo, hi = hint_y - 55.0, hint_y + 40.0
    best_y, best_err = hint_y, 1e9
    for _ in range(15):
        mid = 0.5 * (lo + hi)
        hips.location = (loc_x, mid, loc_z)
        bpy.context.view_layer.update()
        zf = world_loc(arm, foot).z
        err = abs(zf - GROUND_Z)
        if err < best_err:
            best_err, best_y = err, mid
        if zf > GROUND_Z:
            hi = mid
        else:
            lo = mid
    hips.location = (loc_x, best_y, loc_z)
    bpy.context.view_layer.update()
    return best_y


def flatten_foot(arm, side: str) -> None:
    """Z-only ground snap for a near-ground foot. Keeps estimator XZ and
    the foot's own heading (never a world axis).

    The toe aim needs a target FORWARD of the ball along the foot
    heading — aiming ToeBase at its own head position is degenerate and
    bends the toes randomly.
    """
    foot_b = f"mixamorig:{side}Foot"
    toe_b = f"mixamorig:{side}ToeBase"
    ankle = world_loc(arm, foot_b)
    ball = world_loc(arm, toe_b)
    if ankle.z > 0.20:
        return  # genuinely lifted — leave it alone
    heading = Vector((ball.x - ankle.x, ball.y - ankle.y, 0.0))
    if heading.length < 0.02:
        pb = arm.pose.bones[foot_b]
        y_w = (arm.matrix_world.to_3x3() @ pb.y_axis)
        heading = Vector((y_w.x, y_w.y, 0.0))
    if heading.length < 1e-5:
        return
    heading.normalize()
    aim_bone(arm, foot_b, Vector((ball.x, ball.y, 0.034)))
    ball2 = world_loc(arm, toe_b)
    aim_bone(arm, toe_b, Vector((ball2.x + heading.x * 0.06, ball2.y + heading.y * 0.06, 0.028)))


def apply_fingers(arm, amount: float) -> None:
    ident = Quaternion((1.0, 0.0, 0.0, 0.0))
    for name, quat in FIST.items():
        if name not in arm.pose.bones:
            continue
        pb = arm.pose.bones[name]
        pb.rotation_mode = "QUATERNION"
        fq = Quaternion(quat)
        fq.normalize()
        pb.rotation_quaternion = ident.slerp(fq, amount)
        pb.location = Vector((0.0, 0.0, 0.0))


def _face_dir(arm, pb) -> Vector:
    bpy.context.view_layer.update()
    return ((arm.matrix_world @ pb.matrix).to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()


def _rotate_bone(arm, pb, axis_world: Vector, delta: float) -> None:
    if abs(delta) < 1e-5 or axis_world.length < 1e-6:
        return
    axis_arm = dir_to_arm(arm, axis_world.normalized())
    head = pb.matrix.to_translation()
    mat = Quaternion(axis_arm, delta).to_matrix().to_4x4() @ pb.matrix.to_3x3().to_4x4()
    mat.translation = head
    pb.matrix = mat
    bpy.context.view_layer.update()
    q = pb.rotation_quaternion.copy()
    pb.location = Vector((0.0, 0.0, 0.0))
    pb.rotation_quaternion = q
    bpy.context.view_layer.update()


def _solve_axis(arm, pb, axis_world: Vector, measure, target: float, amount: float) -> None:
    """Probe-solve-refine one rotation axis. The head's frame is heavily
    yawed during spins, so the sign of a correction cannot be assumed."""
    e0 = measure()
    want = (target - e0) * amount
    if abs(want) < 1e-4:
        return
    probe = math.copysign(0.05, want)
    _rotate_bone(arm, pb, axis_world, probe)
    e1 = measure()
    slope = (e1 - e0) / probe
    if abs(slope) < 0.1:
        _rotate_bone(arm, pb, axis_world, -probe)
        return
    _rotate_bone(arm, pb, axis_world, ((target - e1) * amount) / slope)
    resid = (target - measure()) * amount
    if abs(resid) > math.radians(0.5):
        _rotate_bone(arm, pb, axis_world, resid / slope)


def orient_face(arm, name: str, gaze, amount: float, max_offset_deg: float = 75.0,
                body_forward: Vector | None = None) -> None:
    """Point a head bone's FACE axis along `gaze` (world direction).

    The FK apply aims only the skull axis (+Y), leaving the face's yaw
    and pitch to whatever the torso is doing — a head that follows its
    chest through every twist. The estimator's real gaze fixes that.
    `max_offset_deg` keeps the neck anatomically sane when the body is
    turned far from where the head looks.
    """
    if amount <= 1e-4 or gaze is None:
        return
    g = Vector((float(gaze[0]), float(gaze[1]), float(gaze[2])))
    if g.length < 1e-6:
        return
    g.normalize()
    pb = arm.pose.bones[name]
    if body_forward is not None and body_forward.length > 1e-6:
        bf = Vector((body_forward.x, body_forward.y, 0.0))
        gf = Vector((g.x, g.y, 0.0))
        if bf.length > 1e-6 and gf.length > 1e-6:
            bf.normalize()
            gf.normalize()
            off = math.atan2(bf.x * gf.y - bf.y * gf.x, bf.x * gf.x + bf.y * gf.y)
            lim = math.radians(max_offset_deg)
            if abs(off) > lim:
                clamped = math.copysign(lim, off)
                q = Quaternion(Vector((0.0, 0.0, 1.0)), clamped - off)
                g = q @ g
    # Yaw about world up, then pitch about the horizontal axis across the gaze.
    tgt_yaw = math.atan2(g.x, -g.y)
    _solve_axis(arm, pb, Vector((0.0, 0.0, 1.0)),
                lambda: math.atan2(_face_dir(arm, pb).x, -_face_dir(arm, pb).y),
                tgt_yaw, amount)
    lateral = Vector((-g.y, g.x, 0.0))
    _solve_axis(arm, pb, lateral,
                lambda: math.asin(max(-1.0, min(1.0, _face_dir(arm, pb).z))),
                math.asin(max(-1.0, min(1.0, g.z))), amount)


def ensure_action(arm, name: str, end: int):
    if arm.animation_data is None:
        arm.animation_data_create()
    if name in bpy.data.actions:
        bpy.data.actions.remove(bpy.data.actions[name])
    action = bpy.data.actions.new(name)
    arm.animation_data.action = action
    try:
        slot = action.slots.new(id_type="OBJECT", name="Armature")
        arm.animation_data.action_slot = slot
    except Exception:
        pass
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = end
    scene.frame_set(1)
    return action


def key_pose(arm, f: int) -> None:
    for pb in arm.pose.bones:
        if pb.name.endswith("_End") or pb.name.endswith("4"):
            continue
        pb.rotation_mode = "QUATERNION"
        pb.keyframe_insert("rotation_quaternion", frame=f)
        if pb.name == "mixamorig:Hips":
            pb.keyframe_insert("location", frame=f)
        else:
            pb.location = Vector((0.0, 0.0, 0.0))
            pb.keyframe_insert("location", frame=f)


def set_linear(action) -> None:
    """Blender 5.x slotted actions: write through the channelbag."""
    try:
        for layer in action.layers:
            for strip in layer.strips:
                for slot in action.slots:
                    bag = strip.channelbag(slot)
                    if not bag:
                        continue
                    for fc in bag.fcurves:
                        for kp in fc.keyframe_points:
                            kp.interpolation = "LINEAR"
    except Exception:
        if getattr(action, "fcurves", None):
            for fc in action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"


def measure(arm, f: int) -> dict:
    bpy.context.scene.frame_set(f)
    bpy.context.view_layer.update()

    def w(n):
        p = world_loc(arm, n)
        return [round(p.x, 4), round(p.y, 4), round(p.z, 4)]

    return {
        "frame": f,
        "hips": w("mixamorig:Hips"),
        "l_hand": w("mixamorig:LeftHand"),
        "r_hand": w("mixamorig:RightHand"),
        "l_foot": w("mixamorig:LeftFoot"),
        "r_foot": w("mixamorig:RightFoot"),
        "l_toe": w("mixamorig:LeftToeBase"),
        "r_toe": w("mixamorig:RightToeBase"),
        "head": w("mixamorig:Head"),
    }


def _pause_deform(arm) -> list:
    """Switch off (viewport only) the Armature modifiers that skin this
    armature's meshes; return them so the caller can switch them back.

    The keying pass calls view_layer.update() ~100 times per frame (every
    aim, the hip search, the gaze solve) and reads nothing but BONE
    matrices, yet each update also re-skins every mesh the armature
    deforms. Pose evaluation does not depend on the meshes, so the keyed
    numbers are identical with the skinning paused.
    """
    paused = []
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        for m in o.modifiers:
            if (m.type == "ARMATURE" and m.show_viewport and m.object is not None
                    and m.object.name == arm.name):
                m.show_viewport = False
                paused.append(m)
    return paused


def _key_frames(arm, spec: dict, frames: list, src_fps: float, dst_fps: float) -> None:
    """The per-frame solve + keying pass (see the module docstring)."""
    # Optional per-window amplitude boost for airborne arcs (the
    # estimator tends to understate jump height; `"boost": 1.3` on a
    # `none` plant window scales the integrated pelvis deltas).
    boost_by_frame = {}
    for wnd in spec.get("plant", []):
        if wnd.get("support") == "none" and wnd.get("boost"):
            a = int(round((wnd["src"][0] - 1) * dst_fps / src_fps + 1))
            b = int(round((wnd["src"][1] - 1) * dst_fps / src_fps + 1))
            for f_ in range(a, b + 1):
                boost_by_frame[f_] = float(wnd["boost"])

    prev_hip_y = 0.0
    prev_ph = None
    for fr in frames:
        f = int(fr["frame"])
        rest = float(fr.get("rest_amount", 0.0))
        fist = float(fr.get("fist_amount", 0.0))
        plant = fr.get("plant", "both")
        ph = float(fr.get("pelvis_height", 0.0))
        if prev_ph is None:
            prev_ph = ph
        reset_pose(arm)
        bx, by, bz = v(fr["basis_x"]), v(fr["basis_y"]), v(fr["basis_z"])
        hips_w = v(fr["hips"])
        set_hips(arm, bx, by, bz, hips_w)
        loc_x = hips_w.x * 100.0
        loc_z = -hips_w.y * 100.0
        hint = (hips_w.z - HIP_HEIGHT) * 100.0
        up = v(fr["basis_z"])
        neck_p = v(fr["neck"])
        head_flat = v(fr["head"]) - neck_p
        head_flat.z = 0.0
        # The head aim is rebuilt from the torso up-axis, so the lifted
        # head's vertical is intentionally ignored — except for an
        # explicit per-frame gaze pitch (spec head_look.pitch_deg), which
        # rotates the aim about the character's lateral axis.
        pitch = math.radians(float(fr.get("head_pitch_deg", 0.0) or 0.0))
        lateral = v(fr["basis_x"])
        for bone, key in AIM:
            if bone == "mixamorig:Neck":
                aim = up * 0.11 + head_flat * 0.25
                if pitch:
                    aim = aim.copy()
                    aim.rotate(Quaternion(lateral, pitch * 0.35))
                aim_bone(arm, bone, neck_p + aim)
            elif bone == "mixamorig:Head":
                aim = up * 0.28 + head_flat * 0.15
                if pitch:
                    aim = aim.copy()
                    aim.rotate(Quaternion(lateral, pitch))
                aim_bone(arm, bone, neck_p + aim)
            else:
                aim_bone(arm, bone, v(fr[key]))
        orient_face(arm, "mixamorig:Head", fr.get("gaze"),
                    float(fr.get("gaze_amount", 0.0) or 0.0),
                    float(spec.get("gaze_max_offset_deg", 75.0)),
                    -v(fr["basis_y"]))
        apply_fingers(arm, fist)

        if rest >= 0.65:
            hips = arm.pose.bones["mixamorig:Hips"]
            hips.location = (loc_x, 0.0, loc_z)
            bpy.context.view_layer.update()
        elif plant == "none":
            # Airborne: no plant, no snapping. Hip height integrates the
            # estimator's real pelvis arc from the last applied height,
            # optionally amplified by the window's "boost".
            hips = arm.pose.bones["mixamorig:Hips"]
            hips.location = (loc_x, prev_hip_y + (ph - prev_ph) * 100.0 * boost_by_frame.get(f, 1.0), loc_z)
            bpy.context.view_layer.update()
        elif plant in ("left", "right"):
            foot = "mixamorig:LeftFoot" if plant == "left" else "mixamorig:RightFoot"
            plant_hips(arm, hint, loc_x, loc_z, foot)
            flatten_foot(arm, "Left" if plant == "left" else "Right")
        else:  # both
            la = world_loc(arm, "mixamorig:LeftFoot").z
            ra = world_loc(arm, "mixamorig:RightFoot").z
            lower = "mixamorig:LeftFoot" if la <= ra else "mixamorig:RightFoot"
            other = "Right" if la <= ra else "Left"
            plant_hips(arm, hint, loc_x, loc_z, lower)
            flatten_foot(arm, "Left" if lower.endswith("LeftFoot") else "Right")
            flatten_foot(arm, other)
        prev_hip_y = float(arm.pose.bones["mixamorig:Hips"].location[1])
        prev_ph = ph
        key_pose(arm, f)


def run(spec_path: str) -> dict:
    spec = json.loads(rpath(spec_path).read_text(encoding="utf-8"))
    payload = json.loads(rpath(spec["joints_out"]).read_text(encoding="utf-8"))
    frames = payload["frames"]
    end = len(frames)
    src_fps = float(payload.get("src_fps", 24))
    dst_fps = float(payload.get("dst_fps", 30))

    use_profile(spec.get("rig_profile"))
    arm = get_armature(spec)
    bpy.context.view_layer.objects.active = arm
    if bpy.context.mode != "POSE":
        bpy.ops.object.mode_set(mode="POSE")
    action = ensure_action(arm, spec["action_name"], end)

    paused = _pause_deform(arm)
    try:
        _key_frames(arm, spec, frames, src_fps, dst_fps)
    finally:
        for m in paused:
            m.show_viewport = True
        bpy.context.view_layer.update()

    set_linear(action)
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()

    qa_dest = [max(1, min(end, int(round((sf - 1) * dst_fps / src_fps + 1)))) for sf in spec.get("qa_src_frames", [])]
    samples = [measure(arm, f) for f in qa_dest]
    clip_dir = rpath(spec["clip_dir"])
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "retarget_samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    result = {
        "action": spec["action_name"],
        "frames": end,
        "samples": samples,
        "constraints": [c.name for pb in arm.pose.bones for c in pb.constraints],
    }
    (clip_dir / "apply_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_stills_render(spec_path: str, dest_frames) -> dict:
    """Front + side stills via a temporary camera + Workbench render.

    Window-independent (viewport screenshots capture black when the
    Blender window is occluded or minimized). The camera is removed
    afterward so the scene stays Armature + meshes only.
    """
    import math

    spec = json.loads(rpath(spec_path).read_text(encoding="utf-8"))
    pose_dir = rpath(spec["clip_dir"]) / "poses"
    pose_dir.mkdir(parents=True, exist_ok=True)
    arm = get_armature(spec)
    bind_action(arm, spec)
    scene = bpy.context.scene

    cam_data = bpy.data.cameras.new("QA_Camera")
    cam = bpy.data.objects.new("QA_Camera", cam_data)
    scene.collection.objects.link(cam)
    old_cam = scene.camera
    old_engine = scene.render.engine
    old_res = (scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage)
    old_path = scene.render.filepath
    scene.camera = cam
    scene.render.engine = "BLENDER_WORKBENCH"
    old_shading = (scene.display.shading.light, scene.display.shading.color_type)
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.render.resolution_x, scene.render.resolution_y = 960, 720
    scene.render.resolution_percentage = 100

    written = []
    try:
        for f in dest_frames:
            scene.frame_set(int(f))
            bpy.context.view_layer.update()
            hips = (arm.matrix_world @ arm.pose.bones["mixamorig:Hips"].matrix).to_translation()
            views = {
                "front": (Vector((hips.x, hips.y - 3.9, max(0.95, hips.z))), (math.radians(90), 0.0, 0.0)),
                "side": (Vector((hips.x + 3.9, hips.y, max(0.95, hips.z))), (math.radians(90), 0.0, math.radians(90))),
            }
            for vn, (loc, rot) in views.items():
                cam.location = loc
                cam.rotation_euler = rot
                dest = pose_dir / f"{vn}_{int(f):04d}.png"
                scene.render.filepath = str(dest)
                bpy.ops.render.render(write_still=True)
                written.append(dest.name)
    finally:
        scene.camera = old_cam
        scene.render.engine = old_engine
        scene.display.shading.light, scene.display.shading.color_type = old_shading
        scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = old_res
        scene.render.filepath = old_path
        bpy.data.objects.remove(cam)
        bpy.data.cameras.remove(cam_data)
    return {"stills": written}


def dump_curves(spec_path: str) -> dict:
    """Every bone, every frame: pose channels + world location."""
    spec = json.loads(rpath(spec_path).read_text(encoding="utf-8"))
    arm = get_armature(spec)
    act = bind_action(arm, spec)
    # The clip's own length: scene.frame_end belongs to whichever clip was
    # applied last, which in a two-character scene may be the other one.
    end = int(round(act.frame_range[1]))
    out = []
    for f in range(1, end + 1):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        bones = {}
        for pb in arm.pose.bones:
            wl = (arm.matrix_world @ pb.matrix).to_translation()
            bones[pb.name] = {
                "location": [round(v, 6) for v in pb.location],
                "rotation_quaternion": [round(v, 6) for v in pb.rotation_quaternion],
                "world_location": [round(wl.x, 6), round(wl.y, 6), round(wl.z, 6)],
            }
        # Gaze audit data: the head's FACE axis and the body's forward, in
        # world. Positions alone cannot reveal where a character looks.
        face = ((arm.matrix_world @ arm.pose.bones["mixamorig:Head"].matrix).to_3x3()
                @ Vector((0.0, 0.0, 1.0))).normalized()
        fwd = ((arm.matrix_world @ arm.pose.bones["mixamorig:Hips"].matrix).to_3x3()
               @ Vector((0.0, 0.0, 1.0))).normalized()
        out.append({"frame": f, "bones": bones,
                    "face_dir": [round(face.x, 5), round(face.y, 5), round(face.z, 5)],
                    "body_forward": [round(fwd.x, 5), round(fwd.y, 5), round(fwd.z, 5)]})
    clip_dir = rpath(spec["clip_dir"])
    (clip_dir / "curves.json").write_text(json.dumps({"frames": out}), encoding="utf-8")
    return {"curves_frames": len(out)}

def pair_mesh_contact(spec_a: str, spec_b: str, f0: int = 1, f1: int = 0, step: int = 1) -> dict:
    """Real mesh-vs-mesh contact between two characters, per frame.

    The capsule proxies in compare_pair.py (a 0.16 m torso cylinder, a
    0.11 m head sphere) are a cheap stand-in that runs outside Blender.
    They are also OPTIMISTIC: a hood, a shoulder pad or a wide chest all
    live outside the cylinder, so a limb can clear every capsule and
    still visibly pass through the character. This is the ground truth —
    the evaluated (posed, skinned) meshes, tested for actual
    intersection.

    Returns per frame: the number of intersecting face pairs, and the
    minimum surface-to-surface distance when they do not intersect.
    """
    import bmesh
    from mathutils.bvhtree import BVHTree

    spec_a = json.loads(rpath(spec_a).read_text(encoding="utf-8"))
    spec_b = json.loads(rpath(spec_b).read_text(encoding="utf-8"))
    scene = bpy.context.scene
    f1 = f1 or scene.frame_end
    deps = bpy.context.evaluated_depsgraph_get()

    def meshes_of(spec):
        arm = get_armature(spec)
        bind_action(arm, spec)     # measure THIS pair's clips, not whatever is bound
        return [o for o in bpy.data.objects
                if o.type == "MESH" and o.find_armature() is arm and not o.hide_render]

    mesh_a, mesh_b = meshes_of(spec_a), meshes_of(spec_b)
    if not mesh_a or not mesh_b:
        raise RuntimeError(f"need meshes for both characters (found {len(mesh_a)} / {len(mesh_b)})")

    def bvh_of(objs):
        bm = bmesh.new()
        for o in objs:
            ev = o.evaluated_get(deps)
            me = ev.to_mesh()
            # to_mesh() gives object-local coordinates and every mesh has
            # its OWN world matrix: bring each one to world before merging
            # (transforming the merged result by the first mesh's matrix
            # misplaces any mesh that is not parented identically).
            me.transform(ev.matrix_world)
            bm.from_mesh(me)
            ev.to_mesh_clear()
        tree = BVHTree.FromBMesh(bm)
        verts = [v.co.copy() for v in bm.verts]
        bm.free()
        return tree, verts

    out = []
    for f in range(f0, f1 + 1, step):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        deps = bpy.context.evaluated_depsgraph_get()
        tree_a, verts_a = bvh_of(mesh_a)
        tree_b, _ = bvh_of(mesh_b)
        pairs = tree_a.overlap(tree_b)
        best = 1e9
        if not pairs:
            for v in verts_a[::7]:      # sampled: we only need the minimum
                hit = tree_b.find_nearest(v)
                if hit and hit[3] is not None:
                    best = min(best, float(hit[3]))
        out.append({"frame": f, "overlaps": len(pairs),
                    "min_dist": round(best, 4) if best < 1e8 else None})
    clip_dir = rpath(spec_a["clip_dir"]).parent
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "pair_contact.json").write_text(json.dumps({"frames": out}), encoding="utf-8")
    hits = [r for r in out if r["overlaps"]]
    return {"frames": len(out), "intersecting_frames": len(hits),
            "worst": max((r["overlaps"] for r in out), default=0),
            "sample": hits[:12] or out[:6]}
