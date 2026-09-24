"""Build the clean rest scene from your own Mixamo character download —
ANY Mixamo character with the standard mixamorig skeleton (Y Bot,
X Bot, or any character from the Mixamo library).

Mixamo characters cannot be redistributed (Adobe terms), so this repo
ships no .fbx/.blend. Download a character yourself:

  1. Log in at https://www.mixamo.com , open the Characters tab,
     pick any character (Y Bot is the reference this pipeline was
     developed on).
  2. Download: Format "FBX Binary", Pose "T-pose", with skin.
  3. Save as  <repo>\\ybot.fbx  (any name; pass it via --fbx)

Then build the scene (Blender 5.1+):

  "C:\\Program Files\\Blender Foundation\\Blender 5.1\\blender.exe" ^
      --background --python pipeline\\setup_rig.py -- ^
      --fbx ybot.fbx --out ybot_rest.blend

Validates the import against the conventions the pipeline relies on
(docs/RIG.md): armature "Armature" rotated X=90deg scale 0.01,
mixamorig: bones rooted at mixamorig:Hips — then saves the .blend the
FK apply runs in AND writes `rig_profile.json` at the repo root: the
character's measured rest-joint positions, bone lengths, hip height
and ground height, and its rest skeleton. The lift/apply/QA read that
profile, which is what makes the pipeline work with any Mixamo
character's proportions (without a profile they fall back to built-in
Y Bot measurements); fk_solve.py uses the skeleton to solve clips
without Blender.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Profile joint -> bone whose (posed-rest) head is that joint.
PROFILE_JOINTS = {
    "hips": "mixamorig:Hips", "spine": "mixamorig:Spine",
    "spine1": "mixamorig:Spine1", "spine2": "mixamorig:Spine2",
    "neck": "mixamorig:Neck", "head": "mixamorig:Head",
    "l_shoulder": "mixamorig:LeftShoulder", "l_arm": "mixamorig:LeftArm",
    "l_elbow": "mixamorig:LeftForeArm", "l_wrist": "mixamorig:LeftHand",
    "l_hand": "mixamorig:LeftHandMiddle1",
    "r_shoulder": "mixamorig:RightShoulder", "r_arm": "mixamorig:RightArm",
    "r_elbow": "mixamorig:RightForeArm", "r_wrist": "mixamorig:RightHand",
    "r_hand": "mixamorig:RightHandMiddle1",
    "l_upleg": "mixamorig:LeftUpLeg", "l_knee": "mixamorig:LeftLeg",
    "l_ankle": "mixamorig:LeftFoot", "l_foot": "mixamorig:LeftToeBase",
    "l_toe": "mixamorig:LeftToe_End",
    "r_upleg": "mixamorig:RightUpLeg", "r_knee": "mixamorig:RightLeg",
    "r_ankle": "mixamorig:RightFoot", "r_foot": "mixamorig:RightToeBase",
    "r_toe": "mixamorig:RightToe_End",
}

# LEN key (as the lift uses them) -> bone whose length it is.
PROFILE_LENGTHS = {
    "spine1": "mixamorig:Spine1", "spine2": "mixamorig:Spine2",
    "neck": "mixamorig:Neck", "head": "mixamorig:Head",
    "l_shoulder": "mixamorig:LeftShoulder", "l_arm": "mixamorig:LeftArm",
    "l_fore": "mixamorig:LeftForeArm", "l_hand": "mixamorig:LeftHand",
    "r_shoulder": "mixamorig:RightShoulder", "r_arm": "mixamorig:RightArm",
    "r_fore": "mixamorig:RightForeArm", "r_hand": "mixamorig:RightHand",
    "l_upleg": "mixamorig:LeftUpLeg", "l_leg": "mixamorig:LeftLeg",
    "l_foot": "mixamorig:LeftFoot",
    "r_upleg": "mixamorig:RightUpLeg", "r_leg": "mixamorig:RightLeg",
    "r_foot": "mixamorig:RightFoot",
}


# Collision capsules measured on the character's own mesh (compare_pair.py).
# part -> (segment start, segment end, bones whose skin belongs to it). The
# torso axis runs from the hips to the MID-SHOULDER line and the head is a
# sphere at the skull centre — the same proxies compare_pair builds, so the
# radii mean what the comparison assumes (docs/PITFALLS.md #29).
CAPSULE_PARTS = {
    "torso": ("mixamorig:Hips", "shoulder_mid", ("Hips", "Spine", "Spine1", "Spine2")),
    "head": ("skull", "skull", ("Head", "HeadTop_End")),
}
for _s, _S in (("l", "Left"), ("r", "Right")):
    CAPSULE_PARTS.update({
        f"{_s}_upperarm": (f"mixamorig:{_S}Arm", f"mixamorig:{_S}ForeArm", (f"{_S}Arm",)),
        f"{_s}_forearm": (f"mixamorig:{_S}ForeArm", f"mixamorig:{_S}Hand", (f"{_S}ForeArm",)),
        f"{_s}_hand": (f"mixamorig:{_S}Hand", f"mixamorig:{_S}HandMiddle1", (f"{_S}Hand",)),
        f"{_s}_thigh": (f"mixamorig:{_S}UpLeg", f"mixamorig:{_S}Leg", (f"{_S}UpLeg",)),
        f"{_s}_shin": (f"mixamorig:{_S}Leg", f"mixamorig:{_S}Foot", (f"{_S}Leg",)),
        f"{_s}_foot": (f"mixamorig:{_S}Foot", f"mixamorig:{_S}Toe_End", (f"{_S}Foot", f"{_S}ToeBase")),
    })


def measure_capsules(arm) -> dict:
    """Radius of each body part, measured on the skinned mesh at rest.

    compare_pair.py's contact proxies used fixed human-sized radii (torso
    0.16 m, head 0.11 m, limbs zero), which scored a kick as clearing by 4 mm
    while the meshes intersected (docs/PITFALLS.md #33). Here every mesh
    vertex belongs to the bone with its largest skin weight, and each part's
    radius is the 95th percentile of its vertices' distance to the part's
    segment (the head: to the skull centre). Fingers count as hand.
    """
    import numpy as np

    W = arm.matrix_world

    def head(n):
        # REST positions (the mesh's own coordinates are the bind pose), so
        # this is right whatever pose or action the armature holds.
        b = arm.data.bones.get(n)
        return None if b is None else np.array(W @ b.head_local)

    pts = {}
    for part, (a, b, _) in CAPSULE_PARTS.items():
        if part == "torso":
            la, ra = head("mixamorig:LeftArm"), head("mixamorig:RightArm")
            pa, pb_ = head(a), (None if la is None or ra is None else 0.5 * (la + ra))
        elif part == "head":
            h, t = head("mixamorig:Head"), head("mixamorig:HeadTop_End")
            pa = pb_ = None if h is None else (h if t is None else 0.5 * (h + t))
        else:
            pa, pb_ = head(a), head(b)
            if pb_ is None and pa is not None and part.endswith("_hand"):
                fore = head(a.replace("Hand", "ForeArm"))
                if fore is not None:
                    pb_ = pa + (pa - fore) / max(np.linalg.norm(pa - fore), 1e-6) * 0.10
        if pa is not None and pb_ is not None:
            pts[part] = (pa, pb_)
    owner = {}
    for part, (_, _, bones) in CAPSULE_PARTS.items():
        for bn in bones:
            owner["mixamorig:" + bn] = part
    for side in ("Left", "Right"):
        for b in arm.pose.bones:
            if b.name.startswith(f"mixamorig:{side}Hand") and b.name != f"mixamorig:{side}Hand":
                owner[b.name] = f"{side[0].lower()}_hand"

    dists = {part: [] for part in pts}
    for o in bpy.data.objects:
        if o.type != "MESH" or o.find_armature() is not arm:
            continue
        names = {g.index: g.name for g in o.vertex_groups}
        mw = o.matrix_world
        for v in o.data.vertices:
            if not v.groups:
                continue
            g = max(v.groups, key=lambda e: e.weight)
            part = owner.get(names.get(g.group, ""))
            if part not in pts:
                continue
            p = np.array(mw @ v.co)
            a, b = pts[part]
            ab = b - a
            t = 0.0 if float(ab @ ab) < 1e-12 else float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
            dists[part].append(float(np.linalg.norm(p - (a + ab * t))))
    return {part: {"radius": round(float(np.percentile(d, 95)), 4), "max": round(float(max(d)), 4),
                   "vertices": len(d)}
            for part, d in dists.items() if len(d) >= 8}


def dump_rig_profile(arm, out_path: Path) -> dict:
    """Measure the character's rest geometry into rig_profile.json."""
    bpy.context.view_layer.update()
    scale = arm.scale[0]

    def w(bone_name):
        pb = arm.pose.bones.get(bone_name)
        if pb is None:
            return None
        p = (arm.matrix_world @ pb.matrix).to_translation()
        return [round(p.x, 5), round(p.y, 5), round(p.z, 5)]

    rest = {}
    for joint, bone in PROFILE_JOINTS.items():
        p = w(bone)
        if p is None and joint in ("l_hand", "r_hand"):
            # No finger bones on this character: extrapolate the hand
            # tip along the forearm direction.
            wrist = rest.get(joint.replace("hand", "wrist"))
            elbow = rest.get(joint.replace("hand", "elbow"))
            if wrist and elbow:
                d = [wrist[i] - elbow[i] for i in range(3)]
                n = max(1e-6, sum(x * x for x in d) ** 0.5)
                hand_len = arm.pose.bones[bone.replace("HandMiddle1", "Hand")].length * scale
                p = [round(wrist[i] + d[i] / n * hand_len, 5) for i in range(3)]
        if p is None:
            raise SystemExit(f"cannot measure profile joint {joint} ({bone})")
        rest[joint] = p

    lengths = {}
    for key, bone in PROFILE_LENGTHS.items():
        pb = arm.pose.bones.get(bone)
        if pb is None:
            raise SystemExit(f"cannot measure bone length {key} ({bone})")
        lengths[key] = round(pb.length * scale, 5)

    import fk_solve

    profile = {
        "character": [o.name for o in bpy.data.objects if o.type == "MESH"],
        "hip_height": rest["hips"][2],
        "ground_z": round(0.5 * (rest["l_ankle"][2] + rest["r_ankle"][2]), 5),
        "rest": rest,
        "lengths": lengths,
        # The rest skeleton (hierarchy + armature-space rest matrices), so
        # fk_solve.py can solve this character's clips outside Blender.
        "skeleton": fk_solve.Skeleton.from_blender(arm).to_profile(),
        # This character's own thickness, for compare_pair.py's contact test.
        "capsules": measure_capsules(arm),
    }
    out_path.write_text(json.dumps(profile, indent=1), encoding="utf-8")
    return profile


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbx", default="ybot.fbx")
    ap.add_argument("--out", default="ybot_rest.blend")
    args = ap.parse_args(argv)

    fbx = Path(args.fbx)
    fbx = fbx if fbx.is_absolute() else REPO / fbx
    out = Path(args.out)
    out = out if out.is_absolute() else REPO / out
    if not fbx.exists():
        raise SystemExit(f"ybot.fbx not found at {fbx} — download it from mixamo.com (see module docstring)")

    # Empty scene, metric, 30 fps.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 30
    scene.frame_start = 1

    bpy.ops.import_scene.fbx(filepath=str(fbx))

    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if len(arms) != 1:
        raise SystemExit(f"expected exactly one armature in the FBX, found {len(arms)}")
    arm = arms[0]
    arm.name = "Armature"

    # Validate the conventions the whole pipeline assumes.
    errors = []
    warnings = []
    if abs(arm.rotation_euler[0] - math.pi / 2) > 1e-3:
        errors.append(f"armature X rotation is {math.degrees(arm.rotation_euler[0]):.1f} deg, expected 90")
    if abs(arm.scale[0] - 0.01) > 1e-5:
        errors.append(f"armature scale is {arm.scale[0]}, expected 0.01 (cm -> m)")
    bones = arm.pose.bones
    if "mixamorig:Hips" not in bones:
        errors.append("root bone mixamorig:Hips missing (do not strip the mixamorig: prefix)")
    bad_names = [b.name for b in bones if not b.name.startswith("mixamorig:")]
    if bad_names:
        errors.append(f"non-mixamorig bones: {bad_names[:5]}")
    if len(bones) != 65:
        warnings.append(f"{len(bones)} pose bones (standard characters have 65 — "
                        "missing fingers are tolerated, fists are skipped for absent bones)")
    mesh_names = sorted(o.name for o in bpy.data.objects if o.type == "MESH")
    if not mesh_names:
        errors.append("no mesh objects in the FBX")

    bpy.context.view_layer.update()
    if "mixamorig:Hips" in bones:
        hip_z = (arm.matrix_world @ bones["mixamorig:Hips"].matrix).to_translation().z
        if not (0.6 < hip_z < 1.4):
            errors.append(f"rest hip world height {hip_z:.3f} m — not a human-scale Mixamo import")
        else:
            print(f"rest hip world height {hip_z:.4f} m")

    if errors:
        raise SystemExit("FBX does not match the expected Mixamo conventions:\n  " + "\n  ".join(errors))
    for wmsg in warnings:
        print("WARNING:", wmsg)

    # No cameras/lights/extras: scene must be the armature + its meshes.
    for o in list(bpy.data.objects):
        if o.type not in ("ARMATURE", "MESH"):
            bpy.data.objects.remove(o)

    profile = dump_rig_profile(arm, REPO / "rig_profile.json")
    print(f"rig_profile.json written — hip {profile['hip_height']} m, "
          f"ground {profile['ground_z']} m, meshes {profile['character']}")

    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    print(f"saved {out} — {len(bones)} bones, meshes {mesh_names}, 30 fps")


if __name__ == "__main__":
    main()
