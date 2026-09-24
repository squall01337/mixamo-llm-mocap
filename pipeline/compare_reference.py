"""Frame-by-frame comparison of a retargeted clip against its source
plate — the refinement stage after the first pass.

Usage:
  tools\\GVHMR\\.venv\\Scripts\\python.exe pipeline\\compare_reference.py --spec action_specs\\<motion>.json
  ... --from-src 130 --to-src 210        limit to a beat
  ... --detail                            per-frame table instead of windows

Why this exists: QA (`qa_clip.py`) proves a clip is not broken — no
explosions, no skate, no pops. It cannot tell you the clip does not
*look* like the video. This does, by measuring both on the quantities
an eye actually reads and reporting where they diverge:

  hand height vs the head      "hands at face level instead of chest"
  hand separation              "fists collide" / "fists too far apart"
  hand distance from chest     "arm not tucked in"
  limb inside the torso        "his arm glitches behind his back"
  gaze yaw / elevation         "looking at the sky / off to the side"

The reference is scaled to the character by arm length, and every
quantity is chosen to be proportion-robust — except hand-vs-head, which
is reported alongside the character's own head/shoulder proportion
offset so pose error and proportion error are not confused.

Findings are grouped into contiguous source-frame windows, ready to be
turned into `arm_pose` / `reach` / `arm_overrides` spec entries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Flag a window when the mean |delta| over it exceeds these (metres, degrees).
TOL = {
    "hand_height": 0.06,
    "separation": 0.06,
    "chest_distance": 0.07,
    "torso_clearance": 0.005,
    "gaze_yaw": 20.0,
    "gaze_elev": 15.0,
}
MIN_RUN = 3          # source frames; shorter excursions are noise
TORSO_R = 0.16       # metres from the spine axis; closer than this is inside the body


def rpath(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO / p)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--from-src", type=int, default=None)
    ap.add_argument("--to-src", type=int, default=None)
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--step", type=int, default=1)
    args = ap.parse_args()

    spec = json.loads(rpath(args.spec).read_text(encoding="utf-8"))
    lm = json.loads(rpath(spec["landmarks"]).read_text(encoding="utf-8"))["frames"]
    curves = json.loads((rpath(spec["clip_dir"]) / "curves.json").read_text(encoding="utf-8"))["frames"]
    prof_path = spec.get("rig_profile", "rig_profile.json")
    profile = json.loads(rpath(prof_path).read_text(encoding="utf-8"))
    src_fps = float(spec.get("src_fps", 24))
    dst_fps = float(spec.get("dst_fps", 30))

    L = profile["lengths"]
    char_arm = L["l_arm"] + L["l_fore"]

    def ref(i, n):
        p = lm[i]["world"][n]
        return np.array([p["x"], p["y"], p["z"]])

    ref_arm = float(np.mean([
        np.linalg.norm(ref(i, "left_shoulder") - ref(i, "left_elbow"))
        + np.linalg.norm(ref(i, "left_elbow") - ref(i, "left_wrist"))
        for i in range(0, len(lm), 5)]))
    S = char_arm / max(ref_arm, 1e-6)

    def bone(f, b):
        return np.array(curves[f - 1]["bones"][b]["world_location"])

    # Character vs performer proportion (head above shoulders), so a
    # hand-height delta can be split into pose and proportion.
    ref_head = float(np.mean([
        -(ref(i, "nose")[1] - 0.5 * (ref(i, "left_shoulder")[1] + ref(i, "right_shoulder")[1]))
        for i in range(min(60, len(lm)))])) * S
    def face_z(f):
        """Height of the character's FACE. The Head bone sits at the base
        of the skull; the performer's reference point is the nose, so
        comparing against the joint bakes in a systematic offset."""
        head = bone(f, "mixamorig:Head")[2]
        try:
            return 0.5 * (head + bone(f, "mixamorig:HeadTop_End")[2])
        except KeyError:
            return head

    app_head = float(np.mean([
        face_z(f) - 0.5 * (bone(f, "mixamorig:LeftArm")[2] + bone(f, "mixamorig:RightArm")[2])
        for f in range(1, min(40, len(curves)) + 1)]))
    proportion = app_head - ref_head

    def metrics_ref(i):
        ls, rs = ref(i, "left_shoulder"), ref(i, "right_shoulder")
        lh, rh = ref(i, "left_hip"), ref(i, "right_hip")
        shm = 0.5 * (ls + rs)
        up = unit(shm - 0.5 * (lh + rh))                  # y is DOWN in this frame
        left = unit((ls - rs) - up * np.dot(ls - rs, up))
        fwd = unit(np.cross(up, left))
        out = {"sep": np.linalg.norm(ref(i, "left_wrist") - ref(i, "right_wrist")) * S}
        for side, tag in (("left", "l"), ("right", "r")):
            v = ref(i, f"{side}_wrist") - shm
            e = ref(i, f"{side}_elbow") - shm
            out[f"h_{tag}"] = -(ref(i, f"{side}_wrist")[1] - ref(i, "nose")[1]) * S
            out[f"d_{tag}"] = float(np.linalg.norm(np.array([np.dot(v, left), np.dot(v, fwd)]))) * S
            out[f"b_{tag}"] = float(np.dot(v, fwd)) * S    # + = in front of the chest
        g = np.array(lm[i].get("gaze", [0, 0, 0]), dtype=float)
        out["gyaw"] = float(np.degrees(np.arctan2(g[0], -g[2]))) if np.any(g) else 0.0
        out["gelev"] = float(np.degrees(np.arcsin(np.clip(-g[1], -1, 1)))) if np.any(g) else 0.0
        return out

    def metrics_app(f):
        la, ra = bone(f, "mixamorig:LeftArm"), bone(f, "mixamorig:RightArm")
        shm = 0.5 * (la + ra)
        # The reference frame is built from the SHOULDERS; so is this one
        # (chest_forward). With a twisting spine the pelvis faces elsewhere.
        fwd = np.array(curves[f - 1].get("chest_forward", curves[f - 1].get("body_forward", [0, -1, 0])),
                       dtype=float)
        fwd[2] = 0.0
        fwd = unit(fwd)
        left = np.array([-fwd[1], fwd[0], 0.0])
        head_z = face_z(f)
        out = {"sep": float(np.linalg.norm(bone(f, "mixamorig:LeftHand") - bone(f, "mixamorig:RightHand")))}
        for side, tag in (("Left", "l"), ("Right", "r")):
            wrist = bone(f, f"mixamorig:{side}Hand")
            elbow = bone(f, f"mixamorig:{side}ForeArm")
            v = wrist - shm
            out[f"h_{tag}"] = float(wrist[2] - head_z)
            out[f"d_{tag}"] = float(np.linalg.norm(np.array([np.dot(v, left), np.dot(v, fwd)])))
            out[f"b_{tag}"] = float(np.dot(v, fwd))
            # Self-intersection: how far the wrist/elbow sit from the spine
            # axis. This is absolute — a limb inside the torso is a defect
            # whatever the reference does — and it is what actually catches
            # "the arm glitches behind his back".
            hips, neck = bone(f, "mixamorig:Hips"), bone(f, "mixamorig:Neck")
            axis = unit(neck - hips)
            for tag2, pt in ((f"clear_{tag}", wrist), (f"clear_e{tag}", elbow)):
                d = pt - hips
                out[tag2] = float(np.linalg.norm(d - axis * np.dot(d, axis)))
        face = np.array(curves[f - 1].get("face_dir", [0, -1, 0]), dtype=float)
        out["gyaw"] = float(np.degrees(np.arctan2(face[0], -face[1])))
        out["gelev"] = float(np.degrees(np.arcsin(np.clip(face[2], -1, 1))))
        return out

    s0 = args.from_src or 1
    s1 = args.to_src or len(lm)
    rows = []
    for sf in range(s0, min(s1, len(lm)) + 1, args.step):
        df = int(round((sf - 1) * dst_fps / src_fps + 1))
        if df < 1 or df > len(curves):
            continue
        R, A = metrics_ref(sf - 1), metrics_app(df)
        rows.append({"src": sf, "dest": df, "R": R, "A": A})

    print(f"reference scaled by {S:.3f} (arm {ref_arm:.3f} -> {char_arm:.3f} m)")
    print(f"character head sits {proportion:+.3f} m relative to the performer's, shoulder-referenced "
          f"— subtract this from hand-height deltas before calling them pose error\n")

    if args.detail:
        print("  src/dest | hand height L/R (ref->app) | separation | behind-chest L/R | gaze yaw")
        for r in rows:
            R, A = r["R"], r["A"]
            print(f"  {r['src']:3d}/{r['dest']:3d} | {R['h_l']:+.2f}->{A['h_l']:+.2f} {R['h_r']:+.2f}->{A['h_r']:+.2f}"
                  f" | {R['sep']:.2f}->{A['sep']:.2f} | {R['b_l']:+.2f}->{A['b_l']:+.2f} {R['b_r']:+.2f}->{A['b_r']:+.2f}"
                  f" | {R['gyaw']:+6.1f}->{A['gyaw']:+6.1f}")
        print()

    # --- findings: contiguous windows where a metric is out of tolerance ---
    checks = [
        # Target: the hand sits as far below the character's head as the
        # performer's does below theirs — that is what the eye compares.
        # (The `proportion` figure printed above explains WHY matching
        # shoulder-relative geometry is not enough; it is not added here.)
        ("hands too high vs the head", "hand_height", lambda R, A: (R["h_l"] - A["h_l"] + R["h_r"] - A["h_r"]) / 2),
        ("hand separation", "separation", lambda R, A: R["sep"] - A["sep"]),
        ("left hand distance from chest", "chest_distance", lambda R, A: R["d_l"] - A["d_l"]),
        ("right hand distance from chest", "chest_distance", lambda R, A: R["d_r"] - A["d_r"]),
        ("LEFT hand inside the torso", "torso_clearance", lambda R, A: min(0.0, A["clear_l"] - TORSO_R)),
        ("RIGHT hand inside the torso", "torso_clearance", lambda R, A: min(0.0, A["clear_r"] - TORSO_R)),
        ("LEFT elbow inside the torso", "torso_clearance", lambda R, A: min(0.0, A["clear_el"] - TORSO_R)),
        ("RIGHT elbow inside the torso", "torso_clearance", lambda R, A: min(0.0, A["clear_er"] - TORSO_R)),
        ("gaze yaw", "gaze_yaw", lambda R, A: (R["gyaw"] - A["gyaw"] + 180) % 360 - 180),
        ("gaze elevation", "gaze_elev", lambda R, A: R["gelev"] - A["gelev"]),
    ]

    print("FINDINGS  (windows where the retarget diverges from the reference)")
    any_found = False
    for label, tol_key, fn in checks:
        tol = TOL[tol_key]
        vals = np.array([fn(r["R"], r["A"]) for r in rows])
        bad = np.abs(vals) > tol
        i = 0
        while i < len(bad):
            if not bad[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(bad) and bad[j + 1]:
                j += 1
            if (rows[j]["src"] - rows[i]["src"]) >= MIN_RUN:
                seg = vals[i:j + 1]
                k = int(np.argmax(np.abs(seg)))
                any_found = True
                print(f"  src {rows[i]['src']:3d}-{rows[j]['src']:3d} (dest {rows[i]['dest']:3d}-{rows[j]['dest']:3d})"
                      f"  {label}: mean {seg.mean():+.3f}, worst {seg[k]:+.3f} @ src {rows[i + k]['src']}")
            i = j + 1
    if not any_found:
        print("  none — every tracked quantity is within tolerance of the reference")


if __name__ == "__main__":
    main()
