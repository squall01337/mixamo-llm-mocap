"""Two-character check: does the retargeted pair stand, reach and miss
each other the way the two performers did?

  tools\\GVHMR\\.venv\\Scripts\\python.exe pipeline\\compare_pair.py ^
      --spec action_specs\\duel_ybot.json --spec action_specs\\duel_ninja.json
  ... --from-src 80 --to-src 160     limit to a beat
  ... --detail                       per-frame table

`qa_clip.py` proves one clip is not broken. `compare_reference.py`
proves one clip looks like its performer. Neither can see the thing a
two-fighter scene is entirely about: the RELATIONSHIP between the two
characters. A pair of individually perfect retargets still fails if the
attacker stands a metre too far away (every punch swings through empty
air) or a metre too close (his fist ends up inside the other's skull).

Three measurements, in the order they matter:

  separation      hip-to-hip distance, reference vs retarget. The
                  reference figure is exact — both performers were
                  estimated in the same camera frame, so `incam_root`
                  puts them on one floor with no assumptions.
  reach           gap from the striking hand/foot to the other
                  character's body (spine axis and head). A punch that
                  stopped 5 cm from the guard in the video should stop
                  ~5 cm from it here.
  intrusion       absolute defect check: an extremity INSIDE the other
                  character's torso or head. No reference needed — it
                  is wrong however the video looked. Limbs and bodies are
                  capsules with each character's OWN radii, measured on
                  its mesh by setup_rig.py (profile `capsules`); a profile
                  without them falls back to thin limbs inside human-sized
                  torso and head, which is optimistic (PITFALLS #33).

Both performers are read in the estimator's CAMERA frame (`incam` in
landmarks.json), the one frame two independently-estimated people
actually share. Their `world` landmarks cannot be used here: those are
normalised per performer so frame 0's facing becomes forward, so two
performers' limbs sit in two differently-yawed frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

TOL = {
    "separation": 0.10,   # metres, hip to hip
    "reach": 0.09,        # metres, striking extremity to the other body
}
MIN_RUN = 3
TORSO_R = 0.16           # metres from the spine axis
HEAD_R = 0.11            # metres from the head joint

STRIKERS = {
    "l_hand": "mixamorig:LeftHand", "r_hand": "mixamorig:RightHand",
    "l_foot": "mixamorig:LeftToeBase", "r_foot": "mixamorig:RightToeBase",
}

# Whole limb segments, for the intrusion test.
LIMBS = {
    "l_forearm": ("mixamorig:LeftForeArm", "mixamorig:LeftHand"),
    "r_forearm": ("mixamorig:RightForeArm", "mixamorig:RightHand"),
    "l_upperarm": ("mixamorig:LeftArm", "mixamorig:LeftForeArm"),
    "r_upperarm": ("mixamorig:RightArm", "mixamorig:RightForeArm"),
    "l_shin": ("mixamorig:LeftLeg", "mixamorig:LeftFoot"),
    "r_shin": ("mixamorig:RightLeg", "mixamorig:RightFoot"),
    "l_foot": ("mixamorig:LeftFoot", "mixamorig:LeftToe_End"),     # ankle to toe tip
    "r_foot": ("mixamorig:RightFoot", "mixamorig:RightToe_End"),
    "l_thigh": ("mixamorig:LeftUpLeg", "mixamorig:LeftLeg"),
    "r_thigh": ("mixamorig:RightUpLeg", "mixamorig:RightLeg"),
    # the hand itself: wrist to whichever reaches further, the middle
    # fingertip (open hand) or the middle knuckle (a fist curls the tip back)
    "l_hand": ("mixamorig:LeftHand", ("mixamorig:LeftHandMiddle4", "mixamorig:LeftHandMiddle1")),
    "r_hand": ("mixamorig:RightHand", ("mixamorig:RightHandMiddle4", "mixamorig:RightHandMiddle1")),
}


def limb_points(f, df, b0, b1):
    """World end points of one LIMBS segment. An end given as several bones
    is the one farthest from the start (bones the rig lacks are skipped)."""
    p0 = f.bone(df, b0)
    best = None
    for name in ((b1,) if isinstance(b1, str) else b1):
        try:
            p = f.bone(df, name)
        except KeyError:
            continue
        if best is None or np.linalg.norm(p - p0) > np.linalg.norm(best - p0):
            best = p
    if best is None:
        raise KeyError(b1)
    return p0, best


def limb_gap(p0, p1, r_limb, hips, neck, head, r_torso, r_head) -> float:
    """Signed clearance between a limb capsule and a body (torso capsule
    from hips to the shoulder line, head sphere): negative = inside."""
    return min(seg_seg(p0, p1, hips, neck) - r_torso - r_limb, seg_dist(head, p0, p1) - r_head - r_limb)
# Paired deliberately: Blender reports a bone's HEAD, so `mixamorig:*Hand`
# is at the wrist — matching the performer's wrist landmark, not their
# fingertips (docs/PITFALLS.md #21).
REF_STRIKERS = {
    "l_hand": "left_wrist", "r_hand": "right_wrist",
    "l_foot": "left_foot_index", "r_foot": "right_foot_index",
}


def rpath(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO / p)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)


def seg_dist(p, a, b) -> float:
    """Distance from point p to segment a-b."""
    ab = b - a
    L = float(np.dot(ab, ab))
    t = 0.0 if L < 1e-12 else float(np.clip(np.dot(p - a, ab) / L, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + ab * t)))


def seg_seg(p1, q1, p2, q2, samples: int = 12) -> float:
    """Distance between two segments (sampled — exact enough at cm scale).

    Limb-vs-limb, not point-vs-limb: a kick that passes THROUGH a head
    can easily keep both its toe and its knee outside the skull while the
    shin goes straight through the middle. Checking only the endpoints of
    a limb is how that ships.
    """
    ts = np.linspace(0.0, 1.0, samples)
    best = 1e9
    for t in ts:
        pt = p1 + (q1 - p1) * t
        best = min(best, seg_dist(pt, p2, q2))
    return float(best)


class Fighter:
    """One character: its retargeted curves and its performer's landmarks."""

    def __init__(self, spec_path: Path):
        self.spec = json.loads(rpath(spec_path).read_text(encoding="utf-8"))
        self.name = self.spec["name"]
        self.curves = json.loads(
            (rpath(self.spec["clip_dir"]) / "curves.json").read_text(encoding="utf-8"))["frames"]
        self.lm = json.loads(rpath(self.spec["landmarks"]).read_text(encoding="utf-8"))["frames"]
        prof = json.loads(rpath(self.spec.get("rig_profile", "rig_profile.json")).read_text(encoding="utf-8"))
        L = prof["lengths"]
        # Mesh-fitted capsule radii (setup_rig.py), when the profile has them.
        self.caps = {k: float(v["radius"]) for k, v in prof.get("capsules", {}).items()}
        self.arm = L["l_arm"] + L["l_fore"]
        self.leg = L["l_upleg"] + L["l_leg"]
        self.src_fps = float(self.spec.get("src_fps", 24))
        self.dst_fps = float(self.spec.get("dst_fps", 30))
        ref_leg = float(np.mean([
            np.linalg.norm(self.ref(i, "left_hip") - self.ref(i, "left_knee"))
            + np.linalg.norm(self.ref(i, "left_knee") - self.ref(i, "left_ankle"))
            for i in range(0, len(self.lm), 5)]))
        self.scale = self.leg / max(ref_leg, 1e-6)

    def dest(self, sf: int) -> int:
        return int(round((sf - 1) * self.dst_fps / self.src_fps + 1))

    def cap(self, part: str, default: float = 0.0) -> float:
        return self.caps.get(part, default)

    def ref(self, i: int, n: str) -> np.ndarray:
        p = self.lm[i]["world"][n]
        return np.array([p["x"], p["y"], p["z"]])

    def cam_pt(self, i: int, n: str) -> np.ndarray:
        """A performer landmark in raw camera space (y down)."""
        q = self.lm[i]["incam"][n]
        return np.array([q["x"], q["y"], q["z"]])

    def gravity_in_camera(self, every: int = 4) -> np.ndarray:
        """Which way is UP, expressed in camera space.

        Solved, not assumed. Each performer is delivered in two frames:
        `world` (gravity-aligned — that is what makes foot planting work
        at all) and `incam` (the camera's). Fitting the rotation between
        them for one performer and pushing the world frame's up axis
        through it says where gravity points in the camera's frame.
        """
        Wm, Cm = [], []
        for i in range(0, len(self.lm), every):
            hw = 0.5 * (self.ref(i, "left_hip") + self.ref(i, "right_hip"))
            hc = 0.5 * (self.cam_pt(i, "left_hip") + self.cam_pt(i, "right_hip"))
            for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip",
                      "left_knee", "right_knee", "left_ankle", "right_ankle",
                      "left_elbow", "right_elbow", "left_wrist", "right_wrist", "nose"):
                Wm.append(self.ref(i, n) - hw)
                Cm.append(self.cam_pt(i, n) - hc)
        Wm, Cm = np.array(Wm), np.array(Cm)
        U, _, Vt = np.linalg.svd(Cm.T @ Wm)          # camera ~= R @ world
        d = np.sign(np.linalg.det(U @ Vt))
        R = U @ np.diag([1.0, 1.0, d]) @ Vt
        return R @ np.array([0.0, -1.0, 0.0])        # world y is DOWN


    def ref_pt(self, i: int, n: str) -> np.ndarray:
        """A performer landmark in the shared, gravity-aligned stage frame."""
        return self.basis @ self.cam_pt(i, n)

    def bone(self, f: int, b: str) -> np.ndarray:
        return np.array(self.curves[f - 1]["bones"][b]["world_location"])


def stage_basis(up_cam: np.ndarray) -> np.ndarray:
    """A gravity-aligned frame shared by both performers.

    z = true up, x = the camera's right flattened into the ground plane
    (which is the fighting line in a side-on plate), y = depth.
    """
    ez = up_cam / np.linalg.norm(up_cam)
    ex = np.array([1.0, 0.0, 0.0])
    ex = ex - ez * float(np.dot(ex, ez))
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    return np.stack([ex, ey, ez])


def body_gap(pt, hips, neck, head) -> float:
    """Signed clearance from a point to a body: negative = inside it.

    `head` must be the SKULL CENTRE on both sides of the comparison. The
    obvious pair — the performer's nose and the character's Head bone —
    are not the same landmark (one is on the face, the other sits at the
    base of the skull) and pairing them bakes a fixed ~10 cm error into
    every strike that is aimed at a head.
    """
    return min(seg_dist(pt, hips, neck) - TORSO_R, float(np.linalg.norm(pt - head)) - HEAD_R)


def ref_body(f, i, S):
    """A performer's torso axis and skull centre, in character metres."""
    hips = 0.5 * (f.ref_pt(i, "left_hip") + f.ref_pt(i, "right_hip")) * S
    neck = 0.5 * (f.ref_pt(i, "left_shoulder") + f.ref_pt(i, "right_shoulder")) * S
    head = 0.5 * (f.ref_pt(i, "left_ear") + f.ref_pt(i, "right_ear")) * S
    return hips, neck, head


def app_body(f, df):
    """A character's torso axis and skull centre.

    The axis tops out at the SHOULDER LINE, not at `mixamorig:Neck`.
    Neck sits above the shoulder joints, so a hips-to-neck capsule is
    5-10 cm longer than the hips-to-shoulder-mid capsule the reference
    side can build from landmarks — and that surplus length reaches out
    to meet an incoming limb, manufacturing an intrusion that is really
    just two different definitions of "torso".
    """
    hips = f.bone(df, "mixamorig:Hips")
    neck = 0.5 * (f.bone(df, "mixamorig:LeftArm") + f.bone(df, "mixamorig:RightArm"))
    try:
        head = 0.5 * (f.bone(df, "mixamorig:Head") + f.bone(df, "mixamorig:HeadTop_End"))
    except KeyError:
        head = f.bone(df, "mixamorig:Head")
    return hips, neck, head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", action="append", required=True,
                    help="action_spec per character (give it twice)")
    ap.add_argument("--from-src", type=int, default=None)
    ap.add_argument("--to-src", type=int, default=None)
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--step", type=int, default=1)
    args = ap.parse_args()
    if len(args.spec) != 2:
        raise SystemExit("compare_pair needs exactly two --spec arguments")

    A, B = Fighter(args.spec[0]), Fighter(args.spec[1])
    # One frame for both, with a real vertical. Measuring height in raw
    # camera space costs whatever the camera's pitch is: on this plate it
    # inflated the reference kick's peak by 0.14 m, which is most of a
    # "the foot is too low" finding that does not exist.
    up = 0.5 * (A.gravity_in_camera() + B.gravity_in_camera())
    basis = stage_basis(up)
    A.basis = B.basis = basis
    tilt = np.degrees(np.arccos(np.clip(np.dot(up / np.linalg.norm(up), [0, -1, 0]), -1, 1)))
    S = 0.5 * (A.scale + B.scale)   # one stage, one scale
    print(f"camera tilt vs gravity: {tilt:.1f} deg")
    n_src = min(len(A.lm), len(B.lm))
    print(f"pair: {A.name} vs {B.name} | stage scale {S:.4f} "
          f"({A.name} {A.scale:.4f}, {B.name} {B.scale:.4f})")
    for f in (A, B):
        if f.caps:
            print(f"capsules {f.name}: measured on its mesh — torso {f.cap('torso', TORSO_R):.3f}, "
                  f"head {f.cap('head', HEAD_R):.3f}, forearm {f.cap('l_forearm'):.3f}, "
                  f"hand {f.cap('l_hand'):.3f}, shin {f.cap('l_shin'):.3f} m")
        else:
            print(f"capsules {f.name}: DEFAULTS (torso {TORSO_R}, head {HEAD_R}, limbs 0) — optimistic; "
                  f"`python pipeline\\run_in_blender.py skeleton <spec>` measures them on the mesh")

    s0 = args.from_src or 1
    s1 = min(args.to_src or n_src, n_src)
    rows = []
    for sf in range(s0, s1 + 1, args.step):
        fa, fb = A.dest(sf), B.dest(sf)
        if fa > len(A.curves) or fb > len(B.curves):
            continue
        i = sf - 1

        ref_sep = float(np.linalg.norm(
            0.5 * (A.ref_pt(i, "left_hip") + A.ref_pt(i, "right_hip"))
            - 0.5 * (B.ref_pt(i, "left_hip") + B.ref_pt(i, "right_hip")))) * S
        app_sep = float(np.linalg.norm(A.bone(fa, "mixamorig:Hips") - B.bone(fb, "mixamorig:Hips")))

        row = {"src": sf, "ref_sep": ref_sep, "app_sep": app_sep, "strikes": {}, "limbs": {}}
        for att, dfn, other, dfo, side in ((A, fa, B, fb, "A"), (B, fb, A, fa, "B")):
            o_hips, o_neck, o_head = app_body(other, dfo)
            for lname, (b0, b1) in LIMBS.items():
                try:
                    p0, p1 = limb_points(att, dfn, b0, b1)
                except KeyError:
                    continue
                row["limbs"][f"{side}:{lname}"] = limb_gap(
                    p0, p1, att.cap(lname), o_hips, o_neck, o_head,
                    other.cap("torso", TORSO_R), other.cap("head", HEAD_R))
        for tag, bone_name in STRIKERS.items():
            rn = REF_STRIKERS[tag]
            for att, dfn, other, dfo, side in ((A, fa, B, fb, "A"), (B, fb, A, fa, "B")):
                gap_a = body_gap(att.bone(dfn, bone_name), *app_body(other, dfo))
                gap_r = body_gap(att.ref_pt(i, rn) * S, *ref_body(other, i, S))
                row["strikes"][f"{side}:{tag}"] = (gap_r, gap_a)
        rows.append(row)

    if not rows:
        raise SystemExit("no overlapping frames")

    if args.detail:
        print("\n  src | separation ref->app | closest strike gap ref->app (limb)")
        for r in rows:
            k, (gr, ga) = min(r["strikes"].items(), key=lambda kv: kv[1][1])
            print(f"  {r['src']:3d} | {r['ref_sep']:.2f} -> {r['app_sep']:.2f}"
                  f" | {gr:+.3f} -> {ga:+.3f}  {k}")
        print()

    print("\nFINDINGS")
    found = False

    def windows(vals, tol):
        bad = np.abs(np.asarray(vals)) > tol
        i = 0
        while i < len(bad):
            if not bad[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(bad) and bad[j + 1]:
                j += 1
            if rows[j]["src"] - rows[i]["src"] >= MIN_RUN:
                yield i, j
            i = j + 1

    # A spec may DECLARE a clearance offset (two characters are bulkier
    # than two humans; see docs/PITFALLS.md #33). Report those windows as
    # declared rather than as an unexplained defect — but still report
    # them, because the number is the departure from the video.
    declared = []
    for f in (A, B):
        for ro in f.spec.get("root_offset", []):
            declared.append((ro["src"][0], ro["src"][1],
                             float(ro.get("dx", 0.0)), float(ro.get("dy", 0.0))))

    def is_declared(s0, s1):
        return any(a - 5 <= s0 and s1 <= b + 5 for a, b, _, _ in declared)

    sep_err = [r["app_sep"] - r["ref_sep"] for r in rows]
    for i, j in windows(sep_err, TOL["separation"]):
        seg = np.array(sep_err[i:j + 1])
        k = int(np.argmax(np.abs(seg)))
        found = True
        tag = " [declared root_offset]" if is_declared(rows[i]["src"], rows[j]["src"]) else ""
        print(f"  src {rows[i]['src']:3d}-{rows[j]['src']:3d}  separation: mean {seg.mean():+.3f} m, "
              f"worst {seg[k]:+.3f} @ src {rows[i + k]['src']} "
              f"({'too far apart' if seg.mean() > 0 else 'too close'}){tag}")

    for key in sorted(rows[0]["strikes"]):
        gaps_r = [r["strikes"][key][0] for r in rows]
        gaps_a = [r["strikes"][key][1] for r in rows]
        # Only judge a limb where the reference actually brings it near
        # the other body — a guard hand resting by the chest is not a
        # strike and its "gap" carries no information.
        err = [(a - r) if (r < 0.35 or a < 0.35) else 0.0 for r, a in zip(gaps_r, gaps_a)]
        for i, j in windows(err, TOL["reach"]):
            seg = np.array(err[i:j + 1])
            k = int(np.argmax(np.abs(seg)))
            found = True
            print(f"  src {rows[i]['src']:3d}-{rows[j]['src']:3d}  {key} reach: mean {seg.mean():+.3f} m, "
                  f"worst {seg[k]:+.3f} @ src {rows[i + k]['src']} "
                  f"({'stops short' if seg.mean() > 0 else 'reaches too far'})"
                  f" | ref gap {gaps_r[i + k]:+.3f} -> {gaps_a[i + k]:+.3f}")

    # Absolute defect: a limb inside the other character. No reference
    # needed and no minimum run — two frames of a shin through a skull is
    # still two frames of a shin through a skull.
    for key in sorted(rows[0].get("limbs", {})):
        pen = np.array([r["limbs"].get(key, 9.0) for r in rows])
        bad = pen < 0.0
        i = 0
        while i < len(bad):
            if not bad[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(bad) and bad[j + 1]:
                j += 1
            k = i + int(np.argmin(pen[i:j + 1]))
            found = True
            print(f"  src {rows[i]['src']:3d}-{rows[j]['src']:3d}  {key} INSIDE the other character: "
                  f"worst {pen[k]:.3f} m past the surface @ src {rows[k]['src']}")
            i = j + 1

    if not found:
        print("  none — separation, reach and clearance all track the reference")

    print("\nCLOSEST APPROACH  (the beats where the two actually meet)")
    for key in sorted(rows[0]["strikes"]):
        gr = np.array([r["strikes"][key][0] for r in rows])
        ga = np.array([r["strikes"][key][1] for r in rows])
        if gr.min() > 0.30 and ga.min() > 0.30:
            continue     # this limb never came near the other fighter
        kr, ka = int(np.argmin(gr)), int(np.argmin(ga))
        flag = "" if abs(ga.min() - gr.min()) < 0.10 else "   <-- differs"
        print(f"  {key:10s} reference {gr.min():+.3f} m @ src {rows[kr]['src']:3d}"
              f" | retarget {ga.min():+.3f} m @ src {rows[ka]['src']:3d}{flag}")

    sep_r = np.array([r["ref_sep"] for r in rows])
    sep_a = np.array([r["app_sep"] for r in rows])
    print(f"\nseparation: reference {sep_r.min():.2f}-{sep_r.max():.2f} m, "
          f"retarget {sep_a.min():.2f}-{sep_a.max():.2f} m, "
          f"mean error {np.mean(sep_a - sep_r):+.3f} m")


if __name__ == "__main__":
    main()
