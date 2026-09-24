"""Size a spec corrector from the numbers instead of by hand.

Correctors were sized by passes: guess a factor, lift, apply, compare,
adjust — "16 passes became 1-2" by discipline, not by construction. With
the lift callable in-process and the FK solve Blender-free, the sizing can
be computed. Two sizers, each printing the spec entry to paste:

  reach      the `reach` factor that brings the retarget's peak arm
             extension in a window to the performer's (the estimator and the
             prefilter compress strike extension, docs/PITFALLS.md #18, #32):

    python pipeline\\size_correctors.py reach --spec action_specs\\spin.json --src 172 204

  clearance  the smallest `root_offset` that keeps two characters' limbs out
             of each other's torso and head over a window, using both
             characters' mesh-fitted capsules (docs/PIPELINE.md 9.6):

    python pipeline\\size_correctors.py clearance --spec action_specs\\duel_ybot.json ^
        --with action_specs\\duel_ninja.json --src 116 185 --ramp 23

It sizes; it does not decide. Whether a window needs the correction, and
where a stage offset may ramp without skating (docs/PITFALLS.md #37), stays
the operator's call, and the mesh contact pass (`run_in_blender.py contact`)
stays the ground truth for clearance.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_pair as cp  # noqa: E402
import lift_to_mixamo as lm  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def rpath(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO / p)


def load(p) -> dict:
    return json.loads(rpath(p).read_text(encoding="utf-8"))


def overlaps(entry, a, b) -> bool:
    return not (entry["src"][1] < a or entry["src"][0] > b)


# ---------------------------------------------------------------------------
# reach

def performer_peak(landmarks: dict, side: str, a: int, b: int) -> float:
    """Peak wrist-to-shoulder distance over the arm's own length, src a..b."""
    F = landmarks["frames"]

    def p(i, n):
        w = F[i]["world"][n]
        return np.array([w["x"], w["y"], w["z"]])

    arm = np.median([np.linalg.norm(p(i, f"{side}_elbow") - p(i, f"{side}_shoulder"))
                     + np.linalg.norm(p(i, f"{side}_wrist") - p(i, f"{side}_elbow")) for i in range(len(F))])
    return max(float(np.linalg.norm(p(i, f"{side}_wrist") - p(i, f"{side}_shoulder"))) / arm
               for i in range(a - 1, min(b, len(F))))


def retarget_peak(joints: list, s: str, a_dest: int, b_dest: int) -> float:
    reach = lm.LEN[f"{s}_arm"] + lm.LEN[f"{s}_fore"]
    return max(float(np.linalg.norm(np.asarray(joints[f - 1][f"{s}_wrist"]) - np.asarray(joints[f - 1][f"{s}_arm"])))
               / reach for f in range(a_dest, b_dest + 1))


def size_reach(spec: dict, a: int, b: int, side: str, ramp: int, max_fraction: float, min_extension: float):
    landmarks = load(spec["landmarks"])
    base = copy.deepcopy(spec)
    base["reach"] = [r for r in base.get("reach", []) if not overlaps(r, a, b)]
    src_fps, dst_fps = float(base.get("src_fps", 24)), float(base.get("dst_fps", 30))
    a_d, b_d = int(round((a - 1) * dst_fps / src_fps + 1)), int(round((b - 1) * dst_fps / src_fps + 1))
    entries = []
    for sd in (("left", "right") if side == "both" else (side,)):
        s = sd[0]
        target = min(performer_peak(landmarks, sd, a, b), max_fraction)

        def peak(factor):
            sp = copy.deepcopy(base)
            if factor > 1.0:
                sp["reach"].append({"src": [a, b], "ramp_src": ramp, "side": sd, "factor": factor,
                                    "max_fraction": max_fraction, "min_extension": min_extension})
            joints, _ = lm.lift(sp)
            return retarget_peak(joints["frames"], s, a_d, b_d)

        p0 = peak(1.0)
        print(f"{sd:5s} arm, src {a}-{b}: performer peaks at {target:.1%} of arm length, retarget at {p0:.1%}")
        if p0 >= target - 0.005:
            print("       no reach needed")
            continue
        lo, hi = 1.0, 1.8
        if peak(hi) < target - 0.005:
            print(f"       even factor {hi} stops at {peak(hi):.1%} (max_fraction {max_fraction}) — not a reach problem")
            continue
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            if peak(mid) < target:
                lo = mid
            else:
                hi = mid
        factor = round(hi, 3)
        print(f"       factor {factor} -> {peak(factor):.1%}")
        entries.append({"src": [a, b], "ramp_src": ramp, "side": sd, "factor": factor,
                        "max_fraction": max_fraction, "min_extension": min_extension,
                        "comment": f"sized by size_correctors.py: performer peak {target:.0%}, retarget was {p0:.0%}"})
    if entries:
        print('\n"reach": ' + json.dumps(entries, indent=2))


# ---------------------------------------------------------------------------
# clearance

class Char:
    def __init__(self, spec: dict):
        self.spec = spec
        cpath = rpath(spec["clip_dir"]) / "curves.json"
        if not cpath.exists():
            raise SystemExit(f"{cpath} missing — run fk_solve.py (or the `curves` stage) for {spec['name']} first")
        self.frames = json.loads(cpath.read_text(encoding="utf-8"))["frames"]
        prof = load(spec.get("rig_profile", "rig_profile.json"))
        self.caps = {k: float(v["radius"]) for k, v in prof.get("capsules", {}).items()}

    def bone(self, f, n):
        return np.array(self.frames[f - 1]["bones"][n]["world_location"])

    def body(self, f, shift):
        hips = self.bone(f, "mixamorig:Hips") + shift
        neck = 0.5 * (self.bone(f, "mixamorig:LeftArm") + self.bone(f, "mixamorig:RightArm")) + shift
        try:
            head = 0.5 * (self.bone(f, "mixamorig:Head") + self.bone(f, "mixamorig:HeadTop_End")) + shift
        except KeyError:
            head = self.bone(f, "mixamorig:Head") + shift
        return hips, neck, head

    def limbs(self, f, shift):
        for name, (b0, b1) in cp.LIMBS.items():
            try:
                yield name, self.bone(f, b0) + shift, self.bone(f, b1) + shift
            except KeyError:
                continue


def clearance(A: Char, B: Char, fa: int, fb: int, shift_a) -> tuple[float, str]:
    """Smallest capsule gap between the two characters at one frame (negative
    = inside), and which limb it is — compare_pair's intrusion test."""
    best, who = 1e9, ""
    for att, fn, sh, oth, fo, so, tag in ((A, fa, shift_a, B, fb, np.zeros(3), "A"),
                                         (B, fb, np.zeros(3), A, fa, shift_a, "B")):
        hips, neck, head = oth.body(fo, so)
        r_t, r_h = oth.caps.get("torso", cp.TORSO_R), oth.caps.get("head", cp.HEAD_R)
        for name, p0, p1 in att.limbs(fn, sh):
            r = att.caps.get(name, 0.0)
            g = min(cp.seg_seg(p0, p1, hips, neck) - r_t - r, cp.seg_dist(head, p0, p1) - r_h - r)
            if g < best:
                best, who = g, f"{tag}:{name}"
    return best, who


def size_clearance(spec_a: dict, spec_b: dict, a: int, b: int, ramp: int, margin: float):
    A, B = Char(spec_a), Char(spec_b)
    src_fps, dst_fps = float(spec_a.get("src_fps", 24)), float(spec_a.get("dst_fps", 30))

    def s2d(sf):
        return (sf - 1) * dst_fps / src_fps + 1

    frames = range(max(1, int(round(s2d(a)))), min(len(A.frames), len(B.frames), int(round(s2d(b)))) + 1)
    amt = {f: lm.window_amount(f, [a, a + ramp], [b - ramp, b], s2d) for f in frames}
    mid = frames[len(frames) // 2]
    away = np.sign(A.bone(mid, "mixamorig:Hips")[0] - B.bone(mid, "mixamorig:Hips")[0]) or 1.0

    def worst(m):
        return min((clearance(A, B, f, f, np.array([away * m * amt[f], 0.0, 0.0])) + (f,) for f in frames),
                   key=lambda t: t[0])

    g0, who0, f0 = worst(0.0)
    print(f"{spec_a['name']} vs {spec_b['name']}, src {a}-{b}: closest {g0:+.3f} m ({who0} @ dest {f0}); "
          f"capsules: {'measured' if A.caps else 'DEFAULT'} / {'measured' if B.caps else 'DEFAULT'}")
    if g0 >= margin:
        print(f"already clear by {g0:.3f} m (margin {margin}) — no offset needed")
        return
    # An offset cannot clear a contact while it is still ramping in: say
    # where, instead of sizing an absurd offset to compensate.
    ramping = [f for f in frames if amt[f] < 0.999 and clearance(A, B, f, f, np.zeros(3))[0] < margin]
    if ramping:
        f = ramping[0]
        print(f"contact at dest {f} (src {(f - 1) * src_fps / dst_fps + 1:.0f}) while the offset is only "
              f"{amt[f]:.0%} in: move the window so its ramp finishes before the contact "
              f"(contact frames src {(ramping[0] - 1) * src_fps / dst_fps + 1:.0f}-"
              f"{(ramping[-1] - 1) * src_fps / dst_fps + 1:.0f} are inside a ramp)")
        return
    m = None
    for step in np.arange(0.005, 0.805, 0.005):
        if worst(step)[0] >= margin:
            m = float(step)
            break
    if m is None:
        print("no offset up to 0.80 m clears it — a contact the stage cannot buy away (an intended hit?)")
        return
    g1, who1, f1 = worst(m)
    dx = round(float(away * m), 3)
    print(f"root_offset dx {dx:+.3f} m on {spec_a['name']} -> closest {g1:+.3f} m ({who1} @ dest {f1})")
    entry = {"src": [a, b], "ramp_src": ramp, "dx": dx,
             "comment": f"sized by size_correctors.py: clearance {g0:+.3f} -> {g1:+.3f} m with mesh capsules; "
                        "ramp through frames where this character already travels (PITFALLS #37)"}
    print('\n"root_offset": ' + json.dumps([entry], indent=2))
    print("\nverify on the meshes: python pipeline\\run_in_blender.py contact <spec> --with <other>")


def main() -> None:
    ap = argparse.ArgumentParser(description="Size spec correctors from the numbers")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reach")
    r.add_argument("--spec", required=True)
    r.add_argument("--src", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    r.add_argument("--side", default="both", choices=["both", "left", "right"])
    r.add_argument("--ramp", type=int, default=4)
    r.add_argument("--max-fraction", type=float, default=0.97)
    r.add_argument("--min-extension", type=float, default=0.30)
    c = sub.add_parser("clearance")
    c.add_argument("--spec", required=True, help="the character whose stage position moves")
    c.add_argument("--with", dest="other", required=True)
    c.add_argument("--src", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    c.add_argument("--ramp", type=int, default=12)
    c.add_argument("--margin", type=float, default=0.01)
    args = ap.parse_args()
    if args.cmd == "reach":
        size_reach(load(args.spec), *args.src, args.side, args.ramp, args.max_fraction, args.min_extension)
    else:
        size_clearance(load(args.spec), load(args.other), *args.src, args.ramp, args.margin)


if __name__ == "__main__":
    main()
