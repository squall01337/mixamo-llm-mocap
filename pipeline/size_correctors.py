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
# frames: the lift maps source frames with the LANDMARKS' fps, and so must
# anything that sizes a window for it.

def fps_pair(spec: dict) -> tuple[float, float]:
    jp = rpath(spec["joints_out"])
    if jp.exists():
        j = json.loads(jp.read_text(encoding="utf-8"))
        return float(j["src_fps"]), float(j["dst_fps"])
    return float(load(spec["landmarks"])["fps"]), float(spec.get("dst_fps", 30))


def check_window(a: int, b: int, n_src: int) -> None:
    if not (1 <= a <= b <= n_src):
        raise SystemExit(f"window src {a}-{b} is outside the clip (source frames 1-{n_src})")


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
               / reach for f in range(a_dest, min(b_dest, len(joints)) + 1))


def reach_entries_without(entries: list, a: int, b: int, sides: tuple) -> list:
    """The existing `reach` list minus what this sizing replaces: entries
    overlapping the window, for the sides being sized. A "both" entry keeps
    its other side as a one-sided copy."""
    out = []
    for e in entries:
        if not overlaps(e, a, b):
            out.append(e)
            continue
        es = ("left", "right") if e.get("side", "both") == "both" else (e["side"],)
        for keep in es:
            if keep not in sides:
                out.append(dict(e, side=keep))
    return out


def size_reach(spec: dict, a: int, b: int, side: str, ramp: int, max_fraction: float, min_extension: float):
    landmarks = load(spec["landmarks"])
    check_window(a, b, len(landmarks["frames"]))
    sides = ("left", "right") if side == "both" else (side,)
    kept = reach_entries_without(spec.get("reach", []), a, b, sides)
    base = copy.deepcopy(spec)
    base["reach"] = copy.deepcopy(kept)
    src_fps, dst_fps = float(landmarks["fps"]), float(spec.get("dst_fps", 30))   # the lift's own mapping
    a_d, b_d = int(round((a - 1) * dst_fps / src_fps + 1)), int(round((b - 1) * dst_fps / src_fps + 1))
    new = []
    for sd in sides:
        s = sd[0]
        target = min(performer_peak(landmarks, sd, a, b), max_fraction)
        cache = {}

        def peak(factor):
            factor = round(factor, 4)
            if factor not in cache:
                sp = copy.deepcopy(base)
                if factor > 1.0:
                    sp["reach"].append({"src": [a, b], "ramp_src": ramp, "side": sd, "factor": factor,
                                        "max_fraction": max_fraction, "min_extension": min_extension})
                payload, _ = lm.lift(sp)
                cache[factor] = retarget_peak(payload["frames"], s, a_d, b_d)
            return cache[factor]

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
        new.append({"src": [a, b], "ramp_src": ramp, "side": sd, "factor": factor,
                    "max_fraction": max_fraction, "min_extension": min_extension,
                    "comment": f"sized by size_correctors.py: performer peak {target:.0%}, retarget was {p0:.0%}"})
    replaced = len(spec.get("reach", [])) - len([e for e in spec.get("reach", []) if not overlaps(e, a, b)])
    print(f"\nthe spec's COMPLETE `reach` list with this window re-sized"
          f"{f' ({replaced} overlapping entr' + ('y' if replaced == 1 else 'ies') + ' replaced)' if replaced else ''}:")
    print('"reach": ' + json.dumps(kept + new, indent=2))


# ---------------------------------------------------------------------------
# clearance

class Char:
    """One character's solved curves as compare_pair sees them."""

    def __init__(self, spec: dict):
        self.spec = spec
        cpath = rpath(spec["clip_dir"]) / "curves.json"
        if not cpath.exists():
            raise SystemExit(f"{cpath} missing — run fk_solve.py (or the `curves` stage) for {spec['name']} first")
        self.frames = json.loads(cpath.read_text(encoding="utf-8"))["frames"]
        prof = load(spec.get("rig_profile", "rig_profile.json"))
        self.caps = {k: float(v["radius"]) for k, v in prof.get("capsules", {}).items()}
        self.shift = np.zeros((len(self.frames), 3))   # per-frame horizontal shift applied on read

    def bone(self, f, n):
        return np.array(self.frames[f - 1]["bones"][n]["world_location"]) + self.shift[f - 1]


def pair_gap(A: Char, B: Char, f: int) -> tuple[float, str]:
    """compare_pair's intrusion test at one frame: the smallest capsule gap
    between either character's limbs and the other's torso and head."""
    best, who = 1e9, ""
    for att, oth, tag in ((A, B, "A"), (B, A, "B")):
        hips, neck, head = cp.app_body(oth, f)
        r_t, r_h = oth.caps.get("torso", cp.TORSO_R), oth.caps.get("head", cp.HEAD_R)
        for name, (b0, b1) in cp.LIMBS.items():
            try:
                p0, p1 = cp.limb_points(att, f, b0, b1)
            except KeyError:
                continue
            g = cp.limb_gap(p0, p1, att.caps.get(name, 0.0), hips, neck, head, r_t, r_h)
            if g < best:
                best, who = g, f"{tag}:{name}"
    return best, who


def size_clearance(spec_a: dict, spec_b: dict, a: int, b: int, ramp: int, margin: float):
    A, B = Char(spec_a), Char(spec_b)
    src_fps, dst_fps = fps_pair(spec_a)
    check_window(a, b, int(round((len(A.frames) - 1) * src_fps / dst_fps + 1)))

    def s2d(sf):
        return (sf - 1) * dst_fps / src_fps + 1

    n = min(len(A.frames), len(B.frames))
    # The curves already carry A's existing root_offset entries. Take out the
    # ones this sizing replaces (a stage offset is a pure translation of the
    # solve), so the result is the WHOLE offset for the window, not an extra.
    replaced = [ro for ro in spec_a.get("root_offset", []) if overlaps(ro, a, b)]
    kept = [ro for ro in spec_a.get("root_offset", []) if not overlaps(ro, a, b)]
    base = np.zeros((len(A.frames), 3))
    for ro in replaced:
        r = ro.get("ramp_src", 8)
        for f in range(1, len(A.frames) + 1):
            amt = lm.window_amount(f, [ro["src"][0], ro["src"][0] + r], [ro["src"][1] - r, ro["src"][1]], s2d)
            base[f - 1] -= np.array([float(ro.get("dx", 0.0)), float(ro.get("dy", 0.0)), 0.0]) * amt

    frames = list(range(max(1, int(round(s2d(a)))), min(n, int(round(s2d(b)))) + 1))
    amt = {f: lm.window_amount(f, [a, a + ramp], [b - ramp, b], s2d) for f in frames}
    mid = frames[len(frames) // 2]
    A.shift = base
    away = float(np.sign(A.bone(mid, "mixamorig:Hips")[0] - B.bone(mid, "mixamorig:Hips")[0]) or 1.0)

    def at(m):
        A.shift = base.copy()
        for f in frames:
            A.shift[f - 1, 0] += away * m * amt[f]
        return min((pair_gap(A, B, f) + (f,) for f in frames), key=lambda t: t[0])

    g0, who0, f0 = at(0.0)
    note = f" (its {len(replaced)} overlapping root_offset entr{'y' if len(replaced) == 1 else 'ies'} removed)" if replaced else ""
    print(f"{spec_a['name']} vs {spec_b['name']}, src {a}-{b}{note}: closest {g0:+.3f} m ({who0} @ dest {f0}); "
          f"capsules: {'measured' if A.caps else 'DEFAULT'} / {'measured' if B.caps else 'DEFAULT'}")
    if g0 >= margin:
        if replaced:
            # The capsules are a screen, not the truth: an offset the mesh
            # contact pass asked for must not be dropped on their say-so.
            print(f"clear by {g0:.3f} m (margin {margin}) even without the existing offset, by the capsules — "
                  "keep it if the mesh contact pass (run_in_blender.py contact) is what required it")
            return
        print(f"clear by {g0:.3f} m (margin {margin}) without any offset in this window")
        m = None
    else:
        # An offset cannot clear a contact while it is still ramping in: say
        # where, instead of sizing an absurd offset to compensate.
        A.shift = base.copy()
        ramping = [f for f in frames if amt[f] < 0.999 and pair_gap(A, B, f)[0] < margin]
        if ramping:
            f = ramping[0]
            print(f"contact at dest {f} (src {(f - 1) * src_fps / dst_fps + 1:.0f}) while the offset is only "
                  f"{amt[f]:.0%} in: move the window so its ramp finishes before the contact "
                  f"(contact frames src {(ramping[0] - 1) * src_fps / dst_fps + 1:.0f}-"
                  f"{(ramping[-1] - 1) * src_fps / dst_fps + 1:.0f} are inside a ramp)")
            return
        if at(0.8)[0] < margin:
            print("no offset up to 0.80 m clears it — a contact the stage cannot buy away (an intended hit?)")
            return
        lo, hi = 0.0, 0.8                          # clearance grows as the pair moves apart
        for _ in range(10):
            mid_m = 0.5 * (lo + hi)
            if at(mid_m)[0] >= margin:
                hi = mid_m
            else:
                lo = mid_m
        m = float(np.ceil(hi / 0.005) * 0.005)
        g1, who1, f1 = at(m)
        print(f"root_offset dx {away * m:+.3f} m on {spec_a['name']} -> closest {g1:+.3f} m ({who1} @ dest {f1})")
    entries = list(kept)
    if m is not None:
        entries.append({"src": [a, b], "ramp_src": ramp, "dx": round(away * m, 3),
                        "comment": f"sized by size_correctors.py: clearance {g0:+.3f} -> {g1:+.3f} m with mesh capsules; "
                                   "ramp through frames where this character already travels (PITFALLS #37)"})
    print(f"\nthe spec's COMPLETE `root_offset` list for {spec_a['name']}"
          f"{' (overlapping entries replaced)' if replaced else ''}:")
    print('"root_offset": ' + json.dumps(entries, indent=2))
    if m is not None:
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
