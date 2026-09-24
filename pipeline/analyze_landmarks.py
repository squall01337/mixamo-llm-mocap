"""Numeric beat detection over a landmarks.json — the tool that writes
the beat sheet's evidence column. Run this BEFORE writing an action_spec
and take every limb decision from these numbers, never from
screen-reading stills (facing the camera, viewer-left is the
character's RIGHT — every screen-read limb guess in this project's
history was wrong at least once).

Usage:
  python pipeline\\analyze_landmarks.py --landmarks plates\\<plate>\\landmarks.json [--step 6]

Prints a per-N-frame table plus detected events:
  - pelvis baseline / dips (ducks, crouches) / peaks (jumps)
  - windows where both feet are airborne (true flight)
  - single-leg-up windows (kicks, knees) with the acting side
  - punch windows per arm (wrist depth toward camera, z < -0.32)
  - arm extension windows in ANY direction (side-on plates, hooks)
  - T-pose spans (wrist span > 1.22 m) — bind candidates
  - shoulder-line yaw (facing wobble vs punch rotation)
  - low 2D-confidence windows per limb: where the 3D limb is a guess
  - a DRAFT `plant` schedule, ready to paste into the spec

True joint heights need `pelvis_height` in the landmarks (the GVHMR
estimator writes it); without it, heights fall back to hip-relative.
The draft schedule uses GVHMR's own per-foot static confidence when the
landmarks carry it (`static`, current estimator), ankle heights otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

PLANT_DOWN = 0.15    # ankle height (m) under which a foot can be the support
PLANT_UP = 0.25      # both ankles above this = airborne
STATIC_ON = 0.5      # GVHMR static probability for "this foot is planted"
MIN_RUN = 3          # source frames; shorter support flickers are merged
EXTENDED = 0.90      # wrist-to-shoulder distance / arm length = a strike
LOW_CONF = 0.5       # ViTPose confidence under which a keypoint is doubtful

LIMBS = {
    "left arm": ("left_shoulder", "left_elbow", "left_wrist"),
    "right arm": ("right_shoulder", "right_elbow", "right_wrist"),
    "left leg": ("left_hip", "left_knee", "left_ankle"),
    "right leg": ("right_hip", "right_knee", "right_ankle"),
    "head": ("nose", "left_ear", "right_ear"),
}


def draft_plant(la_h, ra_h, st_l=None, st_r=None) -> list:
    """Support schedule from the numbers: [(first, last, support)], 1-based.

    A foot is planted when its ankle is near the floor and, if the
    estimator says so, not moving. Frames with no planted foot and both
    ankles high are airborne; an airborne run also claims the unplanted
    frames on either side of it (push-off and touch-down), which is where
    a height-only detector fires late and eats the launch
    (docs/PIPELINE.md, `none` tuning notes). Anything else unplanted
    falls back to `both`, which lets the apply plant the lower foot.
    """
    n = len(la_h)
    lp = la_h < PLANT_DOWN
    rp = ra_h < PLANT_DOWN
    if st_l is not None:
        lp &= st_l > STATIC_ON
        rp &= st_r > STATIC_ON
    lab = []
    for i in range(n):
        if lp[i] and rp[i]:
            lab.append("both")
        elif lp[i]:
            lab.append("left")
        elif rp[i]:
            lab.append("right")
        elif la_h[i] > PLANT_UP and ra_h[i] > PLANT_UP:
            lab.append("none")
        else:
            lab.append("?")
    for i in range(n):                      # airborne runs claim adjacent '?' frames
        if lab[i] == "none":
            j = i - 1
            while j >= 0 and lab[j] == "?":
                lab[j] = "none"
                j -= 1
            j = i + 1
            while j < n and lab[j] == "?":
                lab[j] = "none"
                j += 1
    lab = ["both" if s == "?" else s for s in lab]

    runs = []
    for i, s in enumerate(lab):
        if runs and runs[-1][2] == s:
            runs[-1][1] = i
        else:
            runs.append([i, i, s])
    merged = []
    for r in runs:                          # fold flickers into their predecessor
        if merged and (r[1] - r[0] + 1) < MIN_RUN:
            merged[-1][1] = r[1]
        elif merged and merged[-1][2] == r[2]:
            merged[-1][1] = r[1]
        else:
            merged.append(list(r))
    if len(merged) > 1 and (merged[0][1] - merged[0][0] + 1) < MIN_RUN:
        merged[1][0] = merged[0][0]
        merged.pop(0)
    return [(a + 1, b + 1, s) for a, b, s in merged]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks", required=True, type=Path)
    ap.add_argument("--step", type=int, default=6)
    args = ap.parse_args()
    lm = args.landmarks if args.landmarks.is_absolute() else REPO / args.landmarks
    d = json.loads(lm.read_text(encoding="utf-8"))
    F = d["frames"]
    n = len(F)

    def w(i, name):
        l = F[i]["world"][name]
        return np.array([l["x"], l["y"], l["z"]])

    has_ph = "pelvis_height" in F[0]
    ph = np.array([f.get("pelvis_height", 0.0) for f in F])

    def height(i, name):
        # y is down and per-frame mid-hip centered; pelvis_height restores ground.
        return (ph[i] if has_ph else 0.0) - w(i, name)[1] - (0.0 if has_ph else 0.0)

    la_h = np.array([height(i, "left_ankle") for i in range(n)])
    ra_h = np.array([height(i, "right_ankle") for i in range(n)])
    lk_h = np.array([height(i, "left_knee") for i in range(n)])
    rk_h = np.array([height(i, "right_knee") for i in range(n)])
    lw_z = np.array([w(i, "left_wrist")[2] for i in range(n)])
    rw_z = np.array([w(i, "right_wrist")[2] for i in range(n)])
    span = np.array([np.linalg.norm(w(i, "left_wrist")[[0, 2]] - w(i, "right_wrist")[[0, 2]]) for i in range(n)])
    yaw = np.array([np.degrees(np.arctan2(-(w(i, "left_shoulder") - w(i, "right_shoulder"))[2],
                                          (w(i, "left_shoulder") - w(i, "right_shoulder"))[0])) for i in range(n)])

    print("  f |  ph   | laH   raH  | lkH   rkH  | lwZ    rwZ   | span  yaw")
    for i in range(0, n, max(1, args.step)):
        print(f"{i+1:4d} | {ph[i]:.3f} | {la_h[i]:.2f}  {ra_h[i]:.2f} | {lk_h[i]:.2f}  {rk_h[i]:.2f} "
              f"| {lw_z[i]:+.2f}  {rw_z[i]:+.2f} | {span[i]:.2f}  {yaw[i]:+4.0f}")

    def segments(idx, gap=3):
        return np.split(idx, np.where(np.diff(idx) > gap)[0] + 1) if len(idx) else []

    if has_ph:
        base = float(np.median(ph[: max(10, n // 12)]))
        print(f"\npelvis baseline {base:.3f} | min {ph.min():.3f} @ f{ph.argmin()+1} | max {ph.max():.3f} @ f{ph.argmax()+1}")
        air = np.where((la_h > 0.25) & (ra_h > 0.25))[0]
        for s in segments(air):
            print(f"both feet airborne (>0.25): f{s[0]+1}-f{s[-1]+1} "
                  f"(peak lower-ankle {np.minimum(la_h, ra_h)[s].max():.2f} m)")
        for nm, up, other in (("left", la_h, ra_h), ("right", ra_h, la_h)):
            leg = np.where((up > 0.30) & (other < 0.15))[0]
            for s in segments(leg):
                print(f"{nm} leg up on {('right' if nm == 'left' else 'left')} support: "
                      f"f{s[0]+1}-f{s[-1]+1} (ankle peak {up[s].max():.2f} @ f{s[up[s].argmax()]+1})")

    for nm, z in (("left", lw_z), ("right", rw_z)):
        for s in segments(np.where(z < -0.32)[0]):
            print(f"{nm} wrist toward camera (z<-0.32): f{s[0]+1}-f{s[-1]+1} (min {z[s].min():+.2f} @ f{s[z[s].argmin()]+1})")

    # Strikes in ANY direction. The z test above only sees punches thrown
    # at the camera; on a side-on plate (the duel) they travel across the
    # frame. Extension = wrist-to-shoulder distance over the arm's length.
    for nm in ("left", "right"):
        sh = np.array([w(i, f"{nm}_shoulder") for i in range(n)])
        el = np.array([w(i, f"{nm}_elbow") for i in range(n)])
        wr = np.array([w(i, f"{nm}_wrist") for i in range(n)])
        arm_len = np.median(np.linalg.norm(el - sh, axis=1) + np.linalg.norm(wr - el, axis=1))
        ext = np.linalg.norm(wr - sh, axis=1) / max(arm_len, 1e-6)
        tpose = span > 1.22
        for s in segments(np.where((ext > EXTENDED) & ~tpose)[0]):
            k = s[int(ext[s].argmax())]
            print(f"{nm} arm extended (>{EXTENDED:.0%} of its length, any direction): "
                  f"f{s[0]+1}-f{s[-1]+1} (peak {ext[k]:.0%} @ f{k+1})")

    for s in segments(np.where(span > 1.22)[0], gap=4):
        print(f"T-pose span (>1.22 m): f{s[0]+1}-f{s[-1]+1}")
    print(f"shoulder-line yaw range: {yaw.min():+.0f}..{yaw.max():+.0f} deg "
          "(punches rotate shoulders ±40-60 without turning the hips)")

    if "pelvis_height_incam" in F[0]:
        phc = np.array([f["pelvis_height_incam"] for f in F])

        def excursion(a):
            base_ = float(np.median(a[: max(10, n // 12)]))
            return float(a.max() - base_), float(base_ - a.min())

        (gu, gd), (cu, cd) = excursion(ph), excursion(phc)
        print(f"pelvis arc, gravity-aligned vs camera frame: rise {gu:.3f} vs {cu:.3f} m, "
              f"dip {gd:.3f} vs {cd:.3f} m"
              + (f"  (camera/global rise x{cu / gu:.2f})" if gu > 0.05 else ""))

    # Where the 3D limb is a guess. ViTPose's per-keypoint confidence
    # (`visibility`, current estimator) drops when a limb is occluded or
    # motion-blurred; GVHMR still returns a limb there, from its prior.
    vis = {nm: np.array([F[i]["world"][nm].get("visibility", 1.0) for i in range(n)])
           for nm in ("nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                      "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                      "left_hip", "right_hip", "left_knee", "right_knee",
                      "left_ankle", "right_ankle")}
    if any(float(v.min()) < 0.999 for v in vis.values()):
        found = False
        for limb, joints in LIMBS.items():
            lo = np.min([vis[j] for j in joints], axis=0)
            for s in segments(np.where(lo < LOW_CONF)[0], gap=2):
                if len(s) < MIN_RUN:
                    continue
                found = True
                worst = min(joints, key=lambda j: vis[j][s].min())
                print(f"low 2D confidence, {limb}: f{s[0]+1}-f{s[-1]+1} "
                      f"(min {lo[s].min():.2f} at {worst}) — occluded or blurred; the 3D limb here is a guess")
        if not found:
            print(f"2D confidence: every limb above {LOW_CONF} on every frame")
    else:
        print("2D confidence: not in this landmarks file (re-run the estimator to get it)")

    # A DRAFT support schedule. The estimator's per-foot static confidence
    # is its own contact detector; heights alone are the fallback.
    if has_ph:
        st = None
        if "static" in F[0]:
            def foot_static(side):
                return np.array([max(F[i]["static"][f"{side}_ankle"], F[i]["static"][f"{side}_foot_index"])
                                 for i in range(n)])
            st = (foot_static("left"), foot_static("right"))
        plan = draft_plant(la_h, ra_h, *(st or (None, None)))
        src = "GVHMR static confidence + ankle height" if st else "ankle height only (no `static` in this file)"
        print(f"\ndraft plant schedule — from {src}; check each window against the events above:")
        print('  "plant": [')
        for k, (a, b, s) in enumerate(plan):
            note = ""
            if s in ("left", "right"):
                other, h = ("right", ra_h) if s == "left" else ("left", la_h)
                k_ = a - 1 + int(h[a - 1:b].argmax())
                note = f', "comment": "draft: {other} ankle peaks {h[k_]:.2f} m @ src{k_ + 1}"'
            elif s == "none":
                note = f', "comment": "draft: lower ankle peaks {np.minimum(la_h, ra_h)[a - 1:b].max():.2f} m"'
            print(f'    {{ "src": [{a}, {b}], "support": "{s}"{note} }}{"," if k < len(plan) - 1 else ""}')
        print("  ]")


if __name__ == "__main__":
    main()
