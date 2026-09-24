"""Generic lift: estimator landmarks -> Mixamo-world joints, driven by an
action_spec JSON (see docs/PIPELINE.md for the spec schema).

Usage:
  python pipeline\\lift_to_mixamo.py --spec action_specs\\<motion>.json

This is a direction-preserving retarget: keep the estimator's segment
DIRECTIONS, rebuild positions from the Mixamo sockets with Mixamo bone
lengths. With mesh-quality input (GVHMR) no inverse kinematics, no
authored limb targets and no action envelopes are needed. The spec
contributes only what the estimator cannot know:

  - where the clip blends into exact Y Bot rest (start and/or end)
  - when fists close
  - which foot is the support in each phase (left/right/both/none)
  - authored arm overrides for beats where the owner's read of the
    video beats the estimator (e.g. an occluded arm)
  - the QA frames

Two structural jobs happen here rather than in the FK apply:

  - foot pinning: the estimator world is per-frame hip-centered, so a
    planted foot drifts whenever the body moves over it. The WHOLE pose is
    translated so every planted foot stays where it touched down (the
    pelvis then sways over the feet, which is the correct physics) — in
    single-support windows and, since `pin: "contacts"`, in `both` windows
    too, judged per foot from the support schedule, the foot heights and
    the estimator's static confidence. `pin: "single"` restores the old
    rule (support ankle only, released over 10 frames).
  - hips world Z stays at rest height; the FK apply searches the real
    hip height per frame so the support foot plants at ground level —
    that is what makes crouches and wide stances drop the pelvis. For
    airborne windows the per-frame `pelvis_height` is passed through and
    the apply integrates the arc instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

REPO = Path(__file__).resolve().parents[1]

# Y Bot rest joints, world meters (measured from the live rig; see docs/RIG.md).
REST = {
    "hips": np.array([0.0, 0.0, 0.99792]),
    "spine": np.array([0.0, 0.01227, 1.09715]),
    "spine1": np.array([0.0, 0.0265, 1.21361]),
    "spine2": np.array([0.0, 0.04276, 1.34721]),
    "neck": np.array([0.0, 0.03483, 1.49754]),
    "head": np.array([0.0, 0.00341, 1.60075]),
    "l_shoulder": np.array([0.06103, 0.03571, 1.43832]),
    "l_arm": np.array([0.18758, 0.06171, 1.43567]),
    "l_elbow": np.array([0.46163, 0.06171, 1.4357]),
    "l_wrist": np.array([0.73777, 0.06171, 1.43572]),
    "l_hand": np.array([0.84757, 0.06171, 1.43573]),
    "r_shoulder": np.array([-0.06109, 0.03571, 1.43831]),
    "r_arm": np.array([-0.18764, 0.06171, 1.43564]),
    "r_elbow": np.array([-0.46168, 0.06171, 1.43562]),
    "r_wrist": np.array([-0.73783, 0.06171, 1.43559]),
    "r_hand": np.array([-0.84763, 0.06171, 1.43558]),
    "l_upleg": np.array([0.09124, 0.00055, 0.93136]),
    "l_knee": np.array([0.09369, 0.00571, 0.5254]),
    "l_ankle": np.array([0.09124, 0.0263, 0.10492]),
    "l_foot": np.array([0.09498, -0.11336, 0.03284]),
    "l_toe": np.array([0.09356, -0.21334, 0.03108]),
    "r_upleg": np.array([-0.09124, 0.00055, 0.93136]),
    "r_knee": np.array([-0.09369, 0.0057, 0.5254]),
    "r_ankle": np.array([-0.09124, 0.0263, 0.10492]),
    "r_foot": np.array([-0.09498, -0.11336, 0.03284]),
    "r_toe": np.array([-0.09356, -0.21334, 0.03109]),
}

# Mixamo bone lengths, meters.
LEN = {
    "spine1": 0.13459,
    "spine2": 0.12344,
    "neck": 0.10790,
    "head": 0.19630,
    "l_shoulder": 0.12922,
    "l_arm": 0.27405,
    "l_fore": 0.27614,
    "l_hand": 0.10980,
    "r_shoulder": 0.12922,
    "r_arm": 0.27405,
    "r_fore": 0.27614,
    "r_hand": 0.10980,
    "l_upleg": 0.40599,
    "l_leg": 0.42099,
    "l_foot": 0.15722,
    "r_upleg": 0.40599,
    "r_leg": 0.42099,
    "r_foot": 0.15722,
}

def load_profile(path=None) -> Path | None:
    """Adopt a character's measured proportions.

    A rig profile (written by setup_rig.py / setup_duo.py) overrides the
    built-in Y Bot measurements — this is what makes the lift work with
    any Mixamo character. A multi-character scene has one profile per
    character, named by the spec's `rig_profile`; a solo scene falls back
    to rig_profile.json at the repo root.
    """
    global REST, LEN, HIP_Z
    p = (Path(path) if Path(path).is_absolute() else REPO / path) if path else (REPO / "rig_profile.json")
    if p.exists():
        _p = json.loads(p.read_text(encoding="utf-8"))
        REST = {k: np.array(v, dtype=float) for k, v in _p["rest"].items()}
        LEN = {k: float(v) for k, v in _p["lengths"].items()}
        HIP_Z = float(REST["hips"][2])
        return p
    if path:
        raise SystemExit(f"rig profile not found: {p}")
    return None

MP_USED = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_index", "right_index",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

HIP_Z = float(REST["hips"][2])

# Per-frame keys of a lifted pose that are DIRECTIONS, not positions: they
# must never be translated (pinning, root motion, stage placement).
DIR_PREFIXES = ("basis_", "pelvis_", "chest_")
DIR_KEYS = {"neck_up", "l_hand_lat", "r_hand_lat"}

STATIC_KEYS = ("left_ankle", "left_foot_index", "right_ankle", "right_foot_index")
PIN_LAMBDA = 0.02   # pull of "keep the previous offset" against the planted contacts
PIN_W_MIN = 0.02    # below this weight a contact point is released
PIN_TOL = 0.01      # metres two planted feet may disagree before one is judged moving


def is_dir(k: str) -> bool:
    return k.startswith(DIR_PREFIXES) or k in DIR_KEYS


def rpath(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO / p)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros(3)


def smoother(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def lerp(a, b, t):
    return a * (1.0 - t) + b * t


def mp_to_mix(p):
    """Estimator world (y down, z away-from-camera) -> Mixamo world
    (z up, character faces -Y, left +X)."""
    return np.array([p[0], p[2], -p[1] + HIP_Z], dtype=np.float64)


PREFILTER = 7    # default landmark prefilter width, in SOURCE frames


def smooth_series(arr, window=None, poly=2):
    window = PREFILTER if window is None else window
    n = arr.shape[0]
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 5:
        return arr
    out = np.empty_like(arr)
    for c in range(arr.shape[1]):
        out[:, c] = savgol_filter(arr[:, c], window_length=w, polyorder=poly, mode="interp")
    return out


def load_mp(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = [f for f in data["frames"] if f.get("ok")]
    world = {n: np.zeros((len(frames), 3)) for n in MP_USED}
    times = np.zeros(len(frames))
    pelvis_h = np.zeros(len(frames))
    pelvis_h_incam = np.full(len(frames), np.nan)   # older landmarks files lack it
    gaze = np.zeros((len(frames), 3))
    # Ground trajectory, two independent estimates (see main()).
    root = np.zeros((len(frames), 2))
    incam = np.zeros((len(frames), 3))
    # GVHMR's own "this joint is not moving" probability per foot point
    # (current estimator only): the evidence contact pinning weighs.
    static = {k: np.full(len(frames), np.nan) for k in STATIC_KEYS}
    # Extra skeleton points (current estimator only): spine column,
    # collars, knuckles, toe tips — see estimate_pose_gvhmr.py BODY_*.
    names = sorted(frames[0].get("body", {})) if frames else []
    body = {n: np.full((len(frames), 3), np.nan) for n in names}
    for i, f in enumerate(frames):
        times[i] = f["t"]
        pelvis_h[i] = float(f.get("pelvis_height", 0.0))
        if "pelvis_height_incam" in f:
            pelvis_h_incam[i] = float(f["pelvis_height_incam"])
        g = f.get("gaze")
        if g:
            gaze[i] = g
        r = f.get("root")
        if r:
            root[i] = r
        ic = f.get("incam_root")
        if ic:
            incam[i] = ic
        for k in STATIC_KEYS:
            if k in f.get("static", {}):
                static[k][i] = float(f["static"][k])
        for n in names:
            if n in f.get("body", {}):
                body[n][i] = f["body"][n]
        for n in MP_USED:
            w = f["world"][n]
            world[n][i] = (w["x"], w["y"], w["z"])
    aux = {"static": None if any(np.isnan(v).any() for v in static.values()) else static,
           "body": body if body and not any(np.isnan(v).any() for v in body.values()) else None}
    return data["fps"], times, world, pelvis_h, gaze, root, incam, pelvis_h_incam, aux


def resample(times, series, dst_times):
    out = {}
    for k, arr in series.items():
        sm = smooth_series(arr)
        cols = [np.interp(dst_times, times, sm[:, c]) for c in range(arr.shape[1])]
        out[k] = np.stack(cols, axis=1)
    return out


def place_chain(up, left):
    z = unit(up)
    x = unit(left - z * np.dot(left, z))
    y = unit(np.cross(z, x))
    x = unit(np.cross(y, z))
    return x, y, z


def rest_pose():
    out = {k: v.copy() for k, v in REST.items()}
    for pre in ("basis_", "pelvis_", "chest_"):
        out[pre + "x"] = np.array([1.0, 0.0, 0.0])
        out[pre + "y"] = np.array([0.0, 1.0, 0.0])
        out[pre + "z"] = np.array([0.0, 0.0, 1.0])
    out["neck_up"] = np.array([0.0, 0.0, 1.0])
    # Index-to-pinky knuckle direction in a palms-down T-pose: character back.
    out["l_hand_lat"] = np.array([0.0, 1.0, 0.0])
    out["r_hand_lat"] = np.array([0.0, 1.0, 0.0])
    return out


def blend_pose(a, b, t):
    """Blend pose `a` toward `b` over `a`'s own keys (the rest pose carries
    frames a rigid-torso lift does not emit; they must not appear)."""
    t = float(np.clip(t, 0.0, 1.0))
    out = {}
    for k in a:
        if k not in b:
            out[k] = a[k]
        else:
            out[k] = lerp(np.asarray(a[k], float), np.asarray(b[k], float), t)
    return out


def seg(mp, i, a, b):
    """Unit direction of estimator segment a->b at frame i."""
    return unit(mp[b][i] - mp[a][i])


def rodrigues(v, axis, angle):
    axis = unit(axis)
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


def two_bone(p0, target, l1, l2, pole):
    """Place a two-bone chain from p0 to `target` with exact lengths,
    bending toward `pole`. Used only to re-place a chain after its
    end effector was moved (spec `reach`), never as an IK solver."""
    v = target - p0
    d_raw = np.linalg.norm(v)
    if d_raw < 1e-8:
        return p0 + unit(pole - p0) * l1, p0 + np.array([0.0, 0.0, -1.0]) * (l1 + l2)
    d = float(np.clip(d_raw, 1e-3, l1 + l2 - 1e-4))
    end = p0 + v * (d / d_raw)
    cos_a = (l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d)
    a = float(np.arccos(np.clip(cos_a, -1.0, 1.0)))
    n = np.cross(end - p0, pole - p0)
    if np.linalg.norm(n) < 1e-6:
        n = np.cross(end - p0, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(n) < 1e-6:
        n = np.array([1.0, 0.0, 0.0])
    mid = p0 + unit(rodrigues(unit(end - p0), n, a)) * l1
    return mid, mid + unit(end - mid) * l2


def reconstruct(mp, i):
    """Direction-preserving retarget of one frame onto Mixamo lengths."""
    ls, rs = mp["left_shoulder"][i], mp["right_shoulder"][i]
    lh, rh = mp["left_hip"][i], mp["right_hip"][i]
    nose = mp["nose"][i]

    hip_mid = 0.5 * (lh + rh)
    sh_mid = 0.5 * (ls + rs)
    up = unit(sh_mid - hip_mid)
    if np.linalg.norm(up) < 0.2:
        up = np.array([0.0, 0.0, 1.0])
    left = unit((ls - rs) + (lh - rh) * 0.35)
    x, y, z = place_chain(up, left)

    hips = np.array([hip_mid[0], hip_mid[1], HIP_Z])

    def in_basis(local_xyz):
        return hips + x * local_xyz[0] + y * local_xyz[1] + z * local_xyz[2]

    # Torso: the SMPL-derived landmarks carry no spine names, so the
    # chest chain is built from the hip basis + shoulder line.
    chest_dir = unit(0.65 * z + 0.35 * up)
    spine = hips + chest_dir * np.linalg.norm(REST["spine"] - REST["hips"])
    spine1 = spine + chest_dir * LEN["spine1"]
    spine2 = spine1 + chest_dir * LEN["spine2"]
    neck = spine2 + z * LEN["neck"]
    head = neck + z * LEN["head"] * 0.85 + unit(np.array([nose[0], nose[1], 0.0])) * 0.02

    # Sockets from the posed hip basis.
    l_shoulder = in_basis(REST["l_shoulder"] - REST["hips"])
    r_shoulder = in_basis(REST["r_shoulder"] - REST["hips"])
    l_arm = l_shoulder + unit((ls - sh_mid) + x * 0.35) * LEN["l_shoulder"]
    r_arm = r_shoulder + unit((rs - sh_mid) - x * 0.35) * LEN["r_shoulder"]
    l_upleg = in_basis(REST["l_upleg"] - REST["hips"])
    r_upleg = in_basis(REST["r_upleg"] - REST["hips"])

    # Limbs: estimator directions, Mixamo lengths.
    l_elbow = l_arm + seg(mp, i, "left_shoulder", "left_elbow") * LEN["l_arm"]
    l_wrist = l_elbow + seg(mp, i, "left_elbow", "left_wrist") * LEN["l_fore"]
    l_hand = l_wrist + seg(mp, i, "left_wrist", "left_index") * LEN["l_hand"]
    r_elbow = r_arm + seg(mp, i, "right_shoulder", "right_elbow") * LEN["r_arm"]
    r_wrist = r_elbow + seg(mp, i, "right_elbow", "right_wrist") * LEN["r_fore"]
    r_hand = r_wrist + seg(mp, i, "right_wrist", "right_index") * LEN["r_hand"]

    l_knee = l_upleg + seg(mp, i, "left_hip", "left_knee") * LEN["l_upleg"]
    l_ankle = l_knee + seg(mp, i, "left_knee", "left_ankle") * LEN["l_leg"]
    r_knee = r_upleg + seg(mp, i, "right_hip", "right_knee") * LEN["r_upleg"]
    r_ankle = r_knee + seg(mp, i, "right_knee", "right_ankle") * LEN["r_leg"]

    l_foot = l_ankle + seg(mp, i, "left_ankle", "left_foot_index") * LEN["l_foot"]
    r_foot = r_ankle + seg(mp, i, "right_ankle", "right_foot_index") * LEN["r_foot"]
    # Toe direction in FULL 3D. Flattening it to the ground plane is
    # right for a planted foot and wrong for a kicking one: it drags a
    # foot pointed 40 deg up back to horizontal, which both lowers the
    # toe and throws it forward into whatever the foot is aimed at. A
    # grounded foot does not need the flattening here anyway — the FK
    # apply re-aims any foot below 0.20 m along its own heading
    # (`flatten_foot`), so the ground case is already handled downstream.
    l_fwd = unit(mp["left_foot_index"][i] - mp["left_heel"][i])
    r_fwd = unit(mp["right_foot_index"][i] - mp["right_heel"][i])
    if np.linalg.norm(l_fwd) < 0.1:
        l_fwd = -y
    if np.linalg.norm(r_fwd) < 0.1:
        r_fwd = -y
    l_toe = l_foot + l_fwd * 0.10
    r_toe = r_foot + r_fwd * 0.10

    return {
        "hips": hips, "spine": spine, "spine1": spine1, "spine2": spine2,
        "neck": neck, "head": head,
        "l_shoulder": l_shoulder, "l_arm": l_arm, "l_elbow": l_elbow,
        "l_wrist": l_wrist, "l_hand": l_hand,
        "r_shoulder": r_shoulder, "r_arm": r_arm, "r_elbow": r_elbow,
        "r_wrist": r_wrist, "r_hand": r_hand,
        "l_upleg": l_upleg, "l_knee": l_knee, "l_ankle": l_ankle,
        "l_foot": l_foot, "l_toe": l_toe,
        "r_upleg": r_upleg, "r_knee": r_knee, "r_ankle": r_ankle,
        "r_foot": r_foot, "r_toe": r_toe,
        "basis_x": x, "basis_y": y, "basis_z": z,
    }


def _rest_dist(a, b):
    return float(np.linalg.norm(REST[b] - REST[a]))


def reconstruct_twist(mp, i, body=None):
    """reconstruct() with a pelvis and a chest instead of one rigid torso.

    reconstruct() gives the Hips one frame, mostly the shoulder line
    (shoulders + 0.35 x hips), and builds the spine as a straight rod along
    it: when the performer turned the shoulders 45 degrees with the hips
    square, the retarget turned the HIPS 39 degrees, dragged the leg sockets
    round with them, and had no spine to twist or bend.

    Here the pelvis frame comes from the hip line and the chest frame from
    the shoulder line (the collar line when the estimator provides it); the
    legs hang off the pelvis and the arms off the chest; the solver spreads
    the twist between them over Spine / Spine1 / Spine2. With the
    estimator's `body` block the spine follows the performer's own spine
    joints (it bends), the clavicles follow their collars, the hands point
    at their middle knuckles and carry their roll (index-to-pinky line), and
    the toes point at the real toe tips.

    `basis_*` is still the old mixed frame: arm_overrides, arm_pose and
    head_look were sized in it, and their numbers keep their meaning.
    """
    ls, rs = mp["left_shoulder"][i], mp["right_shoulder"][i]
    lh, rh = mp["left_hip"][i], mp["right_hip"][i]
    nose = mp["nose"][i]
    hip_mid = 0.5 * (lh + rh)
    sh_mid = 0.5 * (ls + rs)
    up = unit(sh_mid - hip_mid)
    if np.linalg.norm(up) < 0.2:
        up = np.array([0.0, 0.0, 1.0])
    x, y, z = place_chain(up, unit((ls - rs) + (lh - rh) * 0.35))
    hips = np.array([hip_mid[0], hip_mid[1], HIP_Z])

    def B(n):
        return body[n][i]

    if body is not None:
        up_p = unit(B("spine1") - hip_mid)
        up_c = unit(B("neck") - B("spine3"))
        lat_c = B("left_collar") - B("right_collar")
    else:
        up_p = up_c = up
        lat_c = ls - rs
    xp, yp, zp = place_chain(up_p if np.linalg.norm(up_p) > 0.2 else up, lh - rh)
    xc, yc, zc = place_chain(up_c if np.linalg.norm(up_c) > 0.2 else up, lat_c)

    def in_frame(origin, bx, by, bz, off):
        return origin + bx * off[0] + by * off[1] + bz * off[2]

    if body is not None:
        spine = in_frame(hips, xp, yp, zp, REST["spine"] - REST["hips"])
        spine1 = spine + unit(B("spine2") - B("spine1")) * _rest_dist("spine", "spine1")
        spine2 = spine1 + unit(B("spine3") - B("spine2")) * _rest_dist("spine1", "spine2")
        neck = spine2 + unit(B("neck") - B("spine3")) * _rest_dist("spine2", "neck")
        head = neck + unit(B("head") - B("neck")) * _rest_dist("neck", "head")
        l_shoulder = in_frame(spine2, xc, yc, zc, REST["l_shoulder"] - REST["spine2"])
        r_shoulder = in_frame(spine2, xc, yc, zc, REST["r_shoulder"] - REST["spine2"])
        l_arm = l_shoulder + unit(ls - B("left_collar")) * LEN["l_shoulder"]
        r_arm = r_shoulder + unit(rs - B("right_collar")) * LEN["r_shoulder"]
    else:
        spine = hips + up * np.linalg.norm(REST["spine"] - REST["hips"])
        spine1 = spine + up * LEN["spine1"]
        spine2 = spine1 + up * LEN["spine2"]
        neck = spine2 + up * LEN["neck"]
        head = neck + up * LEN["head"] * 0.85 + unit(np.array([nose[0], nose[1], 0.0])) * 0.02
        l_shoulder = in_frame(hips, xc, yc, zc, REST["l_shoulder"] - REST["hips"])
        r_shoulder = in_frame(hips, xc, yc, zc, REST["r_shoulder"] - REST["hips"])
        l_arm = l_shoulder + unit((ls - sh_mid) + xc * 0.35) * LEN["l_shoulder"]
        r_arm = r_shoulder + unit((rs - sh_mid) - xc * 0.35) * LEN["r_shoulder"]
    l_upleg = in_frame(hips, xp, yp, zp, REST["l_upleg"] - REST["hips"])
    r_upleg = in_frame(hips, xp, yp, zp, REST["r_upleg"] - REST["hips"])

    l_elbow = l_arm + seg(mp, i, "left_shoulder", "left_elbow") * LEN["l_arm"]
    l_wrist = l_elbow + seg(mp, i, "left_elbow", "left_wrist") * LEN["l_fore"]
    r_elbow = r_arm + seg(mp, i, "right_shoulder", "right_elbow") * LEN["r_arm"]
    r_wrist = r_elbow + seg(mp, i, "right_elbow", "right_wrist") * LEN["r_fore"]
    if body is not None:
        l_hand = l_wrist + unit(B("left_middle1") - B("left_wrist_x")) * LEN["l_hand"]
        r_hand = r_wrist + unit(B("right_middle1") - B("right_wrist_x")) * LEN["r_hand"]
    else:
        l_hand = l_wrist + seg(mp, i, "left_wrist", "left_index") * LEN["l_hand"]
        r_hand = r_wrist + seg(mp, i, "right_wrist", "right_index") * LEN["r_hand"]

    l_knee = l_upleg + seg(mp, i, "left_hip", "left_knee") * LEN["l_upleg"]
    l_ankle = l_knee + seg(mp, i, "left_knee", "left_ankle") * LEN["l_leg"]
    r_knee = r_upleg + seg(mp, i, "right_hip", "right_knee") * LEN["r_upleg"]
    r_ankle = r_knee + seg(mp, i, "right_knee", "right_ankle") * LEN["r_leg"]
    l_foot = l_ankle + seg(mp, i, "left_ankle", "left_foot_index") * LEN["l_foot"]
    r_foot = r_ankle + seg(mp, i, "right_ankle", "right_foot_index") * LEN["r_foot"]
    if body is not None:
        l_fwd = unit(B("left_big_toe") - B("left_foot_x"))
        r_fwd = unit(B("right_big_toe") - B("right_foot_x"))
    else:
        l_fwd = unit(mp["left_foot_index"][i] - mp["left_heel"][i])
        r_fwd = unit(mp["right_foot_index"][i] - mp["right_heel"][i])
    if np.linalg.norm(l_fwd) < 0.1:
        l_fwd = -y
    if np.linalg.norm(r_fwd) < 0.1:
        r_fwd = -y
    l_toe = l_foot + l_fwd * 0.10
    r_toe = r_foot + r_fwd * 0.10

    out = {
        "hips": hips, "spine": spine, "spine1": spine1, "spine2": spine2,
        "neck": neck, "head": head,
        "l_shoulder": l_shoulder, "l_arm": l_arm, "l_elbow": l_elbow,
        "l_wrist": l_wrist, "l_hand": l_hand,
        "r_shoulder": r_shoulder, "r_arm": r_arm, "r_elbow": r_elbow,
        "r_wrist": r_wrist, "r_hand": r_hand,
        "l_upleg": l_upleg, "l_knee": l_knee, "l_ankle": l_ankle,
        "l_foot": l_foot, "l_toe": l_toe,
        "r_upleg": r_upleg, "r_knee": r_knee, "r_ankle": r_ankle,
        "r_foot": r_foot, "r_toe": r_toe,
        "basis_x": x, "basis_y": y, "basis_z": z,
        "pelvis_x": xp, "pelvis_y": yp, "pelvis_z": zp,
        "chest_x": xc, "chest_y": yc, "chest_z": zc,
    }
    if body is not None:
        out["neck_up"] = unit(head - neck)
        out["l_hand_lat"] = unit(B("left_pinky1") - B("left_index1"))
        out["r_hand_lat"] = unit(B("right_pinky1") - B("right_index1"))
    return out


def _ramp(x, lo, hi):
    """0 at `lo`, 1 at `hi`, smooth in between."""
    return smoother((x - lo) / (hi - lo))


def contact_weights(plant, h_ankle, h_ball, static=None) -> dict:
    """How planted each foot point is, per destination frame.

    `plant` is the spec's support per frame; `h_ankle`/`h_ball` the
    performer's ankle and ball heights above the floor ({"l": arr, "r": arr});
    `static` GVHMR's per-point "not moving" probability, when the landmarks
    carry it. Returns {("l"|"r", "ankle"|"ball"): weights}.

    - The spec's single-support foot is planted, full stop (it always was).
    - In `both` windows each foot is judged on its own: near the floor
      (ankle under 0.12 m or ball under 0.06 m, fading out by 0.20 / 0.12),
      and — when the estimator says so — not moving. A foot that steps
      inside a `both` window lifts, lets go of its anchor, and plants again
      where it lands.
    - Within a foot, the estimator's static split says whether the heel or
      the ball holds (a pivot keeps the ball down while the heel turns);
      without it, the ankle holds, as the old single-support pin did.
    """
    n = len(plant)
    W = {(s, p): np.zeros(n) for s in "lr" for p in ("ankle", "ball")}
    for i in range(n):
        for s, side in (("l", "left"), ("r", "right")):
            sup = plant[i]
            if sup == "none" or (sup in ("left", "right") and sup != side):
                continue
            if sup == side:
                c = 1.0
            else:
                c = max(1.0 - _ramp(h_ankle[s][i], 0.12, 0.20), 1.0 - _ramp(h_ball[s][i], 0.06, 0.12))
                if static is not None:
                    c *= _ramp(max(static[s, "ankle"][i], static[s, "ball"][i]), 0.3, 0.7)
            share = 1.0
            if static is not None:
                sa, sb = static[s, "ankle"][i], static[s, "ball"][i]
                share = 0.5 if sa + sb < 0.1 else sa / (sa + sb)
            W[s, "ankle"][i] = c * share
            W[s, "ball"][i] = c * (1.0 - share)
    return W


def pin_offsets(recs, W, lam=PIN_LAMBDA) -> np.ndarray:
    """Horizontal offset per frame that keeps every planted foot point on
    the spot where it touched down.

    Each point takes an anchor when its weight rises (its world position at
    that moment, so a new contact never jerks the body) and drops it when
    the weight falls. The offset is the weighted least-squares fit of all
    live anchors, plus a small pull toward the previous frame's offset,
    which carries it through flight and eases contacts in and out. With
    hip-centred landmarks this is what puts the pelvis over the feet: the
    body moves, the planted feet do not.
    """
    pts = {("l", "ankle"): "l_ankle", ("l", "ball"): "l_foot",
           ("r", "ankle"): "r_ankle", ("r", "ball"): "r_foot"}
    out = np.zeros((len(recs), 2))
    delta = np.zeros(2)
    anchors = {}
    for i, rec in enumerate(recs):
        live = {}
        for key, joint in pts.items():
            w = float(W[key][i])
            if w <= PIN_W_MIN:
                anchors.pop(key, None)
                continue
            p = np.asarray(rec[joint], float)[:2]
            if key not in anchors:
                anchors[key] = p + delta
            live[key] = (w, p)
        # A foot that moves while nominally planted (a kick starting inside a
        # `both` window, a shuffle — frequent when no static confidence says
        # otherwise) shows up as the two feet asking for different offsets.
        # Averaging them would slide BOTH feet and snap when one lets go:
        # the one asking for the bigger change is the one moving, so it is
        # re-anchored where it stands and the other foot holds the body.
        implied = {}
        for s in "lr":
            ks = [k for k in live if k[0] == s]
            if ks:
                ws = sum(live[k][0] for k in ks)
                implied[s] = sum(live[k][0] * (anchors[k] - live[k][1]) for k in ks) / ws
        if len(implied) == 2 and np.linalg.norm(implied["l"] - implied["r"]) > PIN_TOL:
            moving = max("lr", key=lambda s: float(np.linalg.norm(implied[s] - delta)))
            keep = "r" if moving == "l" else "l"
            for k in live:
                if k[0] == moving:
                    anchors[k] = live[k][1] + implied[keep]
        num, den = lam * delta, lam
        for k, (w, p) in live.items():
            num = num + w * (anchors[k] - p)
            den += w
        delta = num / den
        out[i] = delta
    return out


def window_amount(dest_f, rise, fall, src2dest):
    """1 inside [rise..fall] windows given in src frames, smooth edges."""
    r0, r1 = src2dest(rise[0]), src2dest(rise[1])
    f0, f1 = src2dest(fall[0]), src2dest(fall[1])
    if dest_f <= r0 or dest_f >= f1:
        return 0.0
    if dest_f < r1:
        return smoother((dest_f - r0) / max(1.0, r1 - r0))
    if dest_f <= f0:
        return 1.0
    return 1.0 - smoother((dest_f - f0) / max(1.0, f1 - f0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, type=Path)
    args = ap.parse_args()
    spec = json.loads(rpath(args.spec).read_text(encoding="utf-8"))
    used_profile = load_profile(spec.get("rig_profile"))
    # Landmark prefilter width. 7 frames is right for ordinary motion and
    # measurably expensive on a fast strike: on the duel plate's roundhouse
    # it cost 5 deg of shin angle at the peak — the same lesson as
    # docs/PITFALLS.md #18, one stage earlier. Narrow it for plates whose
    # money shot is a single fast beat.
    global PREFILTER
    PREFILTER = int(spec.get("prefilter_window", PREFILTER))

    fps, times, world, pelvis_h, gaze_src, root_src, incam_src, pelvis_h_incam, aux = load_mp(rpath(spec["landmarks"]))
    # Which estimate of the pelvis arc the airborne (`none`) windows
    # integrate. "global" (default) is GVHMR's gravity-aligned trajectory,
    # the source that under-reported the duel's step-in (0.68 m for 0.92 m);
    # a jump `boost` of 1.35 may be making up the same shortfall. "incam"
    # measures the arc in the camera frame instead, as the lateral root
    # motion already does. An A/B, not a new default (docs/PIPELINE.md).
    pelvis_source = spec.get("pelvis_source", "global")
    if pelvis_source == "incam":
        if np.isnan(pelvis_h_incam).any():
            raise SystemExit("pelvis_source 'incam' needs pelvis_height_incam in landmarks.json — re-run the estimator")
        pelvis_h = pelvis_h_incam
    elif pelvis_source != "global":
        raise SystemExit(f"unknown pelvis_source {pelvis_source!r} (expected 'global' or 'incam')")
    dst_fps = float(spec.get("dst_fps", 30))
    duration = float(times[-1])
    n_dst = int(round(duration * dst_fps)) + 1
    dst_times = np.clip(np.arange(n_dst) / dst_fps, times[0], times[-1])

    mp = {n: np.stack([mp_to_mix(p) for p in world[n]]) for n in world}
    mp = resample(times, mp, dst_times)
    # Torso model. "twist" (default): pelvis from the hip line, chest from
    # the shoulder/collar line, the twist spread over the spine, and — when
    # the landmarks carry the estimator's `body` block — a spine that
    # bends, real clavicles, hand roll and toe tips (reconstruct_twist).
    # "rigid": the original single torso frame (reconstruct), for clips
    # signed off before this existed.
    torso = spec.get("torso", "twist")
    if torso not in ("twist", "rigid"):
        raise SystemExit(f"unknown torso {torso!r} (expected 'twist' or 'rigid')")
    body = None
    if torso == "twist" and aux["body"] is not None:
        body = resample(times, {n: np.stack([mp_to_mix(p) for p in arr]) for n, arr in aux["body"].items()},
                        dst_times)
    if len(pelvis_h) >= 7:
        pelvis_h = savgol_filter(pelvis_h, 7, 2, mode="interp")
    pelvis_h = np.interp(dst_times, times, pelvis_h)
    # Real gaze direction (estimator frame) -> Mixamo world direction.
    # Without this the FK apply can only infer the head from the chest,
    # which locks the character's gaze to its torso twist.
    has_gaze = bool(np.abs(gaze_src).sum() > 1e-6)
    gaze_mix = np.zeros((n_dst, 3))
    if has_gaze:
        g = np.stack([np.interp(dst_times, times, gaze_src[:, c]) for c in range(3)], axis=1)
        g = np.stack([np.array([v[0], v[2], -v[1]]) for v in g])   # mp dir -> mixamo dir
        gaze_mix = np.stack([unit(v) for v in g])

    # Scale between the performer and this character (arm length), used
    # by `arm_follow` to place reference-derived targets.
    _ref_arm = float(np.mean([
        np.linalg.norm(world["left_shoulder"][k] - world["left_elbow"][k])
        + np.linalg.norm(world["left_elbow"][k] - world["left_wrist"][k])
        for k in range(0, len(times), 5)]))
    ref_scale = (LEN["l_arm"] + LEN["l_fore"]) / max(_ref_arm, 1e-6)

    # Ground trajectory. A solo clip is retargeted in place (Mixamo
    # convention, and the estimator's landmarks are hip-centred every
    # frame anyway). Two fighters are a different problem: the whole
    # scene IS the distance between them, and this plate closes from
    # 1.95 m to 0.88 m — retargeted in place, every punch would land a
    # metre short of the other character. `root_motion` restores the
    # performer's real travel, scaled by LEG length so a shorter
    # character takes proportionally shorter steps rather than
    # over-striding into its partner.
    root_motion = bool(spec.get("root_motion", False))
    _ref_leg = float(np.mean([
        np.linalg.norm(world["left_hip"][k] - world["left_knee"][k])
        + np.linalg.norm(world["left_knee"][k] - world["left_ankle"][k])
        for k in range(0, len(times), 5)]))
    root_scale = float(spec.get("root_scale", (LEN["l_upleg"] + LEN["l_leg"]) / max(_ref_leg, 1e-6)))
    # TWO estimates of where the performer stood, and they disagree:
    #
    #   "global"  the pelvis in the estimator's gravity-aligned world
    #             frame. Physically consistent, but it is a PREDICTED
    #             trajectory and it drifts — measured on the duel plate it
    #             under-reported a 0.92 m step-in as 0.68 m and left the
    #             performer 0.16 m off his mark at the closing T-pose.
    #   "incam"   the pelvis in camera space, which is tied to what the
    #             camera actually saw. On the same plate it tracked an
    #             independent image-space measurement (hip pixels over
    #             body pixels) to within 2 cm across all 241 frames.
    #
    # So `incam` is the default wherever the estimator emits it. Only its
    # lateral component is trusted: depth is the ill-conditioned axis of a
    # single-camera fit, and a deep crouch pushed it 0.22 m in one frame
    # on this very plate.
    has_incam = bool(np.abs(incam_src).sum() > 1e-6)
    root_source = spec.get("root_source", "incam" if has_incam else "global")
    root = np.zeros((n_dst, 2))
    if root_motion:
        if root_source == "incam":
            if not has_incam:
                raise SystemExit("root_source 'incam' needs incam_root in landmarks.json — re-run the estimator")
            rs = incam_src[:, [0, 2]].copy()          # camera x -> stage X, camera depth -> stage Y
            if not spec.get("root_depth", False):
                rs[:, 1] = 0.0
        else:
            rs = root_src.copy()
        if len(rs) >= 9:
            rs = np.stack([savgol_filter(rs[:, c], 9, 2, mode="interp") for c in range(2)], axis=1)
        root = np.stack([np.interp(dst_times, times, rs[:, c]) for c in range(2)], axis=1) * root_scale
        root -= root[0]  # the clip starts at the character's stage mark
    # Where this character stands. Not an object transform: the FK apply
    # aims bones in armature space, so a translated armature object would
    # silently offset every aim target (docs/RIG.md). The stage offset is
    # baked into the animation instead, carried by Hips like any other
    # translation on a Mixamo rig.
    stage = np.array([float(spec.get("stage_x", 0.0)), float(spec.get("stage_y", 0.0))])

    def src2dest(sf):
        return (sf - 1) * dst_fps / fps + 1

    # Windowed clearance offset (spec "root_offset"): a ramped shift of
    # this character's ground trajectory. It exists because two Mixamo
    # characters are not two humans: their limbs and heads are thicker
    # than the performers', so a strike the video clears by 2 cm goes
    # THROUGH the other character. Reproducing the video exactly is then
    # the wrong answer, and no pose correction fixes it — the clearance
    # has to come from the stage. Ramp it in and out across frames where
    # the character is already travelling, or a planted foot will skate.
    for ro in spec.get("root_offset", []):
        d = np.array([float(ro.get("dx", 0.0)), float(ro.get("dy", 0.0))])
        ramp = ro.get("ramp_src", 8)
        for i in range(n_dst):
            amt = window_amount(i + 1,
                                [ro["src"][0], ro["src"][0] + ramp],
                                [ro["src"][1] - ramp, ro["src"][1]], src2dest)
            if amt > 1e-6:
                root[i] = root[i] + d * amt


    rb = spec["rest_blend_end"]
    rest = rest_pose()

    plant_windows = [(w["src"][0], w["src"][1], w["support"]) for w in spec["plant"]]

    def plant_at(sf):
        for a, b, sup in plant_windows:
            if a <= sf <= b:
                return sup
        return "both"

    recs, extras = [], []
    for i in range(n_dst):
        f = i + 1
        sf = 1 + (f - 1) * fps / dst_fps  # dest -> src frame (float)
        rec = reconstruct_twist(mp, i, body) if torso == "twist" else reconstruct(mp, i)

        # Authored arm overrides (spec): pull a whole arm chain to
        # hip-local targets — for beats where the owner's read of the
        # video beats the estimator (e.g. occluded arm chambered at the
        # chest). Targets are meters in the hip basis, ramped in/out.
        for ov in spec.get("arm_overrides", []):
            amt = window_amount(
                f,
                [ov["src"][0], ov["src"][0] + ov.get("ramp_src", 6)],
                [ov["src"][1] - ov.get("ramp_src", 6), ov["src"][1]],
                src2dest,
            )
            if amt <= 1e-4:
                continue
            s = "l" if ov["side"] == "left" else "r"
            bx, by, bz = rec["basis_x"], rec["basis_y"], rec["basis_z"]
            hips_p = np.asarray(rec["hips"], float)

            def hip_local(v3):
                return hips_p + bx * v3[0] + by * v3[1] + bz * v3[2]

            sock = np.asarray(rec[f"{s}_arm"], float)
            e_t = sock + unit(hip_local(ov["elbow_local"]) - sock) * LEN[f"{s}_arm"]
            w_t = e_t + unit(hip_local(ov["wrist_local"]) - e_t) * LEN[f"{s}_fore"]
            e = lerp(np.asarray(rec[f"{s}_elbow"], float), e_t, amt)
            w = lerp(np.asarray(rec[f"{s}_wrist"], float), w_t, amt)
            e = sock + unit(e - sock) * LEN[f"{s}_arm"]
            w = e + unit(w - e) * LEN[f"{s}_fore"]
            rec[f"{s}_elbow"], rec[f"{s}_wrist"] = e, w
            rec[f"{s}_hand"] = w + unit(w - e) * LEN[f"{s}_hand"]

        # Windowed reference following (spec "arm_follow"): place the
        # wrist where the PERFORMER has it — their wrist offset from the
        # shoulder mid, scaled to this character — and re-solve the chain
        # with exact bone lengths. The estimator preserves directions but
        # proportion differences accumulate into the pose; for a beat the
        # owner is judging frame by frame, following the reference
        # geometry beats any hand-tuned rotation.
        for af in spec.get("arm_follow", []):
            amt = window_amount(
                f,
                [af["src"][0], af["src"][0] + af.get("ramp_src", 5)],
                [af["src"][1] - af.get("ramp_src", 5), af["src"][1]],
                src2dest,
            )
            if amt <= 1e-4:
                continue
            amt *= float(af.get("amount", 1.0))
            drop = float(af.get("drop_m", 0.0))     # extra lowering, e.g. for a head that sits lower
            # Anchor: "shoulders" (the default, as documented) keeps the
            # anatomical shoulder-relative geometry. "head" matches hand
            # height relative to the face instead, but pairs the performer's
            # NOSE with a joint near the character's skull base
            # (docs/PITFALLS.md #29) and can drive the hands into the torso
            # when proportions differ.
            if af.get("anchor", "shoulders") == "head":
                ref_sh = mp["nose"][i]
                char_sh = np.asarray(rec["head"], float)
            else:
                ref_sh = 0.5 * (mp["left_shoulder"][i] + mp["right_shoulder"][i])
                char_sh = 0.5 * (np.asarray(rec["l_arm"], float) + np.asarray(rec["r_arm"], float))
            sides = ("l", "r") if af.get("side", "both") == "both" else (af["side"][0],)
            for s in sides:
                name = "left_wrist" if s == "l" else "right_wrist"
                target = char_sh + (mp[name][i] - ref_sh) * ref_scale
                target = target - np.asarray(rec["basis_z"], float) * drop
                sock = np.asarray(rec[f"{s}_arm"], float)
                l1, l2 = LEN[f"{s}_arm"], LEN[f"{s}_fore"]
                elbow, wrist = two_bone(sock, target, l1, l2, np.asarray(rec[f"{s}_elbow"], float))
                hand = wrist + unit(wrist - elbow) * LEN[f"{s}_hand"]
                rec[f"{s}_elbow"] = lerp(np.asarray(rec[f"{s}_elbow"], float), elbow, amt)
                rec[f"{s}_wrist"] = lerp(np.asarray(rec[f"{s}_wrist"], float), wrist, amt)
                rec[f"{s}_hand"] = lerp(np.asarray(rec[f"{s}_hand"], float), hand, amt)
                # keep exact bone lengths after the blend
                e = sock + unit(rec[f"{s}_elbow"] - sock) * l1
                w_ = e + unit(rec[f"{s}_wrist"] - e) * l2
                rec[f"{s}_elbow"], rec[f"{s}_wrist"] = e, w_
                rec[f"{s}_hand"] = w_ + unit(w_ - e) * LEN[f"{s}_hand"]

        # Windowed arm-chain rotation (spec "arm_pose"): rotate a whole
        # arm rigidly about its shoulder socket. `pitch_deg` lowers (+)
        # or raises (-) the hand along an arc; `yaw_deg` swings it toward
        # the midline (+) or outward. A rigid rotation preserves elbow
        # bend, wrist alignment and the distance between the two hands —
        # translating the joints instead and re-normalizing bone lengths
        # distorts the chain, drags the hands together and folds the
        # wrists, which is exactly what it did at 20 cm of offset.
        for ap in spec.get("arm_pose", []):
            amt = window_amount(
                f,
                [ap["src"][0], ap["src"][0] + ap.get("ramp_src", 6)],
                [ap["src"][1] - ap.get("ramp_src", 6), ap["src"][1]],
                src2dest,
            )
            if amt <= 1e-4:
                continue
            pitch = np.radians(float(ap.get("pitch_deg", 0.0))) * amt
            yaw = np.radians(float(ap.get("yaw_deg", 0.0))) * amt
            drop = float(ap.get("drop_m", 0.0)) * amt
            widen = float(ap.get("widen_m", 0.0)) * amt
            if abs(pitch) < 1e-5 and abs(yaw) < 1e-5 and abs(drop) < 1e-4 and abs(widen) < 1e-4:
                continue
            lat = np.asarray(rec["basis_x"], float)     # character left
            up = np.asarray(rec["basis_z"], float)
            sides = ("l", "r") if ap.get("side", "both") == "both" else (ap["side"][0],)
            for s in sides:
                sock = np.asarray(rec[f"{s}_arm"], float)
                sign = 1.0 if s == "l" else -1.0        # mirror the lateral sense per side
                wrist0 = np.asarray(rec[f"{s}_wrist"], float)

                # `drop_m` / `widen_m` are targets in METERS — the angle
                # that achieves them depends on where the arm points, so
                # solve it per frame instead of guessing a fixed angle.
                def solve(axis, want, comp):
                    lo, hi = 0.0, np.radians(75.0)
                    base = comp(wrist0 - sock)
                    if abs(want) < 1e-4:
                        return 0.0
                    s_dir = 1.0
                    if comp(rodrigues(wrist0 - sock, axis, 0.05)) - base > 0:
                        s_dir = -1.0 if want < 0 else 1.0
                    else:
                        s_dir = 1.0 if want < 0 else -1.0
                    for _ in range(24):
                        mid = 0.5 * (lo + hi)
                        got = comp(rodrigues(wrist0 - sock, axis, mid * s_dir)) - base
                        if abs(got) < abs(want):
                            lo = mid
                        else:
                            hi = mid
                    return 0.5 * (lo + hi) * s_dir

                # Signed targets: a negative drop_m raises the hands and a
                # negative widen_m brings them together.
                a_pitch = pitch
                if abs(drop) > 1e-4:
                    a_pitch += solve(lat, -drop, lambda v: float(np.dot(v, up)))
                a_yaw = yaw * sign
                if abs(widen) > 1e-4:
                    # widen_m is the change in the DISTANCE BETWEEN the hands;
                    # each arm therefore moves half of it.
                    a_yaw += solve(up, widen * 0.5 * sign, lambda v: float(np.dot(v, lat)))
                for k in (f"{s}_elbow", f"{s}_wrist", f"{s}_hand"):
                    v = np.asarray(rec[k], float) - sock
                    if abs(a_pitch) > 1e-5:
                        v = rodrigues(v, lat, a_pitch)
                    if abs(a_yaw) > 1e-5:
                        v = rodrigues(v, up, a_yaw)
                    rec[k] = sock + v
                if f"{s}_hand_lat" in rec:        # the hand's roll turns with the arm
                    d = np.asarray(rec[f"{s}_hand_lat"], float)
                    if abs(a_pitch) > 1e-5:
                        d = rodrigues(d, lat, a_pitch)
                    if abs(a_yaw) > 1e-5:
                        d = rodrigues(d, up, a_yaw)
                    rec[f"{s}_hand_lat"] = d

        # Windowed leg-chain rotation (spec "leg_pose"): rotate a whole
        # leg rigidly about its hip socket so the foot rises (or drops) by
        # a measured amount. The leg analogue of `arm_pose`, added for the
        # same reason (docs/PITFALLS.md #20) — rotating a chain cannot
        # distort it, translating its joints and re-normalising lengths
        # does.
        #
        # What it is for: the estimator's gravity-aligned pose and its
        # in-camera pose disagree about a fast limb at its peak, and the
        # retarget follows the gravity-aligned one (it has to — that is
        # what makes feet plant). On a head-high roundhouse that showed up
        # as a shin ~9 deg low, dropping the foot 0.15 m and putting it
        # through the other fighter instead of over him.
        for lp in spec.get("leg_pose", []):
            amt = window_amount(
                f,
                [lp["src"][0], lp["src"][0] + lp.get("ramp_src", 4)],
                [lp["src"][1] - lp.get("ramp_src", 4), lp["src"][1]],
                src2dest,
            )
            if amt <= 1e-4:
                continue
            lift = float(lp.get("lift_m", 0.0)) * amt
            pitch = np.radians(float(lp.get("pitch_deg", 0.0))) * amt
            if abs(lift) < 1e-4 and abs(pitch) < 1e-5:
                continue
            sides = ("l", "r") if lp.get("side", "both") == "both" else (lp["side"][0],)
            for s in sides:
                sock = np.asarray(rec[f"{s}_upleg"], float)
                v = np.asarray(rec[f"{s}_ankle"], float) - sock
                r = float(np.linalg.norm(v))
                if r < 1e-4:
                    continue
                # Raise the foot VERTICALLY whichever plane the leg swings
                # in: the axis is the horizontal one perpendicular to the
                # leg itself, not the character's lateral axis. A roundhouse
                # and a front kick travel in different planes and a fixed
                # axis would skew one of them sideways.
                axis = np.cross(v, np.array([0.0, 0.0, 1.0]))
                if np.linalg.norm(axis) < 1e-4:
                    continue
                axis = unit(axis)
                ang = pitch
                if abs(lift) > 1e-4:
                    want = float(np.clip(v[2] + lift, -0.98 * r, 0.98 * r))
                    lo, hi = np.radians(-70.0), np.radians(70.0)
                    for _ in range(28):    # monotone in this range
                        mid = 0.5 * (lo + hi)
                        if rodrigues(v, axis, mid)[2] < want:
                            lo = mid
                        else:
                            hi = mid
                    ang += 0.5 * (lo + hi)
                for k in (f"{s}_knee", f"{s}_ankle", f"{s}_foot", f"{s}_toe"):
                    rec[k] = sock + rodrigues(np.asarray(rec[k], float) - sock, axis, ang)

        # Windowed reach scaling (spec "reach"): push the hand further
        # from the shoulder socket along its own direction and re-place
        # the chain with exact bone lengths (current elbow as pole).
        # Estimators + smoothing compress strike extension; this restores
        # it without touching direction or timing.
        for rc in spec.get("reach", []):
            amt = window_amount(
                f,
                [rc["src"][0], rc["src"][0] + rc.get("ramp_src", 4)],
                [rc["src"][1] - rc.get("ramp_src", 4), rc["src"][1]],
                src2dest,
            )
            if amt <= 1e-4:
                continue
            factor = 1.0 + (float(rc.get("factor", 1.15)) - 1.0) * amt
            max_frac = float(rc.get("max_fraction", 0.97))
            sides = ("l", "r") if rc.get("side", "both") == "both" else (rc["side"][0],)
            for s in sides:
                sock = np.asarray(rec[f"{s}_arm"], float)
                wrist = np.asarray(rec[f"{s}_wrist"], float)
                l1, l2 = LEN[f"{s}_arm"], LEN[f"{s}_fore"]
                d = np.linalg.norm(wrist - sock)
                if d < 1e-4:
                    continue
                # Only extend what is already extending (a guard hand
                # near the chin must not be shoved outward).
                if d < float(rc.get("min_extension", 0.30)) * (l1 + l2):
                    continue
                new_d = min(d * factor, (l1 + l2) * max_frac)
                tgt = sock + unit(wrist - sock) * new_d
                elbow, new_wrist = two_bone(sock, tgt, l1, l2, np.asarray(rec[f"{s}_elbow"], float))
                rec[f"{s}_elbow"], rec[f"{s}_wrist"] = elbow, new_wrist
                # An extending strike puts the hand on the forearm line;
                # blending the estimator's (now stale) hand direction here
                # folds the fist back and cancels the added reach.
                rec[f"{s}_hand"] = new_wrist + unit(new_wrist - elbow) * LEN[f"{s}_hand"]

        head_pitch_deg = 0.0
        gaze_amount = float(spec.get("gaze_follow", 1.0))
        head_level = 0.0
        head_level_target = 0.0
        # Windowed gaze correction (spec "head_look"): blend the head's
        # horizontal direction toward character-forward, and/or pitch the
        # skull axis (`pitch_deg`, + = look up, − = look down). For phases
        # where the estimator's head heading or lean drifts.
        for hl in spec.get("head_look", []):
            win = window_amount(
                f,
                [hl["src"][0], hl["src"][0] + hl.get("ramp_src", 6)],
                [hl["src"][1] - hl.get("ramp_src", 6), hl["src"][1]],
                src2dest,
            )
            if win <= 1e-4:
                continue
            # `amount` weights the horizontal blend only; `pitch_deg` is
            # an absolute angle gated by the same window.
            amt = win * float(hl.get("amount", 1.0))
            neck = np.asarray(rec["neck"], float)
            head = np.asarray(rec["head"], float)
            v = head - neck
            if hl.get("amount") is not None:
                flat = np.array([v[0], v[1], 0.0])
                mag = max(np.linalg.norm(flat), 0.02)
                fwd = -np.asarray(rec["basis_y"], float)
                fwd = unit(np.array([fwd[0], fwd[1], 0.0])) * mag
                new_flat = lerp(flat, fwd, amt)
                v = np.array([new_flat[0], new_flat[1], v[2]])
            if hl.get("pitch_deg"):
                # Rotate the neck->head axis about the character's lateral
                # axis. Positive tips the gaze up, negative levels a
                # backward-leaning skull. Recorded per frame as well: the
                # FK apply rebuilds its head aim from the torso up-axis and
                # would otherwise discard this.
                ang = np.radians(float(hl["pitch_deg"])) * win
                v = rodrigues(v, np.asarray(rec["basis_x"], float), ang)
                head_pitch_deg += float(hl["pitch_deg"]) * win
            if hl.get("follow_gaze") is not None:
                gaze_amount = float(hl["follow_gaze"]) * win + gaze_amount * (1.0 - win)
            if hl.get("level_face"):
                # Closed-loop: the apply measures the resulting face
                # elevation and levels it to `level_target_deg`.
                head_level = max(head_level, float(hl["level_face"]) * win)
                head_level_target = float(hl.get("level_target_deg", 0.0))
            rec["head"] = neck + v
        if "neck_up" in rec:
            # The neck aims along the lifted neck->head direction, so a
            # head_look correction to the head carries through.
            rec["neck_up"] = unit(np.asarray(rec["head"], float) - np.asarray(rec["neck"], float))

        rest_amt = smoother((sf - rb["start_src"]) / max(1.0, rb["full_src"] - rb["start_src"]))
        rbs = spec.get("rest_blend_start")
        if rbs:
            start_amt = 1.0 - smoother(
                (sf - rbs["full_src"]) / max(1.0, rbs["release_src"] - rbs["full_src"])
            )
            rest_amt = max(rest_amt, start_amt)
        if rest_amt > 0.0:
            rec = blend_pose(rec, rest, rest_amt)
            for k in rec:
                if is_dir(k):
                    rec[k] = unit(np.asarray(rec[k], float))

        fist = window_amount(f, spec["fists"]["rise_src"], spec["fists"]["fall_src"], src2dest)
        recs.append(rec)
        extras.append({"frame": f, "t": float(dst_times[i]), "rest": rest_amt, "fist": fist,
                       "plant": plant_at(sf) if rest_amt < 0.65 else "both",
                       "pelvis_height": float(pelvis_h[i]),
                       "gaze": gaze_mix[i] if has_gaze else None,
                       "gaze_amount": (gaze_amount if has_gaze else 0.0),
                       "head_pitch_deg": head_pitch_deg,
                       "head_level": head_level,
                       "head_level_target_deg": head_level_target})

    # Optional extra temporal smoothing on selected joints in selected
    # windows (spec "smooth") — for fast flurries the raw estimator
    # tracks can look staccato; a windowed Savitzky-Golay pass with
    # smooth edge blending calms them without killing the snap.
    for sm in spec.get("smooth", []):
        a = max(1, int(round(src2dest(sm["src"][0]))))
        b = min(n_dst, int(round(src2dest(sm["src"][1]))))
        w = int(sm.get("window", 9))
        w = min(w if w % 2 == 1 else w - 1, (b - a + 1) | 1)
        if w < 5 or b - a < 4:
            continue
        keys = sm.get("joints", ["l_elbow", "l_wrist", "l_hand", "r_elbow", "r_wrist", "r_hand"])
        ramp = max(2, w // 2)
        for k in keys:
            series = np.array([np.asarray(recs[f - 1][k], float) for f in range(a, b + 1)])
            smoothed = np.stack(
                [savgol_filter(series[:, c], window_length=w, polyorder=2, mode="interp")
                 for c in range(3)], axis=1)
            for j, f in enumerate(range(a, b + 1)):
                t = min(1.0, min(j, (b - a) - j) / float(ramp))
                blend = smoother(t)
                recs[f - 1][k] = series[j] * (1.0 - blend) + smoothed[j] * blend

    # Foot pinning (see module docstring).
    #
    # The skate this corrects is an ARTEFACT of hip-centred landmarks:
    # with the body's travel discarded, a planted foot appears to slide
    # whenever the pelvis moves over it. When `root_motion` restores that
    # travel the artefact is gone at the source, and re-pinning on top
    # would fight the real trajectory — cancelling the step inside every
    # stance and then snapping it back as the window released.
    #
    # "contacts" (default) pins every planted foot, in `both` windows too,
    # from the spec's support schedule, the performer's foot heights and
    # the estimator's per-foot static confidence (contact_weights). The
    # offset it builds is the pelvis travelling over the feet; it is carried
    # through flight and eased back to the stage mark as the closing rest
    # blend comes in, so the clip still ends where it started.
    # "single" is the original rule: pin the support ankle in single-support
    # windows only, release over 10 frames — which left planted feet
    # sliding in every `both` window (docs/PITFALLS.md #40).
    pin_mode = spec.get("pin", "contacts")
    if pin_mode not in ("contacts", "single"):
        raise SystemExit(f"unknown pin mode {pin_mode!r} (expected 'contacts' or 'single')")
    RELEASE = 10
    pos_keys = [k for k in recs[0] if not is_dir(k)]
    if pin_mode == "contacts" and not root_motion:
        h_ankle = {s: pelvis_h + (mp[f"{side}_ankle"][:, 2] - HIP_Z) for s, side in (("l", "left"), ("r", "right"))}
        h_ball = {s: pelvis_h + (mp[f"{side}_foot_index"][:, 2] - HIP_Z) for s, side in (("l", "left"), ("r", "right"))}
        st = None
        if aux["static"] is not None:
            st = {}
            for s, side in (("l", "left"), ("r", "right")):
                st[s, "ankle"] = np.interp(dst_times, times, aux["static"][f"{side}_ankle"])
                st[s, "ball"] = np.interp(dst_times, times, aux["static"][f"{side}_foot_index"])
        W = contact_weights([ex["plant"] for ex in extras], h_ankle, h_ball, st)
        offs = pin_offsets(recs, W)
        for i, rec in enumerate(recs):
            d = offs[i] * (1.0 - float(extras[i]["rest"]))
            for k in pos_keys:
                rec[k] = np.asarray(rec[k], float)
                rec[k][:2] += d
    for a_src, b_src, sup in plant_windows:
        if pin_mode != "single" or sup not in ("left", "right") or root_motion:
            continue
        a = max(1, int(round(src2dest(a_src))))
        b = min(n_dst, int(round(src2dest(b_src))))
        ankle = "l_ankle" if sup == "left" else "r_ankle"
        anchor = np.asarray(recs[a - 1][ankle], float)[:2].copy()
        last_delta = np.zeros(2)
        for f in range(a, b + 1):
            delta = anchor - np.asarray(recs[f - 1][ankle], float)[:2]
            for k in pos_keys:
                recs[f - 1][k] = np.asarray(recs[f - 1][k], float)
                recs[f - 1][k][:2] += delta
            last_delta = delta
        for j, f in enumerate(range(b + 1, min(n_dst, b + RELEASE) + 1)):
            fade = last_delta * (1.0 - smoother((j + 1) / float(RELEASE)))
            for k in pos_keys:
                recs[f - 1][k] = np.asarray(recs[f - 1][k], float)
                recs[f - 1][k][:2] += fade

    # Stage placement + ground trajectory, applied last so the rest blend
    # still resolves to the character's exact rest pose (blending toward
    # REST after this would drag the character back to the origin).
    if root_motion or stage.any():
        for i, rec in enumerate(recs):
            delta = stage + root[i]
            for k in pos_keys:
                rec[k] = np.asarray(rec[k], float)
                rec[k][:2] += delta

    joints = []
    for rec, ex in zip(recs, extras):
        out = {k: [round(float(v), 5) for v in np.asarray(val)] for k, val in rec.items()}
        out["frame"] = ex["frame"]
        out["t"] = ex["t"]
        out["rest_amount"] = round(float(ex["rest"]), 4)
        out["fist_amount"] = round(float(ex["fist"]), 4)
        out["plant"] = ex["plant"]
        out["pelvis_height"] = round(float(ex["pelvis_height"]), 5)
        if ex["gaze"] is not None:
            out["gaze"] = [round(float(x), 5) for x in ex["gaze"]]
            out["gaze_amount"] = round(float(ex["gaze_amount"]), 4)
        out["head_pitch_deg"] = round(float(ex["head_pitch_deg"]), 3)
        out["head_level"] = round(float(ex["head_level"]), 4)
        out["head_level_target_deg"] = round(float(ex["head_level_target_deg"]), 2)
        joints.append(out)

    payload = {
        "src_fps": fps,
        "dst_fps": dst_fps,
        "duration": duration,
        "frame_count": n_dst,
        "estimator": spec.get("estimator", "unknown"),
        "action_spec": spec["name"],
        "frames": joints,
    }
    out_path = rpath(spec["joints_out"])
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    print("wrote", out_path, "frames", n_dst)
    print(f"profile: {used_profile.name if used_profile else 'built-in Y Bot'} "
          f"(hip {HIP_Z:.3f} m, arm {LEN['l_arm'] + LEN['l_fore']:.3f} m)")
    if root_motion or stage.any():
        print(f"stage x={stage[0]:+.3f} y={stage[1]:+.3f} | root motion "
              f"{'ON (' + root_source + ')' if root_motion else 'off'} scale {root_scale:.3f} "
              f"travel {np.linalg.norm(root[-1] - root[0]):.3f} m, "
              f"max {np.abs(root).max():.3f} m")

    for sfq in spec.get("qa_src_frames", []):
        f = int(round(src2dest(sfq)))
        f = max(1, min(n_dst, f))
        j = joints[f - 1]
        bx = np.asarray(j["basis_x"], float)
        yaw = np.degrees(np.arctan2(-bx[1], bx[0]))
        print(
            f"src{sfq:3d}/dest{f:3d} rest={j['rest_amount']:.2f} fist={j['fist_amount']:.2f} "
            f"plant={j['plant']:5s} yaw={yaw:+4.0f} "
            f"lw={j['l_wrist']} rw={j['r_wrist']} ra={j['r_ankle']}"
        )


if __name__ == "__main__":
    main()
