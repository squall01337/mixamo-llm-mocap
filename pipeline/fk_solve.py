"""Numpy FK solver: lifted joints -> Mixamo pose keys, without Blender.

The per-frame solve that apply_mixamo_fk used to run inside Blender — aim
every bone at its lifted joint, plant the hips, flatten the feet, turn the
face to the estimator's gaze, curl the fists — computed on the rig's rest
matrices instead of a live depsgraph. Wherever a convention decides a number,
Blender's is mirrored: pose-matrix composition, pose -> basis conversion,
shortest-arc rotations, canonical quaternions, the 15-step hip search.
Matched against the Blender solver it replaces (run_legacy), bone by bone and
frame by frame, on a 65-bone Mixamo-shaped rig.

Two ways in:

  inside Blender   apply_mixamo_fk.run() reads the skeleton from the live
                   armature, solves here, and writes the action in bulk —
                   seconds instead of minutes, with no frozen UI.
  outside Blender  python pipeline\\fk_solve.py --spec action_specs\\<motion>.json
                   reads the skeleton from the rig profile and writes the
                   clip's curves.json (the file `run_in_blender.py curves`
                   dumps) plus keys.json, so qa_clip.py and compare_*.py run
                   with no Blender at all. A pass is lift + this: seconds.

Beyond the legacy solve, three things only this solver does:

  torso twist      when the lift provides separate pelvis and chest frames,
                   the Hips follow the pelvis and the twist to the chest is
                   spread over Spine / Spine1 / Spine2 (see docs/PIPELINE.md);
  forearm roll     when the lift provides the hand's lateral direction (from
                   the estimator's real knuckles), half the roll goes to the
                   forearm and the rest to the hand;
  quaternion       every bone's keys are kept in one hemisphere, so a 360
  continuity       degree spin does not interpolate through a flipped key.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FLT_EPSILON = 1.1920929e-07
IDQ = np.array([1.0, 0.0, 0.0, 0.0])

# Y Bot measurements, overridden by the rig profile (see use_profile()).
GROUND_Z = 0.105          # rest ankle height (flat-foot contact)
HIP_HEIGHT = 0.99792      # rest hip height
BALL_Z = 0.03284          # rest ball (ToeBase head) height
# The flat-foot targets the legacy solver used for Y Bot (ball 0.034, toe
# aim 0.028), kept as offsets from the character's own ball height so a
# character with thicker soles is not pushed into the floor.
BALL_TARGET_DZ = 0.034 - 0.03284
TOE_TARGET_DZ = 0.028 - 0.03284

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
SPINE = ("mixamorig:Spine", "mixamorig:Spine1", "mixamorig:Spine2")
FOREARM_ROLL_SHARE = 0.5   # of the hand's roll, carried by the forearm (no twist bones on Mixamo)


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


def keyed(name: str) -> bool:
    """Tip bones carry no skin weights and are never keyed (docs/RIG.md)."""
    return not (name.endswith("_End") or name.endswith("4"))


# ---------------------------------------------------------------------------
# Blender-parity math. Quaternions are (w, x, y, z); matrices act on column
# vectors, as mathutils does.

def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros(3)


def q_normalize(q):
    n = float(np.linalg.norm(q))
    return q / n if n > 0.0 else IDQ.copy()


def q_to_mat3(q):
    w, x, y, z = q_normalize(np.asarray(q, float))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def mat3_to_quat(m):
    """Blender's mat3_normalized_to_quat: the canonical quaternion, w >= 0."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = 2.0 * math.sqrt(tr + 1.0)
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q = q_normalize(q)
    return -q if q[0] < 0.0 else q


def axis_angle_to_quat(axis, angle):
    a = unit(np.asarray(axis, float))
    if not a.any():
        return IDQ.copy()
    h = 0.5 * angle
    return np.array([math.cos(h), *(a * math.sin(h))])


def _angle_normalized(a, b):
    if float(np.dot(a, b)) >= 0.0:
        return 2.0 * math.asin(min(1.0, float(np.linalg.norm(a - b)) / 2.0))
    return math.pi - 2.0 * math.asin(min(1.0, float(np.linalg.norm(a + b)) / 2.0))


def _ortho(v):
    ax, ay, az = abs(v[0]), abs(v[1]), abs(v[2])
    dom = (0 if ax > az else 2) if ax > ay else (1 if ay > az else 2)
    if dom == 0:
        return np.array([-v[1] - v[2], v[0], v[0]])
    if dom == 1:
        return np.array([v[1], -v[0] - v[2], v[1]])
    return np.array([v[2], v[2], -v[0] - v[1]])


def rotation_difference(a, b):
    """mathutils Vector.rotation_difference: the shortest-arc rotation a -> b."""
    a, b = unit(np.asarray(a, float)), unit(np.asarray(b, float))
    axis = np.cross(a, b)
    n = float(np.linalg.norm(axis))
    if n > FLT_EPSILON:
        return axis_angle_to_quat(axis / n, _angle_normalized(a, b))
    if float(np.dot(a, b)) > 0.0:
        return IDQ.copy()
    return axis_angle_to_quat(_ortho(a), math.pi)


def quat_to_axis_angle(q):
    q = q_normalize(np.asarray(q, float))
    ha = math.acos(max(-1.0, min(1.0, q[0])))
    si = math.sin(ha)
    if abs(si) < 0.0005:
        si = 1.0
    axis = q[1:] / si
    if not axis.any():
        axis = np.array([0.0, 1.0, 0.0])
    return axis, 2.0 * ha


def slerp(q1, q2, t):
    """mathutils Quaternion.slerp (shortest path, linear when nearly equal)."""
    q1, q2 = q_normalize(np.asarray(q1, float)), q_normalize(np.asarray(q2, float))
    cosom = float(np.dot(q1, q2))
    if cosom < 0.0:
        cosom, q1 = -cosom, -q1
    if (1.0 - cosom) > 0.0001:
        omega = math.acos(min(1.0, cosom))
        sinom = math.sin(omega)
        s1, s2 = math.sin((1.0 - t) * omega) / sinom, math.sin(t * omega) / sinom
    else:
        s1, s2 = 1.0 - t, t
    return s1 * q1 + s2 * q2


def mat4(r3, t):
    m = np.eye(4)
    m[:3, :3] = r3
    m[:3, 3] = t
    return m


def rigid_inv(m):
    r = m[:3, :3].T
    return mat4(r, -r @ m[:3, 3])


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Skeleton + pose state.

class Skeleton:
    """A rig's rest pose: bone hierarchy, armature-space rest matrices
    (Blender's Bone.matrix_local) and the armature object's world matrix."""

    def __init__(self, names, parents, rest, world):
        depth = {}

        def d(n):
            if n not in depth:
                depth[n] = 0 if parents[n] is None else d(parents[n]) + 1
            return depth[n]

        order = sorted(names, key=lambda n: (d(n), names.index(n)))
        self.names = order
        self.idx = {n: i for i, n in enumerate(order)}
        self.parent = np.array([-1 if parents[n] is None else self.idx[parents[n]] for n in order])
        self.rest = np.array([rest[n] for n in order], float)
        self.offs = np.array([self.rest[i] if p < 0 else rigid_inv(self.rest[p]) @ self.rest[i]
                              for i, p in enumerate(self.parent)])
        self.world = np.asarray(world, float)
        self.world3 = self.world[:3, :3]
        self.world_inv = np.linalg.inv(self.world)
        self.world3_inv = np.linalg.inv(self.world3)
        kids = [[] for _ in order]
        for i, p in enumerate(self.parent):
            if p >= 0:
                kids[p].append(i)
        self.subtree = []
        for i in range(len(order)):
            stack, out = [i], []
            while stack:
                j = stack.pop()
                out.append(j)
                stack.extend(kids[j])
            self.subtree.append(out)

    @classmethod
    def from_blender(cls, arm):
        bones = arm.data.bones
        return cls([b.name for b in bones],
                   {b.name: (b.parent.name if b.parent else None) for b in bones},
                   {b.name: np.array(b.matrix_local) for b in bones},
                   np.array(arm.matrix_world))

    @classmethod
    def from_profile(cls, profile: dict):
        sk = profile.get("skeleton")
        if not sk:
            raise SystemExit("the rig profile has no `skeleton` — re-run setup_rig.py / setup_duo.py, or "
                             "`python pipeline\\run_in_blender.py skeleton <spec>` to add it")
        bones = sk["bones"]
        return cls([b["name"] for b in bones], {b["name"]: b["parent"] for b in bones},
                   {b["name"]: np.array(b["rest"], float).reshape(4, 4) for b in bones},
                   np.array(sk["world"], float).reshape(4, 4))

    def to_profile(self) -> dict:
        return {"world": [round(float(x), 7) for x in self.world.ravel()],
                "bones": [{"name": n,
                           "parent": (self.names[p] if p >= 0 else None),
                           "rest": [round(float(x), 7) for x in self.rest[i].ravel()]}
                          for i, (n, p) in enumerate(zip(self.names, self.parent))]}


class Pose:
    """Pose-bone state (basis quaternion + location per bone) with lazily
    evaluated armature-space pose matrices, as Blender's pose evaluation
    computes them: pose = parent_pose @ (parent_rest^-1 @ rest) @ basis."""

    def __init__(self, skel: Skeleton):
        self.s = skel
        n = len(skel.names)
        self.q = np.tile(IDQ, (n, 1))
        self.loc = np.zeros((n, 3))
        self._pm = np.zeros((n, 4, 4))
        self._ok = np.zeros(n, bool)

    def i(self, name):
        return self.s.idx[name]

    def reset(self):
        self.q[:] = IDQ
        self.loc[:] = 0.0
        self._ok[:] = False

    def set(self, i, q=None, loc=None):
        if q is not None:
            self.q[i] = q
        if loc is not None:
            self.loc[i] = loc
        self._ok[self.s.subtree[i]] = False

    def pm(self, i):
        if not self._ok[i]:
            basis = mat4(q_to_mat3(self.q[i]), self.loc[i])
            p = self.s.parent[i]
            self._pm[i] = (self.s.offs[i] if p < 0 else self.pm(p) @ self.s.offs[i]) @ basis
            self._ok[i] = True
        return self._pm[i]

    def world_loc(self, i):
        return self.s.world[:3, :3] @ self.pm(i)[:3, 3] + self.s.world[:3, 3]

    def basis_from_matrix(self, i, m):
        """Blender's `pose_bone.matrix = m`: back to the bone's basis."""
        p = self.s.parent[i]
        parent_part = self.s.offs[i] if p < 0 else self.pm(p) @ self.s.offs[i]
        basis = rigid_inv(parent_part) @ m
        r = basis[:3, :3] / np.linalg.norm(basis[:3, :3], axis=0)
        return mat3_to_quat(r), basis[:3, 3].copy()

    def to_arm(self, p_world):
        return self.s.world_inv[:3, :3] @ p_world + self.s.world_inv[:3, 3]

    def dir_to_arm(self, d_world):
        return unit(self.s.world3_inv @ d_world)


# ---------------------------------------------------------------------------
# The solve, step for step as apply_mixamo_fk.run_legacy() does it in Blender.

def aim_bone(P: Pose, i, target_world):
    """Aim bone +Y at a world-metre target, in ARMATURE space."""
    P.set(i, IDQ, np.zeros(3))
    m = P.pm(i)
    head = m[:3, 3].copy()
    d = P.to_arm(np.asarray(target_world, float)) - head
    if np.linalg.norm(d) < 1e-5:
        return
    rot = rotation_difference(m[:3, 1], d)
    q, _ = P.basis_from_matrix(i, mat4(q_to_mat3(rot) @ m[:3, :3], head))
    P.set(i, q, np.zeros(3))


def set_hips(P: Pose, i, bx, by, bz, loc_world):
    P.set(i, IDQ, np.zeros(3))
    up_arm, left_arm = P.dir_to_arm(bz), P.dir_to_arm(bx)
    m = P.pm(i)
    rot_y = rotation_difference(m[:3, 1], up_arm)
    q, _ = P.basis_from_matrix(i, mat4(q_to_mat3(rot_y) @ m[:3, :3], m[:3, 3].copy()))
    P.set(i, q)
    m = P.pm(i)
    x_now = unit(m[:3, 0])
    left_flat = left_arm - up_arm * float(np.dot(left_arm, up_arm))
    if np.linalg.norm(left_flat) > 1e-5:
        axis, ang = quat_to_axis_angle(rotation_difference(x_now, left_flat))
        y = m[:3, 1]
        if float(np.dot(axis, y)) < 0.0:
            ang = -ang
        twist = axis_angle_to_quat(unit(y), ang)
        q, _ = P.basis_from_matrix(i, mat4(q_to_mat3(twist) @ m[:3, :3], m[:3, 3].copy()))
        P.set(i, q)
    P.set(i, loc=np.array([loc_world[0] * 100.0, (loc_world[2] - HIP_HEIGHT) * 100.0, -loc_world[1] * 100.0]))


def plant_hips(P: Pose, hips, hint_y, loc_x, loc_z, foot):
    """Search hips.location.y (cm) until `foot` lands at GROUND_Z — the same
    15-step bisection as the Blender solver, so both land on the same value."""
    lo, hi = hint_y - 55.0, hint_y + 40.0
    best_y, best_err = hint_y, 1e9
    for _ in range(15):
        mid = 0.5 * (lo + hi)
        P.set(hips, loc=np.array([loc_x, mid, loc_z]))
        zf = P.world_loc(foot)[2]
        err = abs(zf - GROUND_Z)
        if err < best_err:
            best_err, best_y = err, mid
        if zf > GROUND_Z:
            hi = mid
        else:
            lo = mid
    P.set(hips, loc=np.array([loc_x, best_y, loc_z]))
    return best_y


def flatten_foot(P: Pose, side: str):
    """Z-only ground snap for a near-ground foot: estimator XZ and the foot's
    own heading are kept, only heights are set."""
    foot, toe = P.i(f"mixamorig:{side}Foot"), P.i(f"mixamorig:{side}ToeBase")
    ankle, ball = P.world_loc(foot), P.world_loc(toe)
    if ankle[2] > 0.20:
        return
    heading = np.array([ball[0] - ankle[0], ball[1] - ankle[1], 0.0])
    if np.linalg.norm(heading) < 0.02:
        y_w = P.s.world3 @ P.pm(foot)[:3, 1]
        heading = np.array([y_w[0], y_w[1], 0.0])
    if np.linalg.norm(heading) < 1e-5:
        return
    heading = unit(heading)
    aim_bone(P, foot, np.array([ball[0], ball[1], BALL_Z + BALL_TARGET_DZ]))
    b2 = P.world_loc(toe)
    aim_bone(P, toe, np.array([b2[0] + heading[0] * 0.06, b2[1] + heading[1] * 0.06, BALL_Z + TOE_TARGET_DZ]))


def apply_fingers(P: Pose, amount: float):
    for name, quat in FIST.items():
        if name in P.s.idx:
            P.set(P.i(name), slerp(IDQ, q_normalize(np.array(quat, float)), amount), np.zeros(3))


def _face_dir(P: Pose, i):
    return unit((P.s.world3 @ P.pm(i)[:3, :3]) @ np.array([0.0, 0.0, 1.0]))


def _rotate_bone(P: Pose, i, axis_world, delta):
    if abs(delta) < 1e-5 or np.linalg.norm(axis_world) < 1e-6:
        return
    axis_arm = P.dir_to_arm(unit(np.asarray(axis_world, float)))
    m = P.pm(i)
    q, _ = P.basis_from_matrix(i, mat4(q_to_mat3(axis_angle_to_quat(axis_arm, delta)) @ m[:3, :3],
                                       m[:3, 3].copy()))
    P.set(i, q, np.zeros(3))


def _solve_axis(P: Pose, i, axis_world, measure, target, amount, periodic=False):
    """Probe-solve-refine one rotation axis (the head's frame is heavily yawed
    during spins, so the sign of a correction cannot be assumed). `periodic`
    measures angles wrapped to +-pi, so a partial follow across +-180 degrees
    turns the short way (the legacy solver turned the long way)."""
    def diff(a, b):
        return wrap_pi(a - b) if periodic else a - b
    e0 = measure()
    want = diff(target, e0) * amount
    if abs(want) < 1e-4:
        return
    probe = math.copysign(0.05, want)
    _rotate_bone(P, i, axis_world, probe)
    e1 = measure()
    slope = diff(e1, e0) / probe
    if abs(slope) < 0.1:
        _rotate_bone(P, i, axis_world, -probe)
        return
    _rotate_bone(P, i, axis_world, (diff(target, e1) * amount) / slope)
    resid = diff(target, measure()) * amount
    if abs(resid) > math.radians(0.5):
        _rotate_bone(P, i, axis_world, resid / slope)


def orient_face(P: Pose, i, gaze, amount, max_offset_deg=75.0, body_forward=None):
    """Point the head's FACE axis (local +Z) along `gaze` (world), the
    head-vs-chest yaw capped at `max_offset_deg`."""
    if amount <= 1e-4 or gaze is None:
        return
    g = np.asarray(gaze, float)
    if np.linalg.norm(g) < 1e-6:
        return
    g = unit(g)
    if body_forward is not None and np.linalg.norm(body_forward) > 1e-6:
        bf, gf = np.array([body_forward[0], body_forward[1], 0.0]), np.array([g[0], g[1], 0.0])
        if np.linalg.norm(bf) > 1e-6 and np.linalg.norm(gf) > 1e-6:
            bf, gf = unit(bf), unit(gf)
            off = math.atan2(bf[0] * gf[1] - bf[1] * gf[0], bf[0] * gf[0] + bf[1] * gf[1])
            lim = math.radians(max_offset_deg)
            if abs(off) > lim:
                g = q_to_mat3(axis_angle_to_quat([0.0, 0.0, 1.0], math.copysign(lim, off) - off)) @ g
    _solve_axis(P, i, np.array([0.0, 0.0, 1.0]),
                lambda: math.atan2(_face_dir(P, i)[0], -_face_dir(P, i)[1]),
                math.atan2(g[0], -g[1]), amount, periodic=True)
    lateral = np.array([-g[1], g[0], 0.0])
    _solve_axis(P, i, lateral, lambda: math.asin(max(-1.0, min(1.0, _face_dir(P, i)[2]))),
                math.asin(max(-1.0, min(1.0, g[2]))), amount)


def _roll_to(P: Pose, i, rest_local, target_world, share=1.0):
    """Twist bone `i` about its own +Y so that the direction it carried at
    rest (`rest_local`, bone-local) lines up with `target_world` around the
    bone axis. Convention-free: the rest direction is measured on the rig,
    never assumed from a bone roll. Returns the full angle it wanted."""
    m = P.pm(i)
    y = unit(m[:3, 1])
    cur = m[:3, :3] @ rest_local
    tgt = P.s.world3_inv @ np.asarray(target_world, float)
    cur, tgt = cur - y * float(np.dot(cur, y)), tgt - y * float(np.dot(tgt, y))
    if np.linalg.norm(cur) < 1e-6 or np.linalg.norm(tgt) < 1e-6:
        return 0.0
    cur, tgt = unit(cur), unit(tgt)
    ang = math.atan2(float(np.dot(np.cross(cur, tgt), y)), float(np.dot(cur, tgt)))
    if abs(ang * share) > 1e-6:
        q, _ = P.basis_from_matrix(i, mat4(q_to_mat3(axis_angle_to_quat(y, ang * share)) @ m[:3, :3],
                                           m[:3, 3].copy()))
        P.set(i, q, np.zeros(3))
    return ang


def rest_axes(skel: Skeleton) -> dict:
    """What each bone that can be twisted carries at rest, in its own frame:
    the character's LEFT for the spine (the chest's lateral axis), and the
    hand's index-to-pinky direction (character BACK in a palms-down T-pose)
    for the forearms and hands."""
    P = Pose(skel)
    left_arm = P.dir_to_arm(np.array([1.0, 0.0, 0.0]))
    back_arm = P.dir_to_arm(np.array([0.0, 1.0, 0.0]))
    out = {}
    for n in SPINE:
        if n in skel.idx:
            out[n] = P.pm(P.i(n))[:3, :3].T @ left_arm
    for side in ("Left", "Right"):
        for n in (f"mixamorig:{side}ForeArm", f"mixamorig:{side}Hand"):
            if n in skel.idx:
                out[n] = P.pm(P.i(n))[:3, :3].T @ back_arm
    return out


def solve(frames: list, spec: dict, skel: Skeleton) -> dict:
    """Solve every frame. Returns keys (local quaternions per keyed bone,
    hips location in pose centimetres), world positions of every bone,
    face and body directions, and the pose channels per frame."""
    src_fps = float(spec.get("src_fps", 24))
    dst_fps = float(spec.get("dst_fps", 30))
    boost_by_frame = {}
    for wnd in spec.get("plant", []):
        if wnd.get("support") == "none" and wnd.get("boost"):
            a = int(round((wnd["src"][0] - 1) * dst_fps / src_fps + 1))
            b = int(round((wnd["src"][1] - 1) * dst_fps / src_fps + 1))
            for f_ in range(a, b + 1):
                boost_by_frame[f_] = float(wnd["boost"])
    max_off = float(spec.get("gaze_max_offset_deg", 75.0))
    roll_share = float(spec.get("forearm_roll_share", FOREARM_ROLL_SHARE))

    P = Pose(skel)
    I = skel.idx
    hips, head_i = I["mixamorig:Hips"], I["mixamorig:Head"]
    rest_ax = rest_axes(skel)
    chest_i = I.get("mixamorig:Spine2", hips)
    chest_fwd_local = P.pm(chest_i)[:3, :3].T @ P.dir_to_arm(np.array([0.0, -1.0, 0.0]))
    n_b, n_f = len(skel.names), len(frames)
    Q = np.zeros((n_f, n_b, 4))
    LOC = np.zeros((n_f, n_b, 3))
    WORLD = np.zeros((n_f, n_b, 3))
    FACE = np.zeros((n_f, 3))
    FWD = np.zeros((n_f, 3))
    CHEST = np.zeros((n_f, 3))

    def v(k, fr):
        return np.asarray(fr[k], float)

    prev_hip_y, prev_ph = 0.0, None
    for t, fr in enumerate(frames):
        f = int(fr["frame"])
        rest = float(fr.get("rest_amount", 0.0))
        fist = float(fr.get("fist_amount", 0.0))
        plant = fr.get("plant", "both")
        ph = float(fr.get("pelvis_height", 0.0))
        if prev_ph is None:
            prev_ph = ph
        P.reset()
        # Twist-aware lifts carry the pelvis frame separately from the torso
        # frame the correctors use; the Hips follow the pelvis.
        pk = "pelvis_" if "pelvis_x" in fr else "basis_"
        set_hips(P, hips, v(pk + "x", fr), v(pk + "y", fr), v(pk + "z", fr), v("hips", fr))
        hips_w = v("hips", fr)
        loc_x, loc_z = hips_w[0] * 100.0, -hips_w[1] * 100.0
        hint = (hips_w[2] - HIP_HEIGHT) * 100.0
        # With a bending spine (`neck_up`, from the estimator's own neck
        # joints) the neck and head aim along that direction from wherever
        # the spine put them; otherwise along the torso's up, from the
        # lifted neck point, with the lifted head's small horizontal offset.
        along_neck = "neck_up" in fr
        up = v("neck_up", fr) if along_neck else v("basis_z", fr)
        neck_p = v("neck", fr)
        head_flat = v("head", fr) - neck_p
        head_flat[2] = 0.0
        if along_neck:
            head_flat[:] = 0.0
        pitch = math.radians(float(fr.get("head_pitch_deg", 0.0) or 0.0))
        lateral = v("basis_x", fr)
        for bone, key in AIM:
            if bone not in I:
                continue
            if bone == "mixamorig:Neck":
                aim = up * 0.11 + head_flat * 0.25
                if pitch:
                    aim = q_to_mat3(axis_angle_to_quat(lateral, pitch * 0.35)) @ aim
                aim_bone(P, I[bone], (P.world_loc(I[bone]) if along_neck else neck_p) + aim)
            elif bone == "mixamorig:Head":
                aim = up * 0.28 + head_flat * 0.15
                if pitch:
                    aim = q_to_mat3(axis_angle_to_quat(lateral, pitch)) @ aim
                aim_bone(P, I[bone], (P.world_loc(I[bone]) if along_neck else neck_p) + aim)
            else:
                aim_bone(P, I[bone], v(key, fr))
            if bone == "mixamorig:Spine2" and "chest_x" in fr:
                # Spread the pelvis-to-chest twist over the three spine bones.
                # Twisting a bone about its own axis leaves its aim (and the
                # collinear bones above it) where they were.
                total = _roll_to(P, I["mixamorig:Spine2"], rest_ax["mixamorig:Spine2"], v("chest_x", fr), 0.0)
                for sb in SPINE:
                    _rotate_bone(P, I[sb], P.s.world3 @ P.pm(I[sb])[:3, 1], total / 3.0)
        for side, s in (("Left", "l"), ("Right", "r")):
            lat = fr.get(f"{s}_hand_lat")
            fa, hd = f"mixamorig:{side}ForeArm", f"mixamorig:{side}Hand"
            if lat is None or fa not in I or hd not in I:
                continue
            # The hand's roll, from the estimator's real knuckles. Mixamo has
            # no twist bones: part of it goes to the forearm (so the forearm
            # skin turns with the hand), then the hand is re-aimed and takes
            # the rest.
            want = _roll_to(P, I[hd], rest_ax[hd], np.asarray(lat, float), 0.0)
            _rotate_bone(P, I[fa], P.s.world3 @ P.pm(I[fa])[:3, 1], want * roll_share)
            aim_bone(P, I[hd], v(f"{s}_hand", fr))
            _roll_to(P, I[hd], rest_ax[hd], np.asarray(lat, float), 1.0)
        fwd_k = "chest_y" if "chest_y" in fr else "basis_y"
        orient_face(P, head_i, fr.get("gaze"), float(fr.get("gaze_amount", 0.0) or 0.0), max_off,
                    -v(fwd_k, fr))
        apply_fingers(P, fist)

        if rest >= 0.65:
            P.set(hips, loc=np.array([loc_x, 0.0, loc_z]))
        elif plant == "none":
            P.set(hips, loc=np.array([loc_x, prev_hip_y + (ph - prev_ph) * 100.0 * boost_by_frame.get(f, 1.0),
                                      loc_z]))
        elif plant in ("left", "right"):
            side = "Left" if plant == "left" else "Right"
            plant_hips(P, hips, hint, loc_x, loc_z, I[f"mixamorig:{side}Foot"])
            flatten_foot(P, side)
        else:
            la = P.world_loc(I["mixamorig:LeftFoot"])[2]
            ra = P.world_loc(I["mixamorig:RightFoot"])[2]
            lower, other = ("Left", "Right") if la <= ra else ("Right", "Left")
            plant_hips(P, hips, hint, loc_x, loc_z, I[f"mixamorig:{lower}Foot"])
            flatten_foot(P, lower)
            flatten_foot(P, other)
        prev_hip_y, prev_ph = float(P.loc[hips][1]), ph

        Q[t] = P.q
        LOC[t] = P.loc
        for b in range(n_b):
            WORLD[t, b] = P.world_loc(b)
        FACE[t] = _face_dir(P, head_i)
        FWD[t] = unit((skel.world3 @ P.pm(hips)[:3, :3]) @ np.array([0.0, 0.0, 1.0]))
        CHEST[t] = unit(skel.world3 @ (P.pm(chest_i)[:3, :3] @ chest_fwd_local))

    # One hemisphere per bone: q and -q are the same rotation, but keys that
    # flip between them interpolate through garbage (a 360-degree spin
    # crosses w = 0, where Blender's canonical w >= 0 flips the sign).
    for t in range(1, n_f):
        flip = np.einsum("bi,bi->b", Q[t], Q[t - 1]) < 0.0
        Q[t, flip] *= -1.0
    return {"names": skel.names, "frames": [int(fr["frame"]) for fr in frames],
            "q": Q, "loc": LOC, "world": WORLD, "face": FACE, "fwd": FWD, "chest": CHEST}


# ---------------------------------------------------------------------------
# Outputs.

def curves_payload(sol: dict) -> dict:
    """The same structure `run_in_blender.py curves` dumps from Blender."""
    out = []
    for t, f in enumerate(sol["frames"]):
        bones = {}
        for b, n in enumerate(sol["names"]):
            bones[n] = {"location": [round(float(x), 6) for x in sol["loc"][t, b]],
                        "rotation_quaternion": [round(float(x), 6) for x in sol["q"][t, b]],
                        "world_location": [round(float(x), 6) for x in sol["world"][t, b]]}
        out.append({"frame": f, "bones": bones,
                    "face_dir": [round(float(x), 5) for x in sol["face"][t]],
                    "body_forward": [round(float(x), 5) for x in sol["fwd"][t]],
                    "chest_forward": [round(float(x), 5) for x in sol["chest"][t]]})
    return {"frames": out, "solver": "fk_solve"}


def keys_payload(sol: dict) -> dict:
    keys = {}
    for b, n in enumerate(sol["names"]):
        if keyed(n):
            keys[n] = {"q": np.round(sol["q"][:, b], 7).tolist()}
            if n == "mixamorig:Hips":
                keys[n]["loc"] = np.round(sol["loc"][:, b], 6).tolist()
    return {"frames": sol["frames"], "keys": keys}


def use_profile(profile: dict) -> None:
    global GROUND_Z, HIP_HEIGHT, BALL_Z
    GROUND_Z = float(profile.get("ground_z", GROUND_Z))
    HIP_HEIGHT = float(profile.get("hip_height", HIP_HEIGHT))
    rest = profile.get("rest", {})
    if "l_foot" in rest and "r_foot" in rest:
        BALL_Z = 0.5 * (float(rest["l_foot"][2]) + float(rest["r_foot"][2]))


def set_constants(ground_z: float, hip_height: float, ball_z: float | None = None) -> None:
    """For the in-Blender path, which reads these from apply_mixamo_fk."""
    global GROUND_Z, HIP_HEIGHT, BALL_Z
    GROUND_Z, HIP_HEIGHT = float(ground_z), float(hip_height)
    if ball_z is not None:
        BALL_Z = float(ball_z)


def rpath(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO / p)


def main() -> None:
    ap = argparse.ArgumentParser(description="Solve a lifted clip to Mixamo keys and curves, without Blender")
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    spec = json.loads(rpath(args.spec).read_text(encoding="utf-8"))
    profile = json.loads(rpath(spec.get("rig_profile", "rig_profile.json")).read_text(encoding="utf-8"))
    use_profile(profile)
    skel = Skeleton.from_profile(profile)
    frames = json.loads(rpath(spec["joints_out"]).read_text(encoding="utf-8"))["frames"]
    import time
    t0 = time.perf_counter()
    sol = solve(frames, spec, skel)
    dt = time.perf_counter() - t0
    clip = rpath(spec["clip_dir"])
    clip.mkdir(parents=True, exist_ok=True)
    (clip / "curves.json").write_text(json.dumps(curves_payload(sol)), encoding="utf-8")
    (clip / "keys.json").write_text(json.dumps(keys_payload(sol)), encoding="utf-8")
    print(f"solved {len(frames)} frames in {dt:.1f} s -> {clip / 'curves.json'} (+ keys.json)")


if __name__ == "__main__":
    main()
