"""GVHMR estimator: video plate -> landmarks.json.

The estimator step of the video -> Mixamo Y Bot pipeline (see
docs/PIPELINE.md). Runs GVHMR (SMPL-X mesh regression) on a locked-
camera plate and emits a 33-landmark landmarks.json in a MediaPipe-
compatible convention, plus an absolute `pelvis_height` per frame:

  world:  per-frame mid-hip origin, y DOWN, z NEGATIVE toward camera,
          meters (a punch toward the camera drives wrist z below -0.3)
  image:  normalized [0..1], x right, y down
  pelvis_height: meters above the estimator's ground plane (y=0 in
          GVHMR's gravity-aligned frame) — used by airborne windows

Run with the GVHMR venv (see docs/INSTALL.md), from anywhere:

  <repo>\\tools\\GVHMR\\.venv\\Scripts\\python.exe \\
      pipeline\\estimate_pose_gvhmr.py \\
      --video plates\\<plate>\\<clip>.mp4 \\
      --out   plates\\<plate>\\landmarks.json

Joints come from the SMPL 24-joint regressor applied to the predicted
mesh (same path as GVHMR's own render_global): SMPL-X verts ->
smplx2smpl_sparse -> J_regressor. The face landmarks (nose, eyes, ears)
are read from real mesh vertices, and each frame carries a `gaze` unit
vector (nose vs ear midpoint) — head orientation is NOT recoverable
from joint positions, and synthesizing those points from the torso
basis silently locks the retargeted character's gaze to its chest.
The remaining extras (heels, hand points) are approximated; the lift
uses them for direction hints, not geometry.

GVHMR gives real depth, plausible proportions and no per-frame Z
spikes; it still does not know Mixamo bone lengths and will not plant
feet. That stays the lift's job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GVHMR_ROOT = Path(os.environ.get("GVHMR_ROOT", REPO / "tools" / "GVHMR"))

# hydra configs and every checkpoint path in GVHMR are relative to the
# GVHMR repo root. chdir before importing hmr4d.
os.chdir(GVHMR_ROOT)
sys.path.insert(0, str(GVHMR_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

MP_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# SMPL 24-joint regressor indices.
J = {
    "pelvis": 0, "l_hip": 1, "r_hip": 2, "spine1": 3,
    "l_knee": 4, "r_knee": 5, "spine2": 6, "l_ankle": 7, "r_ankle": 8,
    "spine3": 9, "l_foot": 10, "r_foot": 11, "neck": 12,
    "l_collar": 13, "r_collar": 14, "head": 15,
    "l_shoulder": 16, "r_shoulder": 17, "l_elbow": 18, "r_elbow": 19,
    "l_wrist": 20, "r_wrist": 21, "l_hand": 22, "r_hand": 23,
}

# SMPL topology vertex ids for the face. The head's ORIENTATION is not
# recoverable from joint positions (the head joint sits inside the
# skull), so the nose/eye/ear landmarks — and the gaze vector — are read
# from the mesh itself. Synthesizing them from the torso basis, as an
# earlier version did, silently locks the character's gaze to its chest.
FACE_VERTS = {"nose": 332, "left_eye": 2800, "right_eye": 6260,
              "left_ear": 583, "right_ear": 4071}

BODY_MODELS = GVHMR_ROOT / "inputs/checkpoints/body_models"
CHECKPOINTS = {
    "gvhmr": GVHMR_ROOT / "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt",
    "hmr2": GVHMR_ROOT / "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt",
    "vitpose": GVHMR_ROOT / "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth",
    "yolo": GVHMR_ROOT / "inputs/checkpoints/yolo/yolov8x.pt",
    # dpvo is only for moving cameras; plates are locked. Not required.
    "smplx_neutral": BODY_MODELS / "smplx/SMPLX_NEUTRAL.npz",
}


def preflight() -> None:
    missing = [f"  {k}: {p}" for k, p in CHECKPOINTS.items() if not p.exists()]
    if missing:
        raise SystemExit(
            "missing model files:\n" + "\n".join(missing) + "\n\n"
            "smplx_neutral is license-gated: register at https://smpl-x.is.tue.mpg.de/ ,\n"
            "download 'SMPL-X v1.1 (NPZ+PKL)' and copy SMPLX_NEUTRAL.npz to\n"
            f"  {BODY_MODELS / 'smplx'}\\\n"
            "The rest are non-gated checkpoints (see docs/INSTALL.md)."
        )


def video_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()
    return float(fps)


def prepare_video_input(path: Path, fps_override: float | None = None) -> tuple[Path, float | None]:
    """Return a PyAV-compatible input, converting animated GIFs when needed.

    GVHMR's preprocessing chain expects a video container readable through
    ImageIO's PyAV plugin. GIF frames are decoded with Pillow, sampled on a
    constant clock, and cached as H.264 without changing the source file.
    Per-frame GIF durations are honored, including variable-duration GIFs.

    The returned FPS is ``None`` for ordinary videos, preserving their
    existing probe path and behavior.
    """
    if path.suffix.lower() != ".gif":
        return path, None

    from PIL import Image, ImageSequence
    from hmr4d.utils.video_io_utils import get_writer

    frames: list[np.ndarray] = []
    durations_ms: list[float] = []
    with Image.open(path) as gif:
        default_ms = float(gif.info.get("duration", 100) or 100)
        for frame in ImageSequence.Iterator(gif):
            frames.append(np.asarray(frame.convert("RGB"), dtype=np.uint8))
            durations_ms.append(float(frame.info.get("duration", default_ms) or default_ms))

    if not frames:
        raise SystemExit(f"GIF has no frames: {path}")
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise SystemExit(f"GIF frames have inconsistent dimensions: {path}")

    durations = np.maximum(np.asarray(durations_ms, dtype=np.float64) / 1000.0, 0.01)
    fps = float(fps_override) if fps_override is not None else float(
        np.clip(1.0 / np.median(durations), 1.0, 60.0)
    )
    if not np.isfinite(fps) or fps <= 0.0:
        raise SystemExit(f"GIF FPS must be positive, got {fps}")

    starts = np.concatenate(([0.0], np.cumsum(durations)[:-1]))
    total = float(durations.sum())
    sample_times = np.arange(0.0, total, 1.0 / fps)
    indices = np.searchsorted(starts, sample_times, side="right") - 1
    indices = np.clip(indices, 0, len(frames) - 1)

    identity = f"{path.resolve()}|{path.stat().st_size}|{path.stat().st_mtime_ns}|{fps:.8f}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    cache_dir = GVHMR_ROOT / "outputs" / "input_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    converted = cache_dir / f"{path.stem}_{digest}.mp4"
    if not converted.exists():
        writer = get_writer(converted, fps=fps, crf=17)
        try:
            for index in indices:
                writer.write_frame(frames[int(index)])
        finally:
            writer.close()
    print(f"GIF input: {len(frames)} frames / {total:.3f}s -> "
          f"{len(indices)} frames at {fps:.3f} fps ({converted})")
    return converted, fps


# ---------------------------------------------------------------------------
# Stage 1 — GVHMR demo pipeline (preprocess + predict), no rendering.
# Results are cached in <GVHMR_ROOT>/outputs/demo/<video_name>/.

def side_track(tracker, video_path: str, person: str, n_people: int = 2):
    """Bounding boxes for ONE performer in a multi-person plate, selected
    by the side of frame they occupy.

    GVHMR's own `get_one_track` keeps the single largest track, which is
    useless when two people fight. Selecting by YOLO track id is fragile:
    ids swap when two bodies touch, and a swap silently splices half of
    each performer into one "track". Screen side is a fact instead of a
    guess — as long as the plate never lets them cross (see the duel
    plate's SOURCE.md), left stays left for the whole take.

    Detections are assigned per frame to `n_people` slots ordered by x,
    with nearest-centroid continuity so a momentarily missed detection
    does not shift everyone one slot over.
    """
    import numpy as _np
    import torch as _torch
    from hmr4d.utils.seq_utils import (frame_id_to_mask, get_frame_id_list_from_mask,
                                       linear_interpolate_frame_ids, rearrange_by_mask)
    from hmr4d.utils.net_utils import moving_average_smooth
    from hmr4d.utils.video_io_utils import get_video_lwh

    history = tracker.track(video_path)
    length = get_video_lwh(video_path)[0]

    def cx(b):
        return 0.5 * (float(b[0]) + float(b[2]))

    # Seed the slots from the first frame that sees everyone.
    seed = None
    for frame in history:
        if len(frame) >= n_people:
            seed = sorted(frame, key=lambda d: cx(d["bbx_xyxy"]))[:n_people]
            break
    if seed is None:
        raise SystemExit(f"never saw {n_people} people in the same frame — is this a solo plate?")
    slots_x = [cx(d["bbx_xyxy"]) for d in seed]

    boxes = _np.zeros((n_people, length, 4), dtype=_np.float32)
    seen = [[] for _ in range(n_people)]
    for f, frame in enumerate(history[:length]):
        dets = sorted(frame, key=lambda d: cx(d["bbx_xyxy"]))
        if len(dets) >= n_people:
            # Enough boxes: order alone identifies them.
            chosen = list(range(n_people))
            if len(dets) > n_people:
                # Extra detections (reflections, a bystander): keep the
                # n_people whose centres are closest to the known slots.
                chosen = []
                for sx in slots_x:
                    k = min(range(len(dets)), key=lambda j: abs(cx(dets[j]["bbx_xyxy"]) - sx) + 1e6 * (j in chosen))
                    chosen.append(k)
                chosen = sorted(chosen, key=lambda j: cx(dets[j]["bbx_xyxy"]))
            for s, j in enumerate(chosen):
                boxes[s, f] = dets[j]["bbx_xyxy"]
                slots_x[s] = cx(dets[j]["bbx_xyxy"])
                seen[s].append(f)
        else:
            for d in dets:  # partial frame: nearest slot wins
                s = min(range(n_people), key=lambda k: abs(cx(d["bbx_xyxy"]) - slots_x[k]))
                boxes[s, f] = d["bbx_xyxy"]
                slots_x[s] = cx(d["bbx_xyxy"])
                seen[s].append(f)

    order = {"left": 0, "right": n_people - 1}
    slot = order[person] if person in order else int(person)
    if not (0 <= slot < n_people):
        raise SystemExit(f"--person {person} out of range for {n_people} people")

    frame_ids = _torch.tensor(sorted(set(seen[slot])))
    if len(frame_ids) < length * 0.5:
        raise SystemExit(f"person '{person}' detected in only {len(frame_ids)}/{length} frames")
    bbx = _torch.tensor(boxes[slot][frame_ids.numpy()])
    mask = frame_id_to_mask(frame_ids, length)
    out = rearrange_by_mask(bbx, mask)
    out = linear_interpolate_frame_ids(out, get_frame_id_list_from_mask(~mask))
    out = moving_average_smooth(out, window_size=5, dim=0)
    out = moving_average_smooth(out, window_size=5, dim=0)
    print(f"person '{person}' -> slot {slot}: detected in {len(frame_ids)}/{length} frames, "
          f"mean centre x {boxes[slot][:, [0, 2]].mean():.0f} px")
    return out


def run_gvhmr(video: Path, person: str | None = None) -> dict:
    import hydra
    from hydra import compose, initialize_config_module

    from hmr4d.configs import register_store_gvhmr
    from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
    from hmr4d.utils.geo.hmr_cam import estimate_K, get_bbx_xys_from_xyxy
    from hmr4d.utils.geo_transform import compute_cam_angvel
    from hmr4d.utils.net_utils import detach_to_cpu
    from hmr4d.utils.preproc import Extractor, Tracker, VitPoseExtractor
    from hmr4d.utils.video_io_utils import get_video_lwh, get_video_reader, get_writer

    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        register_store_gvhmr()
        cfg = compose(
            config_name="demo",
            overrides=[f"video_name={video.stem if not person else video.stem + '_' + person}",
                       "static_cam=True", "verbose=False", "use_dpvo=False"],
        )
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # GVHMR restamps the working copy at 30 fps; frame count is unchanged
    # and we keep our own source-fps clock for landmarks.json.
    if not Path(cfg.video_path).exists() or get_video_lwh(video)[0] != get_video_lwh(cfg.video_path)[0]:
        reader = get_video_reader(video)
        writer = get_writer(cfg.video_path, fps=30, crf=23)
        for img in reader:
            writer.write_frame(img)
        writer.close()
        reader.close()

    paths = cfg.paths
    if not Path(paths.bbx).exists():
        tracker = Tracker()
        bbx_xyxy = (side_track(tracker, cfg.video_path, person) if person
                    else tracker.get_one_track(cfg.video_path)).float()
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
    bbx_xys = torch.load(paths.bbx)["bbx_xys"]

    if not Path(paths.vitpose).exists():
        vitpose_extractor = VitPoseExtractor()
        torch.save(vitpose_extractor.extract(cfg.video_path, bbx_xys), paths.vitpose)
        del vitpose_extractor

    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        torch.save(extractor.extract_video_features(cfg.video_path, bbx_xys), paths.vit_features)
        del extractor

    if not Path(paths.hmr4d_results).exists():
        length, width, height = get_video_lwh(cfg.video_path)
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1)
        data = {
            "length": torch.tensor(length),
            "bbx_xys": bbx_xys,
            "kp2d": torch.load(paths.vitpose),
            "K_fullimg": K_fullimg,
            "cam_angvel": compute_cam_angvel(torch.eye(3).repeat(length, 1, 1)),
            "f_imgseq": torch.load(paths.vit_features),
        }
        model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        pred = detach_to_cpu(model.predict(data, static_cam=True))
        pred.pop("net_outputs", None)  # heavy intermediates, not needed
        torch.save(pred, paths.hmr4d_results)

    return {"pred": torch.load(paths.hmr4d_results), "cfg": cfg}


# ---------------------------------------------------------------------------
# Stage 2 — SMPL-X params -> SMPL 24 joints (global ay frame + incam).

def smpl_joints(pred: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from einops import einsum
    from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
    from hmr4d.utils.net_utils import to_cuda
    from hmr4d.utils.smplx_utils import make_smplx

    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load(GVHMR_ROOT / "hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    J_regressor = torch.load(GVHMR_ROOT / "hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()

    face_idx = torch.tensor(list(FACE_VERTS.values()))

    def joints_of(params: dict, want_face: bool = False):
        with torch.no_grad():
            verts = smplx(**to_cuda(params)).vertices  # (L, Vx, 3)
            verts = torch.stack([torch.matmul(smplx2smpl, v) for v in verts])  # (L, 6890, 3)
            joints = einsum(J_regressor, verts, "j v, l v i -> l j i")  # (L, 24, 3)
            if want_face:
                return joints, verts[:, face_idx.to(verts.device)]  # (L, 5, 3)
            return joints

    joints_ay, face_ay = joints_of(pred["smpl_params_global"], want_face=True)

    # Same normalization as GVHMR render_global: frame-0 pelvis to the
    # origin (XZ), ground to y=0, then rotate so frame 0 faces +z.
    offset = joints_ay[0, J["pelvis"]].clone()
    offset[1] = joints_ay[:, :, 1].min()
    joints_ay = joints_ay - offset
    face_ay = face_ay - offset
    T_ay2ayfz = compute_T_ayfz2ay(joints_ay[[0]], inverse=True)
    joints_ayfz = apply_T_on_points(joints_ay, T_ay2ayfz)
    face_ayfz = apply_T_on_points(face_ay, T_ay2ayfz)

    joints_incam, face_incam = joints_of(pred["smpl_params_incam"], want_face=True)
    K = pred["K_fullimg"][0].cpu().numpy()
    return (joints_ayfz.cpu().numpy(), joints_incam.cpu().numpy(), K,
            face_ayfz.cpu().numpy(), face_incam.cpu().numpy())


# ---------------------------------------------------------------------------
# Stage 3 — MediaPipe-33 synthesis.
#
# ayfz frame: y up, ground y=0, character faces +z at the bind frame.
# Output world: per-frame mid-hip origin, y down, z toward camera is
# NEGATIVE. ayfz -> output is the 180° rotation about x: (x,y,z) -> (x,-y,-z).

def ayfz_to_mp(p: np.ndarray) -> np.ndarray:
    out = p.copy()
    out[..., 1] *= -1.0
    out[..., 2] *= -1.0
    return out


def mp33_from_smpl(j: np.ndarray, face: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """j: (24, 3) one frame, any right-handed y-down frame. Returns 33 pts.

    `face`: (5, 3) real mesh landmarks (nose, l_eye, r_eye, l_ear, r_ear)
    in the same frame. Without them the face points are approximated from
    the torso basis and all head-orientation information is lost.
    """

    def g(name):
        return j[J[name]]

    ls, rs = g("l_shoulder"), g("r_shoulder")
    pelvis, neck, head = g("pelvis"), g("neck"), g("head")

    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else np.zeros(3)

    left = unit(ls - rs)
    up = unit(neck - pelvis)  # y-down frame: points toward smaller y
    fwd = unit(np.cross(left, up))  # toward the camera

    def hand_pts(wrist, elbow, side):
        d = unit(wrist - elbow)
        lat = left * (0.03 if side == "l" else -0.03)
        return {
            "index": wrist + d * 0.09 + lat * 0.3,
            "pinky": wrist + d * 0.08 - lat,
            "thumb": wrist + d * 0.04 + lat,
        }

    def heel_of(ankle, toe):
        f = toe - ankle
        f[1] = 0.0
        f = unit(f)
        heel = ankle - f * 0.06
        heel[1] = ankle[1] + 0.04  # y-down: slightly below the ankle
        return heel

    lh = hand_pts(g("l_wrist"), g("l_elbow"), "l")
    rh = hand_pts(g("r_wrist"), g("r_elbow"), "r")
    l_heel = heel_of(g("l_ankle"), g("l_foot"))
    r_heel = heel_of(g("r_ankle"), g("r_foot"))
    if face is not None:
        nose, eye_l, eye_r, ear_l, ear_r = (face[0], face[1], face[2], face[3], face[4])
        # The mesh's left/right follow SMPL topology; orient them to this
        # skeleton's left so the landmark names stay honest.
        if np.dot(ear_l - ear_r, left) < 0:
            eye_l, eye_r = eye_r, eye_l
            ear_l, ear_r = ear_r, ear_l
    else:
        nose = head + fwd * 0.09
        eye_l = head + fwd * 0.07 + left * 0.03 - up * (-0.02)
        eye_r = head + fwd * 0.07 - left * 0.03 - up * (-0.02)
        ear_l, ear_r = head + left * 0.07, head - left * 0.07

    return {
        "nose": nose,
        "left_eye_inner": eye_l - left * 0.01, "left_eye": eye_l, "left_eye_outer": eye_l + left * 0.01,
        "right_eye_inner": eye_r + left * 0.01, "right_eye": eye_r, "right_eye_outer": eye_r - left * 0.01,
        "left_ear": ear_l, "right_ear": ear_r,
        "mouth_left": nose + left * 0.02 + up * (-0.03), "mouth_right": nose - left * 0.02 + up * (-0.03),
        "left_shoulder": ls, "right_shoulder": rs,
        "left_elbow": g("l_elbow"), "right_elbow": g("r_elbow"),
        "left_wrist": g("l_wrist"), "right_wrist": g("r_wrist"),
        "left_pinky": lh["pinky"], "right_pinky": rh["pinky"],
        "left_index": lh["index"], "right_index": rh["index"],
        "left_thumb": lh["thumb"], "right_thumb": rh["thumb"],
        "left_hip": g("l_hip"), "right_hip": g("r_hip"),
        "left_knee": g("l_knee"), "right_knee": g("r_knee"),
        "left_ankle": g("l_ankle"), "right_ankle": g("r_ankle"),
        "left_heel": l_heel, "right_heel": r_heel,
        "left_foot_index": g("l_foot"), "right_foot_index": g("r_foot"),
    }


def project(p: np.ndarray, K: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    z = max(float(p[2]), 1e-6)
    u = (K[0, 0] * p[0] / z + K[0, 2]) / wh[0]
    v = (K[1, 1] * p[1] / z + K[1, 2]) / wh[1]
    return np.array([u, v, 0.0])


def main() -> None:
    ap = argparse.ArgumentParser(description="GVHMR -> MediaPipe-style landmarks.json")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fps", type=float, default=None, help="source fps for the t clock (default: probe the video)")
    ap.add_argument("--person", default=None,
                    help="multi-person plates: which performer to estimate — 'left', 'right', "
                         "or a 0-based slot index ordered left to right. Omit for a solo plate.")
    args = ap.parse_args()

    def rp(p: Path) -> Path:
        return p if p.is_absolute() else (REPO / p)

    video = rp(args.video)
    out_path = rp(args.out)
    if not video.exists():
        raise SystemExit(f"video not found: {video}")
    preflight()
    video, gif_fps = prepare_video_input(video, args.fps)
    fps = args.fps or gif_fps or video_fps(video)

    res = run_gvhmr(video, args.person)
    pred = res["pred"]
    joints_ayfz, joints_incam, K, face_ayfz, face_incam = smpl_joints(pred)
    L = joints_ayfz.shape[0]

    cap = cv2.VideoCapture(str(video))
    wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080)
    cap.release()

    frames = []
    for i in range(L):
        mp_world = mp33_from_smpl(ayfz_to_mp(joints_ayfz[i]), ayfz_to_mp(face_ayfz[i]))
        hip_mid = 0.5 * (mp_world["left_hip"] + mp_world["right_hip"])
        mp_img = mp33_from_smpl(joints_incam[i], face_incam[i])  # camera frame is already y-down
        # Gaze: real face direction from the mesh (nose vs ear midpoint),
        # in the same convention as `world`. Without this the retarget can
        # only lock the head to the chest.
        ear_mid = 0.5 * (mp_world["left_ear"] + mp_world["right_ear"])
        gaze = mp_world["nose"] - ear_mid
        gn = float(np.linalg.norm(gaze)) or 1.0
        gaze = gaze / gn
        # Ground trajectory. `world` below is re-centred on the hips every
        # frame (MediaPipe convention), which throws the performer's travel
        # away — fine for a solo in-place clip, fatal for two fighters whose
        # whole relationship IS the distance between them. `root` keeps it:
        # the hip mid's ground position in the same frame as `world`,
        # starting at (0, 0) because the estimator normalises frame 0's
        # pelvis to the origin. `incam_root` is the pelvis in CAMERA space,
        # which is the only frame two separately-estimated performers share
        # — that is what measures how far apart they actually stand.
        incam_pelvis = joints_incam[i][J["pelvis"]]
        rec = {"frame": i + 1, "t": i / fps, "ok": True,
               "pelvis_height": round(float(-hip_mid[1]), 5),
               "root": [round(float(hip_mid[0]), 5), round(float(hip_mid[2]), 5)],
               "incam_root": [round(float(x), 5) for x in incam_pelvis],
               "gaze": [round(float(x), 5) for x in gaze],
               "image": {}, "world": {}, "incam": {}}
        for n in MP_NAMES:
            w = mp_world[n] - hip_mid  # per-frame mid-hip centered
            im = project(mp_img[n], K, wh)
            rec["world"][n] = {"x": round(float(w[0]), 6), "y": round(float(w[1]), 6), "z": round(float(w[2]), 6),
                               "visibility": 1.0, "presence": 1.0}
            # Camera-space landmark. `world` above is normalised into the
            # performer's OWN frame (frame 0's facing becomes forward),
            # which is what makes a solo retarget camera-independent — and
            # exactly what makes two performers incomparable: each one's
            # frame carries its own yaw, so stitching their limbs onto a
            # shared root misplaces them by however much those yaws differ.
            # The camera frame is the one frame both genuinely share.
            rec["incam"][n] = {"x": round(float(mp_img[n][0]), 6),
                               "y": round(float(mp_img[n][1]), 6),
                               "z": round(float(mp_img[n][2]), 6)}
            rec["image"][n] = {"x": round(float(im[0]), 6), "y": round(float(im[1]), 6), "z": 0.0,
                               "visibility": 1.0, "presence": 1.0}
        frames.append(rec)

    payload = {
        "source": "gvhmr_siga24",
        "person": args.person,
        "model": str(CHECKPOINTS["gvhmr"]),
        "fps": fps,
        "frame_count": L,
        "ok_count": L,
        "landmark_names": MP_NAMES,
        "frames": frames,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload), encoding="utf-8")

    b = frames[0]["world"]
    span = float(np.hypot(b["left_wrist"]["x"] - b["right_wrist"]["x"], b["left_wrist"]["z"] - b["right_wrist"]["z"]))
    print(f"wrote {out_path} frames={L} fps={fps:.3f} bind wrist span={span:.3f} m")
    print("sanity: a T-pose bind span lands ~1.25-1.40 m with SMPL wrists; the lift needs no absolute scale.")


if __name__ == "__main__":
    main()
