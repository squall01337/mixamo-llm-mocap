# The pipeline, operationally

> The upstream Mixamo path documented below remains unchanged. For the
> experimental native VRM 1.0 adapter, see [VRM.md](VRM.md). The VRM adapter
> consumes the semantic joint-position stream directly and does not convert a
> Mixamo Blender action.

How a video becomes a Y Bot clip, stage by stage, with the decision
rules an operator (human or AI) needs.

There are two retarget paths. **The SMPL-X path (section S, below) is
the default**: it transfers the estimator's joint rotations, solves the
floor, foot contacts and trajectory from the video, and runs entirely
headless. The **landmark path** (sections 1–10) is the original one:
positions only, corrected by hand with spec fields over review passes.
It is kept for existing clips and for its two-character tooling.

## S. The SMPL-X path

```
plate.mp4
  └─ estimate_pose_gvhmr.py --smplx-only   → smplx.npz (+ smplx_raw.npz)
       └─ retarget_smplx.py --spec ...       → <clip_dir>/motion.npz, curves.json
            ├─ bl_motion.py apply|render     (blender -b: keys / renders, no MCP)
            ├─ render_review.py              → review_<view>.mp4 (2×2 grid), review_details.mp4 (zoom)
            └─ eval_fidelity.py              → fidelity.json
```

```
tools\GVHMR\.venv\Scripts\python.exe pipeline\estimate_pose_gvhmr.py --video plates\<p>\<p>.mp4 --out plates\<p>\landmarks.json --smplx-only
    (runs detect_feet.py for heel/toe keypoints, then the 2D refinement)
python pipeline\retarget_smplx.py --spec action_specs\<motion>.json --rig-profile rig_profiles\ybot.json
blender -b ybot_rest.blend -P pipeline\bl_motion.py -- apply --motion clips\<clip>\motion.npz --save clips\<clip>\<clip>.blend
python pipeline\render_review.py --spec action_specs\<motion>.json --clip clips\<clip> [--legacy clips\<old_clip>]
python pipeline\eval_fidelity.py --spec action_specs\<motion>.json --clip clips\<clip> [--vs clips\<old_clip>]
```

The rig profile must carry the bone table: `setup_rig.py` writes it; for
an older profile run
`blender -b <rig>_rest.blend -P pipeline\bl_motion.py -- dump-rig --profile <profile.json>`.

### S.1 Estimate (and refine)

`estimate_pose_gvhmr.py` now keeps what GVHMR computes instead of
reducing it to 33 points: `smplx.npz` holds the SMPL-X parameters in the
world frame and in the camera frame, the camera intrinsics, the
performer's shaped rest skeleton and foot geometry, GVHMR's per-frame
contact confidences and the ViTPose 2D keypoints.

- `--postproc feet` (default) keeps GVHMR's foot clean-up but never
  treats a wrist as static: `full`, GVHMR's demo behaviour, can hold a
  fist back at the apex of a punch.
- **Foot keypoints** (`detect_feet.py`, run automatically). COCO-17
  stops at the ankle, so nothing in GVHMR's input says which way a foot
  points or whether its heel is up. MediaPipe's Pose Landmarker tracks the
  heel and the toe tip of each foot; `feet2d.npz` is cached next to the
  plate. MediaPipe is not in the GVHMR venv: the estimator calls
  `detect_feet.py` with a Python that has it (`--feet-python`, else
  `python` on PATH) and refines without feet if none is found. The model
  file is downloaded to `tools/models/` on first use.
- **Hand keypoints** (`detect_hands.py`, run automatically after the
  raw export). GVHMR estimates no fingers, and on a full-body 720p plate
  a hand is 30–50 px — a hand detector run on the frame sees a blob. Each
  hand is cropped around its ViTPose wrist (a third of a forearm beyond
  it, 0.9 forearm half-size), upscaled to 256 px, and MediaPipe's Hand
  Landmarker reads 21 points there; the detection nearest the expected
  wrist is kept. Found on 83–100 % of the frames of the five test plates
  (fists, open hands, a motion-blurred punch). `hands2d.npz` is cached
  next to the plate.
- **Refinement** (`refine_smplx.py`, on by default, `--no-refine` to
  skip). GVHMR regresses toward typical motion and flattens extremes:
  the fight kick apex came out 35–65 px (0.10–0.18 m) below where
  ViTPose sees the ankle in the same frames. The body pose is optimised
  so the model's COCO-17 keypoints (the same mesh-based definition as the
  detector's) and its heel/toe vertices land on the detections, with a
  prior toward GVHMR's pose, a smooth correction, low weight on the tiny
  face points, a strong prior on head and neck, and one physical rule: when
  GVHMR has both feet planted, their lowest points share one height (a 2D
  fit cannot see depth and otherwise tilts a wide stance). Keypoint error
  at the subject goes 2–4 cm → ~1 cm, heel/toe 2.5–6 cm → ~1 cm. The raw
  estimate is kept as `smplx_raw.npz`.

### S.2 Retarget

`retarget_smplx.py` reads the same action_spec as the legacy lift and
uses only what a video cannot know:

| Spec field | Used for |
|---|---|
| `smplx` (optional) | the parameters file (default: `smplx.npz` next to `landmarks`) |
| `rig_profile`, `action_name`, `clip_dir`, `dst_fps` | as before |
| `fists` | fist windows; hands hold a relaxed curl elsewhere |
| `rest_blend_start` / `rest_blend_end` | settle the **upper body** into the rig's exact T-pose; legs and root keep the performer's real stance |
| `arm_overrides` | authored wrist/elbow targets for an arm the estimator could not see (same schema, two-bone solve, the hand rides along) |
| `smplx_retarget` | tuning block, below |

`plant`, `smooth`, `reach`, `arm_pose`, `arm_follow`, `leg_pose`,
`head_look` and `root_offset` belong to the landmark lift and are
ignored: contacts, head orientation and extension come from the video.

What the stage does, in order:

1. **Resample** 24 → 30 fps: local rotations on a C1 spline of unrolled
   quaternions (linear interpolation leaves a velocity kink at every
   plate frame). Then **the feet's steady tilt** comes off: GVHMR stands
   each foot 0–12° toes-up and rolled, differently per foot but steadily
   through a take. Its median tilt over the frames the network calls the
   whole foot static (ankle and ball > 0.8) is removed from every frame,
   capped at 12° (`foot_tilt_fix`). Planted feet lock flat anyway; the
   bias showed as a flap at each lock ramp and a toes-up swing foot, and
   removing it improved leg angles on four plates of five and foot
   direction on all.
2. **Camera-consistent trajectory.** The static camera is fitted from the
   world and camera-frame roots over the first frames. Height is taken
   from the camera continuously (GVHMR's world track drifts 5–15 cm
   vertically); horizontal drift (up to 0.45 m on these plates) is
   applied one footfall at a time, so a planted foot never slides to pay
   for it.
3. **Contacts.** Four kinds of evidence plant a foot: GVHMR's static
   confidence (validated by foot speed and height), a still sole on the
   floor once the floor is known, a foot that does not move **on screen**
   (median speed of ankle, heel and toe keypoints on median-filtered
   tracks, sole within 12 cm of the floor — the estimate can leave a
   planted foot 6–9 cm up; the camera is static, so still in the image is
   still in the world), and **single support**: when one foot is more than
   15 cm above the other and the lower one is within 14 cm of the floor,
   the lower one carries the body — through the spin plate's kicks GVHMR
   slides it at up to 2 m/s while the video shows it fixed, and it holds
   the jump's take-off foot 10–13 cm up. Only observed contacts (network,
   image, floor) set the floor's height: an inferred support locks its
   foot, and its leg reaches down, but grounding on it would lower the
   whole body — and the jump after it — below where the camera sees it.
   Runs of the same foot separated by a confidence dip, with the foot
   unmoved, are joined.
4. **Planted feet stay put**: where a foot is flagged (network, image, or
   single support), the body's horizontal translation absorbs the foot's
   motion (GVHMR's own rule, re-applied on the refined pose, anchored on
   the stillest flagged joint: the ball of a support foot whose heel is
   up, so the body turns about it). The correction is smoothed over a
   frame (`pin_smooth`): as the support passes from one foot to the
   other, the body would otherwise jolt.
5. **Floor**: on planted frames the lowest sole goes on the floor.
6. **Camera footfalls**: each footfall takes the camera correction
   measured when it lands; later disagreement while it stays planted is
   the camera's depth estimate breathing with the pose.
7. **Rotation copy**: `P_b = S_j · A_b · B_b` in armature space (the
   Mixamo armature's own axes are SMPL-X's). `A_b` aligns rest
   directions for thighs, shins, upper arms and forearms; feet are
   aligned in yaw only, measured heel-to-toes on the SMPL-X mesh (the
   ankle→ball joint direction reads 15° more toe-out than the foot
   has). Hands and toes inherit their parent's. Spine, neck, head and
   collars are not aligned: there the skeletons differ in proportion,
   not pose.
8. **Root and legs**: the pelvis trajectory is scaled by the leg-length
   ratio; each leg is re-solved so the ankle lands where the performer's
   scaled leg puts it, knee in the source's plane.
9. **Foot locks**: heel and ball down locks the ankle and lays the rig's
   own flat foot at the stance's median heading — the heading of the
   source foot laid flat, so a foot the estimate rolls onto its edge keeps
   the direction it points in; otherwise the foot pivots on whichever
   point is lower (ball for a raised heel, heel for a toe-up plant) and
   keeps its real pitch. A toe-up heel plant held longer than 0.5 s is
   laid flat: heel strikes and turns on the heel are brief, and GVHMR
   returns the planted rear foot of a long stance toes-up (see
   PITFALLS 55). Toes lie on the floor on flat and ball contacts. A lock
   only holds while the planted point is still (a single-support foot,
   or one still on screen, always is): stretches where it slides are left
   to follow the source smoothly, a lock that drifts more than 4 cm is
   cut, and so is a flat lock whose foot turns more than 20° (a stance
   pivoting on its heel gets a heading per piece). Ramps of 4 frames. A
   lock a leg cannot reach lowers the hips, at most 5 cm. A free foot the
   estimate carries more than 1 cm under the floor is lifted by its own
   leg (`sink_tolerance`: a toe grazing the floor is left alone — lifting
   it bends the knee away from the video); one hovering less than 6 cm
   above it while the body stands is set down by its leg, height only
   (`floor_snap`: the estimate lifts a foot the video shows down through
   turning footwork).
10. **Fingers** from `hands2d.npz`: per finger joint, the angle between
    consecutive bones of MediaPipe's metric landmarks (view-independent),
    median-filtered over confident frames and smoothed; each Mixamo
    finger bone bends by it about the axis of the rig-validated fist
    (capped a little past the fist), the thumb follows one curl. Where
    the detector lost the hand for more than a few frames, the spec's
    `fists` windows and the relaxed curl (`hand_relaxed`) take over.
11. Rest blends, then local quaternions for every bone and the Hips
    location, plus the plate camera (for the overlay) and QA numbers.

`smplx_retarget` keys (defaults): `contact_threshold` 0.5,
`contact_max_speed` 0.3 m/s, `contact_max_height` 0.06 m,
`contact_heel_height` 0.03 m, `contact_geo_height` 0.025 m,
`contact_geo_speed` 0.2 m/s, `contact_still_px` 15 px/s,
`contact_slide_speed` 0.12 m/s, `contact_max_drift` 0.04 m,
`contact_heel_max_s` 0.5 s, `contact_still_height` 0.12 m,
`contact_swing_height` 0.15 m, `contact_support_height` 0.10 m,
`contact_max_turn_deg` 20, `contact_support_height` 0.14 m, `floor_snap` true,
`floor_snap_height` 0.06 m, `pin_smooth` 1, `sink_tolerance` 0.01 m, `foot_tilt_fix` true,
`contact_image_anchor` false (with `contact_anchor_max` 0.12 m,
`contact_anchor_drop` 0.015 m), `contact_ramp` 4, `max_hips_drop` 0.05 m,
`ground_sigma` 3, `hand_relaxed` 0.3, `hand_detect` true, `pin_static_feet` true,
`camera_root` true, `camera_fit_frames` 12, `camera_landing_frames` 6,
and an optional
`smooth: {"min_cutoff": 3, "beta": 8}` (One-Euro). Smoothing is **off**
by default: even the gentlest setting cost 3–5 % of punch extension and
3 cm of kick height on the fight plate.

`contact_image_anchor` moves each flat lock across the line of sight to
where the plate shows the ankle (the camera ray through ViTPose's ankle,
met by the plane at the performer's ankle height). It puts planted feet
exactly on the performer's in the overlay, but it is off by default: a rig
with wider hips than the performer (the Y Bot's hip joints are 18 cm
apart, the performers' 11–13 cm scaled) must then angle its legs away
from the video's — leg angles 0.3° better on two plates, 0.2–0.6° worse on
three. Along the line of sight the image cannot place a foot at all: the
ray meets the floor at 12–19°, where 1 cm of ankle height is 3–4 cm of
depth.

### S.3 Review and measure

`render_review.py` renders headlessly and writes one 2×2 video per view:
the plate, the clip rendered through the plate's recovered camera and
composited over it, and (with `--legacy`) the old clip next to the new
one with the same camera and look. The overlay is the check to trust:
where character and performer disagree, it is visible. It also writes
`review_details.mp4`: both hands and both feet zoomed, plate and overlay
side by side, following the wrists and ankles (the plate camera's view is
rendered at twice the plate's resolution for it; `--no-details` skips
it) — fists, fingers and foot placement can be judged there.

`eval_fidelity.py` projects the clip through that camera and compares
every limb's on-screen angle with the ViTPose keypoints on every plate
frame, each foot's ankle→toe line with MediaPipe's, and reports
planted-foot slide (on the frames GVHMR flags, not the clip's own
locks), sole height and jerk. Read the angles as relative: joint centres
and COCO keypoints are different anatomical points (GVHMR's own
skeleton scores ~5.9° on the fight plate), and the foot line of a rig
with a higher ankle and a longer foot than the performer's differs from
MediaPipe's ankle→big-toe line by a few degrees on its own — treat the
foot number as ±5° and judge finer differences on the overlay.

### S.4 What the literature adds, and what it would take

A last survey (September 2026) of what could push the feet further:

- **Contact handling** is where the pipeline already matches the
  references: footskate cleanup by constrained IK with blended edges
  (Kovar, Schreiner & Gleicher 2002), and game-style foot locking with
  0.1–0.2 s inertialized engage/release, locking the part of the foot that
  carries the weight and letting a real slide go (Holden, *Inverse
  Kinematics and Foot Locking*). Ramps here are smootherstep over 4 frames
  (0.13 s).
- **Learned contact detectors** (UnderPressure 2022, ContactVision 2025)
  and **physics tracking** (PhysCap; CRISP 2025, which fits planar
  support surfaces to contacts and imitates the motion with an RL
  controller) remove the remaining implausibility — a character that
  cannot push off a floor it does not touch — at the cost of a simulator
  or a trained model per plate. That is the next level, not a tuning.
- **Post-refinement of GVHMR** by aligning velocities and accelerations
  predicted from the video (HTD-Refine 2026) cut jitter 58 % and world
  MPJPE 42 % on EMDB, but needs its learned velocity network; the idea
  that transfers — penalise jerk, keep contacts by IK — is already here,
  and pre-smoothing was measured to cost punch extension.
- **Hands**: HaMeR/WiLoR recover MANO hands from crops and would beat
  MediaPipe on blurred fists, but need the gated MANO model; the crop-
  then-detect pattern is the same one `detect_hands.py` uses.

## 0. The plate (source video)

Quality is won or lost here. Rules (see docs/PROMPTING.md for
generating plates with AI video):

- Locked camera, full body in frame the whole time, even light, 720p+.
- **T-pose held ~1 s at the start AND the end**, facing the camera
  square. The T-poses are the rest anchors; the pipeline blends into
  exact Y Bot rest there.
- One continuous take, on the spot. Every moving limb must stay
  visible — an occluded limb cannot be tracked (it can be fixed with an
  `arm_override`, but only because a human saw the video).
- Facing changes are fine (the kungfu clip turns ±100°); occlusion is
  the enemy, not rotation.

Save as `plates/<name>/<clip>.mp4` with a short `SOURCE.md`.

Animated GIF plates are accepted as well. The estimator preserves their
per-frame timing and writes a deterministic cached H.264 working copy under
`tools/GVHMR/outputs/input_cache/`; the original GIF is never modified. GIFs
often have coarse timing, so use `--fps <value>` only when the intended source
rate is known and the embedded durations are wrong.

## 1. Estimate

```
tools\GVHMR\.venv\Scripts\python.exe pipeline\estimate_pose_gvhmr.py ^
    --video plates\<name>\<clip>.mp4 --out plates\<name>\landmarks.json
```

Output: 33 landmarks/frame (MediaPipe-compatible convention: per-frame
mid-hip origin, y down, z **negative toward camera**, meters),
`pelvis_height` (absolute, above the estimator's ground plane) and
`gaze` (unit vector, the performer's real face direction taken from
mesh vertices — nose vs ear midpoint). The face landmarks are real mesh
points, not torso-derived: head orientation cannot be recovered from
joint positions, and a retarget without it can only lock the character's
gaze to its chest.
GVHMR caches its preprocessing under `tools/GVHMR/outputs/demo/<stem>/`
— re-runs are fast.

## 2. Numeric beat sheet — BEFORE any spec

```
tools\GVHMR\.venv\Scripts\python.exe pipeline\analyze_landmarks.py --landmarks plates\<name>\landmarks.json
```

Write `plates/<name>/BEATSHEET.md`: a table of beats with src frame
windows, the acting limb, the support foot, and the *numeric evidence*
(this tool's output). Hard rules:

- **Take every limb decision from the numbers, never from stills.**
  Facing the camera, viewer-left is the character's RIGHT; every
  screen-read limb guess in this project's history was wrong at least
  once. Gen-video also deviates from its prompt (a prompted left-knee
  jump arrived as right-knee).
- Punches: wrist z dips below −0.32 (toward camera). Kicks/knees: one
  ankle height rises with the other grounded. Flight: both ankles
  > 0.25 m. Ducks/jumps: `pelvis_height` dips/peaks.
- Shoulder-line yaw swings ±40–60° during punches without the hips
  turning — don't read those as facing changes.

## 3. The action_spec (the motion as data)

`action_specs/<motion>.json` — all frame numbers are **source** frames
(the lift converts to 30 fps dest internally):

```jsonc
{
  "name": "combo",                          // slug
  "action_name": "YBot_ComboFromVideo",     // Blender action name
  "clip_dir": "clips/ybot_combo_from_video",// outputs (repo-relative)
  "landmarks": "plates/combo/landmarks.json",
  "joints_out": "plates/combo/joints_mixamo.json",
  "estimator": "gvhmr_siga24",
  "src_fps": 24, "dst_fps": 30,

  // Blend into exact Y Bot rest. End is required; start is optional
  // (use it when the plate opens on a clean T-pose).
  "rest_blend_start": { "full_src": 1, "release_src": 16 },
  "rest_blend_end":   { "start_src": 214, "full_src": 232 },

  // Fist amount 0->1->0, smooth edges.
  "fists": { "rise_src": [20, 28], "fall_src": [200, 212] },

  // Optional authored overrides for beats where the human's read of
  // the video beats the estimator (occluded arm etc.). Targets are
  // METERS in the hip basis (x=character left, y=back, z=up), ramped.
  "arm_overrides": [
    { "side": "right", "src": [76, 163], "ramp_src": 6,
      "wrist_local": [-0.17, -0.12, 0.33],
      "elbow_local": [-0.25, 0.05, 0.12] }
  ],

  // The support schedule — the heart of the spec. Windows must cover
  // the clip. Supports: "both", "left", "right", "none" (airborne).
  "plant": [
    { "src": [1, 129],   "support": "both"  },
    { "src": [130, 154], "support": "left"  },   // right leg acts
    { "src": [155, 158], "support": "right" },
    { "src": [159, 168], "support": "none"  },   // flight
    { "src": [169, 177], "support": "left"  },   // landing
    { "src": [178, 241], "support": "both"  }
  ],

  "qa_src_frames": [1, 30, 60, 109, 139, 163, 241]  // beat frames for QA/stills
}
```

What each support does in the FK apply:
- `left`/`right`: hip height searched so that foot lands at 0.105 m;
  the foot is flattened (Z-only, keeps estimator XZ and its own
  heading); the lift also **pins the support ankle's XZ** for the whole
  window (the pose translates so the planted foot never skates; the
  correction decays over 10 frames after the window).
- `both`: plant the lower foot, flatten both if near ground. Feet may
  genuinely step inside a `both` window — that is fine.
- `none`: no plant, no snapping; hip height integrates the estimator's
  `pelvis_height` arc from the last planted frame (continuous takeoff,
  ballistic flight). Landing pops are caught by QA's hip-step check.
  Two tuning notes from real passes: (a) airborne detection fires only
  once feet clear ~0.25 m, which EATS the launch rise if the window
  starts there — start the `none` window at the true liftoff (watch
  `pelvis_height` start climbing steeply); (b) the estimator tends to
  understate jump amplitude — an optional `"boost": 1.3` on the window
  scales the integrated arc (landing stays continuous because the arc's
  endpoint is scaled too).

An optional top-level `"smooth"` list applies extra windowed
Savitzky-Golay smoothing to selected joints (default: both arm chains)
where fast flurries read staccato:

```jsonc
"smooth": [ { "src": [172, 204], "window": 9 } ]
```

Windowed correctors for systematic estimator bias inside a phase (all
ramped; they leave the rest of the clip untouched):

```jsonc
// Follow the performer's own arm geometry for a beat: the wrist is
// placed at their wrist offset (scaled to this character) and the chain
// re-solved with exact bone lengths. This is the strongest corrector —
// it copies real human geometry, so it cannot self-intersect — and it
// needs no hand-tuned numbers. Anchor "shoulders" (default) keeps the
// anatomy honest; "head" matches hand-to-face height instead, which can
// drive hands into the torso when proportions differ. Use it for any
// beat being judged frame by frame.
"arm_follow": [
  { "src": [140, 202], "ramp_src": 4, "side": "both", "anchor": "shoulders" }
],

// Rotate a whole arm rigidly about its shoulder socket. Targets are in
// METRES and the angle is solved per frame, because a fixed angle over-
// or under-shoots as the arm swings. Rigid rotation preserves elbow
// bend, wrist alignment and the distance between the hands — offsetting
// the joints instead (and re-normalizing bone lengths) collapses the
// hands together and folds the wrists at anything beyond ~2 cm.
"arm_pose": [
  { "src": [138, 206], "ramp_src": 6, "side": "both",
    "drop_m": 0.145,      // + lowers the hands along an arc
    "widen_m": 0.085 }    // + opens the hands apart
  // also accepts raw "pitch_deg" / "yaw_deg" if you prefer angles
],

// Restore strike extension. Scales the hand's distance from its
// shoulder socket along its own direction, then re-places the chain
// with exact bone lengths (current elbow as pole). Only arms already
// past `min_extension` of full length are affected, so guards stay put.
"reach": [
  { "src": [172, 204], "ramp_src": 4, "side": "both",
    "factor": 1.22, "max_fraction": 0.97, "min_extension": 0.30 }
],

// Authored targets for a beat the estimator got wrong (occluded arm).
// Positions in metres in the hip basis (x=left, y=back, z=up).
"arm_overrides": [
  { "side": "right", "src": [76, 163], "ramp_src": 6,
    "wrist_local": [-0.17, -0.12, 0.33], "elbow_local": [-0.25, 0.05, 0.12] }
]
```

**Sizing a correction:** measure, don't guess. Compare the retarget
against the reference on quantities the eye actually reads — hand height
relative to the HEAD, distance between the hands, hand distance from the
chest — and remember to subtract the part that is proportion rather than
pose (a character whose head sits lower on its shoulders will always
read "hands high" at identical joint angles).

### Gaze

The head follows the estimator's real gaze by default — no spec entry
needed. Two top-level knobs tune it:

```jsonc
"gaze_follow": 1.0,          // 0..1, how strongly the head tracks the real gaze
"gaze_max_offset_deg": 70    // cap on head-vs-chest yaw, so a 360 spin
                             // does not twist the neck off
```

Per-window overrides go in `head_look` entries as `"follow_gaze"`, and
the manual correctors remain for the rare case where the estimator's
head is wrong: `"amount"` (blend heading toward character-forward),
`"level_face"` + `"level_target_deg"` (put the face at a fixed elevation
by closed loop on the rig).

`qa_clip.py` audits the result: it flags a head locked to the chest
(gaze-vs-chest std < 2°, the signature of an estimator that isn't
emitting real head orientation) and a head staring upward (mean
elevation > 12°).

**Smoothing eats strikes.** A `smooth` window wide enough to calm a
fast flurry (9 frames) also averages away its extension peaks — the
punches get visibly shorter. Keep flurry windows narrow (5) and
restore amplitude with `reach` rather than widening the smoothing.

## 4. Lift

```
tools\GVHMR\.venv\Scripts\python.exe pipeline\lift_to_mixamo.py --spec action_specs\<motion>.json
```

Direction-preserving retarget: estimator segment directions × Mixamo
bone lengths from the Mixamo sockets. No IK. Read the QA lines it
prints (one per `qa_src_frames`): rest/fist amounts, plant, yaw, key
positions. Sanity: rest frames print the exact Y Bot rest
(`lw=[0.73777, 0.06171, 1.43572]`).

## 5. Apply (inside live Blender)

Blender must be open on `ybot_rest.blend` with the MCP add-on server
running. One command drives all three Blender-side stages:

```
python pipeline\run_in_blender.py all action_specs\<motion>.json
```

(`apply` keys the action — ~2–4 min per 300 frames, Blender's UI
freezes, that's normal; `curves` dumps every bone every frame;
`stills` renders front+side PNGs at the spec's QA frames. Run stages
individually with `apply|curves|stills`.) Agents with MCP can instead
call `apply_mixamo_fk.run/dump_curves/run_stills_render` directly via
`execute_blender_code`.

Stills and previews use a temporary camera + Workbench render
(window-independent); never use viewport screenshots — they capture
black when the Blender window is occluded.

To produce the review videos (`preview.mp4` + the side-by-side
`showcase.mp4` against the source plate):

```
tools\GVHMR\.venv\Scripts\python.exe pipeline\render_preview.py action_specs\<motion>.json --showcase
```

## 6. Compare against the reference (refinement)

```
tools\GVHMR\.venv\Scripts\python.exe pipeline\compare_reference.py --spec action_specs\<motion>.json
... --from-src 130 --to-src 210     limit to a beat
... --detail                        per-frame table
```

QA proves a clip is not broken; this proves it looks like the video. It
measures both on the quantities an eye reads — hand height relative to
the face, distance between the hands, hand distance from the chest,
limbs inside the torso, gaze yaw and elevation — and reports the
**source-frame windows** where they diverge, ready to become spec
entries. Run it after every pass; it catches over-corrections (a fix
that pushes a hand into the body) before a human has to.

Two measurement traps it handles, worth knowing if you extend it:
compare against a FACE proxy (mid-skull), not the head joint, which
sits at the skull base and bakes in a false offset; and report the
character's own head/shoulder proportion separately, so proportion is
never mistaken for pose error.

## 7. QA

```
tools\GVHMR\.venv\Scripts\python.exe pipeline\qa_clip.py --spec action_specs\<motion>.json
```

| Check | Limit | Meaning |
|---|---|---|
| world bounds | \|x\|,\|y\| < 2, z ∈ (−0.5, 3) | exploded bones |
| hip Z step | < 0.12 m/frame | pops (takeoff/landing included) |
| in-place travel | end-to-end ≈ 0 | root discipline |
| single-support | z-err ≈ 0, XZ wander ≈ 0 | plant + no skate |
| airborne | lower-foot clearance > 0.12 m | the jump actually flies |
| end frame | hand error < 0.03 vs rest | clean T-pose out |
| frame-jumps | < 0.30 m per bone | teleporting limbs |

`WARN` on a `both` window usually means a genuine step (weight shift,
stance widening) — check the video before "fixing" it. **Numbers can
pass while the motion is wrong: always look at the stills and play the
clip. The human eye is the last word.**

## 8. Iteration protocol (how 16 passes became 1–2)

- Beat sheet before any spec. A wrong limb in the spec is a wasted pass.
- Human notes come as frame ranges + a region ("frames 18–50, punching
  side"). Map the range to the beat and to the spec field that owns it
  BEFORE editing anything.
- One constraint per pass. Never touch a region the human has signed
  off to polish a different one.
- Re-render stills after every apply; never argue from a stale image.
- When the estimator and the human's read of the video disagree, the
  human is right (occlusion, foreshortening) — encode the fix as an
  `arm_override`, don't fight the estimator globally.

---

## 9. Two characters in one scene

A multi-performer plate is not "run the pipeline twice". Two
individually perfect retargets still fail if the characters stand at
the wrong distance — every punch swings through empty air, or a fist
ends up inside a skull. What changes:

### 9.1 Estimate each performer separately

```
... estimate_pose_gvhmr.py --video plates\duel\duel.mp4 --person left  --out plates\duel\landmarks_grey.json
... estimate_pose_gvhmr.py --video plates\duel\duel.mp4 --person right --out plates\duel\landmarks_white.json
```

`--person left|right|<index>` selects a performer by **which side of
frame they occupy**, not by tracker id. Ids swap when two bodies touch,
and a swap splices half of each performer into one "track"; screen side
is a fact as long as the plate never lets them cross — which is why the
plate contract forbids it (docs/PROMPTING.md). Caches are per person,
so the two runs do not fight over the same preprocessing.

Each landmarks file then carries, besides the solo fields, `root`
(ground trajectory, gravity-aligned frame), `incam_root` and `incam`
(pelvis and all 33 landmarks in CAMERA space).

### 9.2 Build one scene with both characters

```
blender --background --python pipeline\setup_duo.py -- ^
    --char YBot=ybot.fbx --char Ninja=ninja_rest.blend --out duel_rest.blend
```

Each character gets `Armature_<Name>` and its own
`rig_profiles/<name>.json`. Two characters do **not** share a hip
height — planting the Ninja at the Y Bot's ground offset buries its
feet — so every stage now takes the profile from the spec.

Both armature objects stay at the origin. Stage placement is baked into
the animation instead (`stage_x`), because the FK apply aims bones in
armature space and a translated armature object would silently offset
every aim target (docs/RIG.md).

### 9.3 Spec fields a paired clip adds

```jsonc
{
  "armature": "Armature_YBot",              // which character in the scene
  "rig_profile": "rig_profiles/ybot.json",  // its measured proportions

  "root_motion": true,     // keep the performer's travel (see below)
  "root_source": "incam",  // "incam" (default) or "global"
  "root_depth": false,     // include the depth axis too (default: lateral only)
  "stage_x": -0.943,       // where this character stands
  "root_scale": 0.9715,    // ONE stage, ONE scale — shared by both fighters

  "prefilter_window": 5    // landmark prefilter width, source frames (default 7)
}
```

**Root motion.** Solo clips are retargeted in place — Mixamo convention
and correct, since the estimator's landmarks are hip-centred every
frame anyway. Two fighters are the opposite case: the distance between
them *is* the scene. On the duel plate the attacker closed 1.95 m →
0.88 m and back out again; in place, every punch lands a metre short.

Two estimates of that travel exist and they disagree. `global` is the
gravity-aligned world trajectory: physically consistent but *predicted*,
and it drifts (it under-reported that 0.92 m step-in as 0.68 m and left
the performer 0.16 m off his mark at the closing T-pose). `incam` is
tied to what the camera saw and tracked an independent image-space
measurement to within 2 cm across all 241 frames — so it is the
default. Only its lateral component is used; depth is the
ill-conditioned axis of a single-camera fit.

Pinning is skipped when `root_motion` is on. The foot skate it corrects
is an artefact of *discarding* the travel; restore the travel and the
artefact is gone at the source, while re-pinning on top would cancel
each step inside its own stance and snap it back on release.

**Sizing the stage.** Measure the performers' separation at the T-pose
from `incam_root`, scale it by the shared stage scale, and split it
either side of the origin. Then verify with `compare_pair.py` rather
than trusting the arithmetic.

### 9.4 `leg_pose` — the leg's `arm_pose`

```jsonc
"leg_pose": [
  { "src": [142, 150], "ramp_src": 3, "side": "right", "lift_m": 0.18 }
]
```

Rotates a whole leg rigidly about its hip socket so the foot rises (or
drops) by a measured amount, solved per frame. The axis is the
horizontal one perpendicular to the leg itself — a roundhouse and a
front kick travel in different planes, and a fixed lateral axis would
skew one of them sideways.

It exists because the estimator's gravity-aligned pose and its
in-camera pose disagree about a fast limb at its peak, and the retarget
must follow the gravity-aligned one (that is what makes feet plant). On
a head-high roundhouse that showed as a shin ~9° low, dropping the foot
0.15 m — putting it *through* the ducking defender instead of over him.
Size it from `compare_pair.py`, never by eye.

### 9.5 Render and check the pair

```
... render_preview.py action_specs\duel_ybot.json --also action_specs\duel_ninja.json ^
      --out-dir clips\duel --showcase --social
... compare_pair.py --spec action_specs\duel_ybot.json --spec action_specs\duel_ninja.json
```

`--also` binds every character's action before rendering (otherwise one
fighter renders frozen in its T-pose while the other fights it) and
frames the camera from the dumped curves so both stay in shot.

`compare_pair.py` measures the three things neither `qa_clip.py` nor
`compare_reference.py` can see:

| Check | What it catches |
|---|---|
| separation | the pair standing too far apart / too close, per frame |
| reach | a strike that stops short of, or drives past, where it landed in the video |
| intrusion | a limb *inside* the other character — absolute, no reference needed |

It reads both performers in the **camera** frame, re-expressed in a
gravity-aligned basis solved from the data (docs/PITFALLS.md #26–27),
and tests whole limb SEGMENTS, not endpoints.

Run all three: `qa_clip` per character (is it broken?),
`compare_reference` per character (does it look like its performer?),
`compare_pair` once (do the two relate like the two performers did?).

### 9.6 Clearance — the part that has no solo equivalent

Two individually faithful retargets can still be wrong on screen,
because **two Mixamo characters are not two humans**. Their limbs,
heads and hands are thicker, and a fight choreography is built out of
near-misses. On the duel plate the attacker's roundhouse passes the
defender's guard hand 0.111 m centre-to-centre; with real forearms that
is a 2 cm miss, and with these characters' meshes it is a collision.
The retarget reproduced that 0.111 m to the centimetre and was
*therefore* wrong.

**Measure it on the meshes.** The capsule proxies in `compare_pair.py`
are right about separation and reach and optimistic about contact — they
scored this kick as clearing by 4 mm while the meshes intersected over
230 face pairs. Ground truth is the evaluated, skinned geometry:

```
python pipeline\run_in_blender.py contact action_specs\duel_ybot.json ^
       --with action_specs\duel_ninja.json
```

BVH overlap between the two characters, every frame, written to
`<clip_dir>/../pair_contact.json`: intersecting face-pair counts, and
the surface-to-surface gap where they do not intersect. Run it over the
WHOLE clip — restricting it to the beat under repair is how four other
intersections survived two passes here, one of them in the closing
T-pose.

**Then separate intended contact from unintended.** Punches landing on
a block *should* touch; two rigid hands cannot deform, so those frames
show deep overlap and that is the strike landing. Ask what the source
video does at that frame.

**Buy the clearance from the stage, not the poses.** Every pose lever
was tried on this collision and each one moved the problem elsewhere:
raising the kick swept the foot through the guard hand, lowering the
defender dropped his hands into the rising foot, curling his spine swung
his arms forward, leaning him away lifted his head into the arc. Only
distance helped every frame at once. Requirements that flip between
adjacent frames are the signature of a graze no pose fix can satisfy.

```jsonc
"root_offset": [
  { "src": [116, 185], "ramp_src": 23, "dx": -0.20 },   // room for the kick
  { "src": [186, 241], "ramp_src": 18, "dx": -0.07 }    // room for the bind
]
```

A windowed, ramped shift of one character's ground trajectory. Two
rules: ramp it across frames where that character is **already
travelling** (0.20 m applied under a planted foot is 0.20 m of skate —
here it rides the 0.24 m retreat he already makes between the punches
and the kick), and **declare it**, so `compare_pair.py` prints
`[declared root_offset]` next to the separation it causes instead of
reporting it as an unexplained defect. The cost stays visible.

The second window above is a different instance of the same physics:
these characters' arms are ~4% longer relative to their legs than the
performers', so at a leg-scaled stage separation their fingertips cross
while the arms swing out into the closing T-pose. The settled T-poses
were always clear (0.09–0.13 m); only the spread collided.

### 9.7 Order of operations for a paired clip

1. Estimate both performers, write both beat sheets.
2. Size the stage from `incam_root` at the T-pose; lift and apply both.
3. `qa_clip` each — is either clip broken?
4. `compare_reference` each — does each look like its performer?
5. `compare_pair` — do the two relate like the performers did?
6. `run_in_blender.py contact` over the whole clip — do the meshes
   actually touch?
7. Only now reach for clearance, and only for contacts the video does
   not have.

Steps 3–6 answer different questions and none of them substitutes for
another. Step 6 is the one that agrees with a human watching the video.

## 10. The review pass (do not skip this)

Every defect that mattered in this project was found by putting the
source frame and the retargeted frame **side by side at the same beat
and looking at them**, then measuring what the eye flagged. Numbers
alone shipped a foot through a head; an eye alone could not say by how
much or why. The loop is:

1. Render the showcase (`render_preview.py --showcase`).
2. Pull paired crops at the beat frames — source left, retarget right,
   labelled with the source AND destination frame number. A contact
   sheet every ~10 frames for the whole clip, then a tight crop on any
   beat that looks wrong.
3. Name what looks wrong in one sentence, in body terms ("the foot goes
   through his head", "his guard is too narrow").
4. Turn that sentence into a measurement, and only then edit a spec.

**When your eye and your numbers disagree, one of them is measuring the
wrong thing — and it is usually the numbers.** Every time it happened
here the cause was a mismatched proxy: a nose compared to a skull-base
joint, a shoulder line to a Neck joint, limb endpoints instead of whole
segments, a capsule instead of the mesh. Before trusting a number,
name the anatomical point on both sides and check they are the same
one. Before dismissing what a human saw, assume the model is wrong.

This is the whole reason `compare_reference.py`, `compare_pair.py` and
the `contact` stage exist: they turn "his arm clips his back" into a
frame window and a distance in metres, so the fix can be sized instead
of guessed — and so an over-correction is caught on the next run
instead of after it ships.
