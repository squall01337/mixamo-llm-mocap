# Architecture & Feasibility Report: Native VRM Retargeting

Status: architecture audit complete; implementation gated on Phase 0 baseline.

Audit baseline: commit `00dfd5385506022d533c84f6737a09f5f4392623`, which is identical on `origin/main` and `upstream/main` at the time of audit.

## Executive conclusion

The pipeline can become rig-independent without changing GVHMR. Its useful rig-independent product is reconstructed, time-sampled human joint positions plus semantic per-frame state. The current implementation combines that product with Mixamo proportions in `lift_to_mixamo.py`, then combines the resulting positions with Mixamo bone names, bind axes, FK behavior, and Blender keying in `apply_mixamo_fk.py`.

The best seam is therefore immediately after motion reconstruction and temporal/spec processing, but before character rest positions and segment lengths are applied. The present `joints_mixamo.json` is close to a canonical motion stream in shape, but is not canonical in meaning: its coordinates, proportions, rest blend, stage placement, and naming are already tailored to Mixamo/Y Bot conventions.

Recommended architecture:

```text
GVHMR landmarks + action spec
              |
              v
CanonicalHumanoidMotion (positions, body basis, timing, semantics)
              |
        +-----+-----+
        |           |
        v           v
MixamoRigAdapter  VrmRigAdapter
        |           |
        v           v
Blender FK/action  Blender FK/action
```

This is direct motion-to-rig retargeting. A Mixamo-animation-to-VRM converter is not the target design.

## Phase 0 baseline status

The checked-out repository contains source code, specs, documentation, and showcase GIFs only. It does not contain:

- a source plate (`.mp4`);
- a Mixamo character (`.fbx`) or prepared `.blend` scene;
- GVHMR source/environment/checkpoints;
- the license-gated `SMPLX_NEUTRAL.npz` body model;
- generated `landmarks.json`, lifted joints, curves, or QA output;
- a VRM model.

Blender 5.1 is installed at `C:\Program Files\Blender Foundation\Blender 5.1`, and the machine has an RTX 4070 Laptop GPU with 8 GB VRAM. A local GVHMR environment, checkpoints, and the license-gated SMPL-X model were subsequently installed and used to complete the baseline.

Consequently, a successful single-character Mixamo run cannot be reproduced from repository contents alone. The bundled showcase is evidence of an upstream run, but is not a reproducible baseline. Per the repository rules, no architecture implementation should be committed until the missing baseline assets are supplied and the documented run succeeds.

The baseline evidence bundle should preserve:

```text
plates/<baseline>/<baseline>.mp4
  -> plates/<baseline>/landmarks.json
  -> action_specs/<baseline>.json
  -> plates/<baseline>/joints_mixamo.json
  -> clips/<baseline>/apply_result.json + curves.json
  -> clips/<baseline>/preview.mp4 + showcase.mp4
  -> qa_clip and compare_reference console logs
```

Do not commit assets whose Mixamo, SMPL-X, or source-video terms forbid redistribution.

## Existing internal motion representations

There are four representations rather than one.

### 1. GVHMR prediction

`estimate_pose_gvhmr.py` consumes GVHMR SMPL-X output. It derives SMPL-24 joints and five mesh-based face points in both a gravity-aligned world frame and camera frame. This stage is rig-independent.

### 2. Landmark stream (`landmarks.json`)

Each frame carries synthesized MediaPipe-33-style points under `world`, camera-space points under `incam`, projected image points, pelvis/root data, and gaze-related data. The world stream is hip-centered and expressed in the performer's normalized frame. It uses estimator coordinates: X lateral, Y down, and negative Z toward the camera. This is the cleanest existing rig-independent boundary.

### 3. Action specification (`action_specs/*.json`)

The action spec is semantic and corrective data layered over the estimate. Useful future Combat Glyph fields already present include:

- `plant` windows: left, right, both, or none (support and airborne state);
- `fists` rise/fall (continuous fist state);
- `root_motion`, `root_source`, `root_scale`, `stage_x`, and `root_offset` (root displacement and stage relationship);
- `rest_blend_start` / `rest_blend` (animation phase boundaries);
- `head_look`, gaze settings, and body basis (orientation);
- per-limb correction windows (`arm_follow`, `arm_pose`, `leg_pose`, `reach`, overrides);
- two specs' shared stage scale and `compare_pair`/mesh-contact results (attacker/defender geometry and collision evidence).

The spec does not currently declare interaction roles, intended contacts, velocity, or named animation phases as first-class fields. These should remain optional metadata on canonical frames/clips, not dependencies of retargeting.

### 4. Lifted joint stream (`joints_mixamo.json`)

`lift_to_mixamo.py` writes per-frame rig-sized joint positions, `basis_x/y/z`, time/frame, rest amount, fist amount, support state, pelvis height, gaze, and head controls. This is the direct input to Blender FK.

Although its keys are mostly anatomical (`hips`, `l_elbow`, and so on), it is already Mixamo-specific because it:

- converts to Mixamo world coordinates in `mp_to_mix`;
- uses Y Bot or Mixamo rig-profile rest positions and lengths;
- reconstructs every chain at those lengths;
- blends toward the target Mixamo rest pose;
- applies stage/root offsets in the target coordinate convention;
- encodes shoulder, hand, foot, and toe points shaped around the Mixamo chain.

## Exact Mixamo-specific components

### `pipeline/setup_rig.py`

Fully Mixamo-specific. It imports FBX, requires `mixamorig:` names and `mixamorig:Hips`, expects the imported armature's X rotation to be 90 degrees and scale to be 0.01, expects roughly 65 bones, assumes a T-pose, maps semantic joints to Mixamo bones, measures Mixamo bone lengths, and writes the rig profile.

### `pipeline/setup_duo.py`

Fully Mixamo-specific scene assembly and validation. It preserves the Mixamo bone namespace, applies the same object transform assumptions, creates `Armature_<Name>`, and measures each character with `setup_rig` logic.

### `pipeline/lift_to_mixamo.py`

Mixed responsibility and the main extraction target. Rig-independent work includes resampling, smoothing, support scheduling, semantic correction windows, gaze propagation, source-root selection, and phase metadata. Mixamo-specific work includes:

- hard-coded Y Bot `REST` and `LEN` fallbacks;
- Mixamo coordinate conversion and facing convention;
- reconstruction directly at Mixamo proportions;
- Mixamo rest-pose blending;
- Mixamo-character scaling for arm follow and root travel;
- foot/hand/toe endpoint construction based on the target profile.

This file is where reconstructed human motion first becomes target-character motion.

### `pipeline/apply_mixamo_fk.py`

Fully Mixamo-specific adapter plus reusable Blender mechanics. Specific assumptions include:

- literal `mixamorig:*` bone names in `AIM`, fingers, hips, feet, head, measurements, rendering, and contact paths;
- armature object default name `Armature`;
- bone local +Y is the aim axis;
- hard-coded Mixamo finger quaternion curls and axes;
- hips-only translation with centimeter pose-bone channels under an armature scaled 0.01;
- a particular hips basis/twist solution;
- Mixamo foot flattening and ground/hip defaults;
- FK-only application and no constraints;
- Mixamo head face axis assumptions;
- action keying and curve dumping by Mixamo names.

Reusable mechanics are action creation, quaternion key insertion, linear interpolation, frame iteration, scene output, and mesh BVH contact testing, once parameterized by an adapter.

### Validation and comparison

- `qa_clip.py` hard-codes Mixamo hips, hands, feet, and head names and Y Bot rest fallbacks.
- `compare_reference.py` hard-codes Mixamo anatomical proxies and head/shoulder conventions.
- `compare_pair.py` hard-codes Mixamo striker and limb-segment bone names.
- `render_preview.py` assumes armature/action bindings and labels output as Mixamo; camera fitting from curves is otherwise reusable.
- `run_in_blender.py` imports and invokes `apply_mixamo_fk` directly.

### Unmodified/mostly unmodified components

- `estimate_pose_gvhmr.py`: keep unchanged for the first milestone.
- `analyze_landmarks.py`: keep unchanged; it reads reconstructed anatomy, not rig bones.
- `blender_exec.py`: keep unchanged; it is transport.
- GVHMR, SMPL-X, NumPy/SciPy/OpenCV/PyTorch: already rig-independent.
- action specs: preserve schema and behavior; only add optional semantic metadata later.
- mesh BVH contact algorithm: keep, but move armature/mesh selection behind a rig adapter.

## Canonical humanoid layer

Use full, rig-neutral names and never store Mixamo or VRM names in canonical data:

```text
hips, spine, chest, upperChest?, neck, head,
leftShoulder?, leftUpperArm, leftLowerArm, leftHand,
rightShoulder?, rightUpperArm, rightLowerArm, rightHand,
leftUpperLeg, leftLowerLeg, leftFoot, leftToes?,
rightUpperLeg, rightLowerLeg, rightFoot, rightToes?
```

The minimum canonical clip should contain:

- version, source FPS, output FPS, duration, and frame count;
- coordinate-system declaration (right-handed, metres, explicit up/forward axes);
- per-frame semantic joint positions before target proportions are imposed;
- per-frame hips position and orthonormal body basis;
- optional gaze direction;
- semantic state: support limb(s), airborne/contact state, fist amounts, rest/phase amount;
- optional root trajectory, velocity, orientation, interaction target/role, intended contact, and collision annotations;
- provenance back to source frames and action-spec corrections.

Recommended canonical coordinates are Blender-compatible right-handed metres with +Z up and a declared forward axis. The choice matters less than making it explicit and testing every adapter against it.

The initial extraction should preserve exact Mixamo output: implement the canonical producer using the current math, then make `MixamoRigAdapter` consume it and prove byte/numerical equivalence within tolerances before changing semantics or coordinates further.

## Adapter contract

A rig adapter should own:

- semantic bone-to-rig-bone mapping;
- required/optional bone validation;
- armature discovery;
- rest-pose measurement and target proportions;
- canonical-to-armature coordinate conversion;
- per-bone primary axis and roll/orientation correction;
- hips/root translation units and object-transform handling;
- FK application, foot orientation, fingers, and head/face orientation;
- rig-aware curve/measurement lookup for validation and export.

It should not own source video processing, GVHMR inference, beat analysis, semantic support schedules, or source-motion corrections.

## VRM mapping requirements

VRM mapping must come from humanoid metadata, not guessed node names. A VRM importer should resolve humanoid bone assignments to Blender pose bones and map them to canonical semantics.

Required first-proof mappings:

| Canonical | VRM humanoid role |
|---|---|
| hips | hips |
| spine | spine |
| chest | chest (fall back carefully if omitted) |
| upperChest | upperChest (optional) |
| neck | neck |
| head | head |
| left/rightUpperArm | left/rightUpperArm |
| left/rightLowerArm | left/rightLowerArm |
| left/rightHand | left/rightHand |
| left/rightUpperLeg | left/rightUpperLeg |
| left/rightLowerLeg | left/rightLowerLeg |
| left/rightFoot | left/rightFoot |
| left/rightToes | left/rightToes (optional) |

Shoulders, eyes, jaw, and fingers are optional extensions. The first proof should animate open hands and defer VRM finger curls; the existing Mixamo quaternion curls cannot safely be reused because local axes and rolls differ.

The adapter must validate hierarchy as well as presence. Missing `upperChest`, shoulders, or toes should collapse the canonical chain deliberately rather than shifting every downstream mapping.

## Rest-pose and coordinate risks

1. **T-pose versus A-pose.** Mixamo input is explicitly T-pose. VRM humanoids are commonly authored with different arm angles. A direction-only bone aim can lose twist and produce shoulder/hand roll errors unless canonical rotations are expressed relative to each rig's measured bind/rest transforms.
2. **Bone roll and primary axis.** Current FK assumes pose-bone +Y points down the bone. Imported VRM bones may use different local orientation/roll. Compute each bone's rest direction and a rest-space correction quaternion instead of assuming an axis.
3. **World and object transforms.** Mixamo relies on X=90 degrees and scale 0.01. VRM importers commonly produce metre-scale objects with different object transforms. All adapter math must explicitly cross world, armature, rest, and pose spaces.
4. **Forward/up convention.** The estimator, lifted Mixamo stream, Blender, and VRM metadata use different conventions. A declared canonical basis plus adapter conversion is mandatory; silent axis swaps will mirror limbs or reverse travel.
5. **Hips/root semantics.** VRM may include nodes above hips or importer-generated roots. Translation must target the semantic hips/root policy without double-applying object motion.
6. **Scale and ground.** Measure rest hips, ankle/sole/toe positions, and leg lengths from the imported VRM. Mesh soles may sit below the foot bone. Ground calibration cannot assume the Y Bot ankle offset.
7. **Twist bones and helper bones.** VRM meshes may include twist/helper bones not named in humanoid metadata. Driving only humanoid bones should allow skinning helpers to inherit naturally; directly keying helpers would reduce portability.
8. **Upper-body distribution.** Optional chest/upperChest changes how torso rotation should be distributed. Copying the Mixamo chain one-for-one can over-rotate a shorter spine chain.
9. **Hands/head.** Hand roll and face-forward axes are especially visible and importer-dependent. Derive them from rest transforms and humanoid metadata, with explicit adapter overrides only when necessary.
10. **Foot planting.** The current solver changes hips height and flattens Mixamo feet. Preserve the semantic plant state, but implement target-specific sole height and foot orientation.

## Components requiring modification

Minimum architecture work after Phase 0:

1. Extract rig-neutral frame/schema types and serialization from `lift_to_mixamo.py`.
2. Split semantic motion processing from target-proportion reconstruction.
3. Wrap current lift/apply behavior as a Mixamo adapter with regression fixtures.
4. Parameterize Blender application by semantic mapping, rest transforms, axes, units, and armature resolver.
5. Add VRM humanoid metadata discovery/profile generation.
6. Make QA/comparison consume semantic measurement names through the adapter or a normalized curve dump.
7. Route Blender execution by adapter type while preserving old commands/defaults.
8. Generalize preview labels and armature selection.

No first-proof changes are needed in GVHMR inference, landmark analysis, source tracking, action timing, smoothing, or socket transport.

## Minimum VRM proof of concept

After the Mixamo baseline passes:

1. Freeze its source video, landmarks, action spec, lifted output, QA logs, and preview as regression fixtures (or hashes/metrics where assets cannot be committed).
2. Import one known-good VRM using a Blender/VRM importer compatible with Blender 5.1.
3. Read its humanoid metadata and write a VRM rig profile: bone mapping, hierarchy, rest matrices, bone directions, scale, hips height, sole/ground offsets.
4. Run the same canonical motion stream into both adapters.
5. For the first VRM pass, animate hips, torso, head, arms, legs, feet, and optional toes; leave fingers open.
6. Render synchronized front/side previews and compare the two characters at identical frames.
7. Validate finite/stable quaternions, root travel, ground alignment, timing, joint continuity, and semantic support-foot height. Visually inspect limb orientation and hand roll.

Acceptance for this proof is semantic equivalence, not polish: both rigs perform the same movement on the same timing, with correct limb sides, plausible joint orientation, root translation, and ground alignment, and without explosions or frame-to-frame instability.

## Estimated implementation phases

Effort estimates assume the baseline assets and one redistribution-safe test VRM are available.

| Phase | Deliverable | Estimate |
|---|---|---:|
| 0 | Reproduce and archive Mixamo baseline | 0.5-2 days plus dependency/checkpoint download time |
| 1 | Golden metrics and Mixamo-assumption tests | 1-2 days |
| 2 | Canonical schema/producer and unchanged Mixamo adapter | 3-5 days |
| 3 | VRM import/profile/mapping and rest-space FK prototype | 4-7 days |
| 4 | Ground/root/foot/head/hand stabilization and semantic QA | 3-6 days |
| 5 | Single-character jab, side kick, and roundhouse validation | 2-4 days |
| 6 | Hip throw and two-character semantic/contact validation | 5-10 days |
| 7 | Optional Combat Glyph export seam and richer interaction metadata | 2-5 days |

Main feasibility risk is not bone naming; VRM humanoid metadata solves most naming. The hard part is bind/rest-space rotation retargeting across A/T poses and different bone rolls while preserving feet, root motion, and hand orientation. The existing position-based reconstruction, semantic support schedule, measured rig profiles, and Blender FK machinery provide a strong foundation.

## Milestone result (2026-08-24)

The immediate gate below was satisfied. The `baseline_dance` plate completed
GVHMR reconstruction and the unchanged Mixamo path with automated QA PASS. A
native VRM 1.0 adapter was then implemented in `pipeline/apply_vrm_fk.py`.

Verified results:

- the same 240-frame motion stream animated two different local VRM 1.0
  humanoids through metadata-driven bone discovery;
- both preserved their authored proportions and exact rest-pose holds;
- explicit left/right support windows measured less than 0.006 mm vertical
  ground error and less than 0.001 mm horizontal foot wander;
- the adapter wrote measured profiles, animated Blender files, QA stills, and
  H.264 previews; and
- VRM 0.x was detected and rejected explicitly rather than mapped by guessed
  names.

This advances the feasibility conclusion from architectural analysis to a
working experimental VRM 1.0 adapter. It does not yet establish broad VRM
compatibility or complete the canonical-schema extraction.

## Original immediate gate to proceed (satisfied)

Provide or place locally (untracked if licensing requires):

- one locked-camera source plate with T-pose bookends;
- the matching action spec, or permission to author it after landmark analysis;
- one standard Mixamo T-pose FBX;
- GVHMR checkout/environment/checkpoints and `SMPLX_NEUTRAL.npz`, or a known-good generated `landmarks.json` for that plate;
- one known-good VRM permitted for local testing.

With a precomputed `landmarks.json`, Phase 0 can skip GPU inference but still exercise the crucial lift, Blender FK, QA, comparison, and render boundary. Without these inputs, implementing even a synthetic VRM test would violate the stated requirement to protect upstream behavior before changing architecture.
