# Native VRM adapter

The experimental VRM adapter applies reconstructed semantic motion directly to
a VRM 1.0 humanoid. It does not read or convert a Mixamo Blender action.

## Status

- VRM 1.0 (`VRMC_vrm`) is supported.
- VRM 0.x is detected and rejected with an explicit error for now.
- Two locally supplied VRM 1.0 avatars with different proportions have
  completed the 240-frame `baseline_dance` test.
- The existing Mixamo path is unchanged.

The current input is `joints_out` from an action spec. Its fields are semantic
human joints (`hips`, `l_elbow`, `r_knee`, and so on), but the file is still
named `joints_mixamo.json` by the upstream lift. Treat that filename as a
transitional boundary; the adapter never reads Mixamo bone names or animation
curves from it.

## Run

From a normal terminal:

```powershell
python pipeline\run_vrm_adapter.py `
  --spec action_specs\baseline_dance.json `
  --vrm "assets\vrm\My Avatar.vrm" `
  --out-dir clips\baseline_dance_vrm
```

If Blender is not installed at the documented Windows path, pass
`--blender C:\path\to\blender.exe` or set `BLENDER_EXE`.

The equivalent direct Blender command is:

```powershell
blender --background --python pipeline\apply_vrm_fk.py -- `
  --spec action_specs\baseline_dance.json `
  --vrm "assets\vrm\My Avatar.vrm" `
  --out-dir clips\baseline_dance_vrm
```

Useful options:

- `--profile-out`: location for the measured VRM rig profile;
- `--blend-out`: explicit animated `.blend` destination;
- `--action-name`: Blender action name;
- `--qa-frames 1,84,114,240`: destination frames rendered from front and side;
- `--no-stills`: skip QA rendering.

## Outputs

The output directory contains:

- `<motion>_vrm_animated.blend`: animated VRM scene;
- `vrm_rig_profile.json`: humanoid mapping and measured rest geometry;
- `vrm_adapter_result.json`: inputs, action, scale, outputs, and contact metrics;
- `poses/front_*.png` and `poses/side_*.png`: QA stills unless disabled.

VRM files and generated `.blend` files remain local and are ignored by Git.

## Render an MP4

With `ffmpeg` installed or supplied explicitly:

```powershell
python pipeline\render_vrm_video.py `
  --blend clips\baseline_dance_vrm\baseline_dance_vrm_animated.blend `
  --profile clips\baseline_dance_vrm\vrm_rig_profile.json `
  --out clips\baseline_dance_vrm\baseline_dance_vrm.mp4
```

Use `--ffmpeg C:\path\to\ffmpeg.exe` when it is not on `PATH`. The renderer
automatically frames the full authored humanoid at 1280x720, renders a PNG
sequence through Blender Workbench, and encodes H.264/yuv420p at 30 fps.

## Adapter behavior

1. Parse `VRMC_vrm.humanoid.humanBones` from the VRM's GLB metadata.
2. Import the VRM with Blender's glTF importer.
3. Validate required humanoid roles and resolve metadata nodes to pose bones.
4. Measure hips, feet, semantic joint heads, anatomical rest directions, and
   authored foot pitch.
5. Derive scale from the motion stream's strongest rest frame rather than a
   hard-coded Mixamo character.
6. Apply per-segment motion directions to the VRM's own bone lengths.
7. Preserve authored hand roll and exact rest-pose holds.
8. Ground the semantic support foot and lock it in X/Y during explicit
   single-support windows.
9. Key a 30 fps Blender action and report contact validation.

## Compatibility and known limits

- The adapter requires a valid VRM 1.0 humanoid mapping.
- Optional `upperChest`, shoulder, and toe roles are used when available.
- Finger curls are not transferred yet; fingers remain in the authored pose.
- Direction-only joints cannot fully reconstruct forearm twist or palm facing.
- Contact transitions, chest/head distribution, root travel, and airborne
  motion need broader validation.
- VRM spring bones and expressions import with the model but are not animated
  by this body-motion adapter.
- Combat and two-character interaction remain future validation milestones.

## Why this is native VRM retargeting

The branch point is reconstructed motion:

```text
Reconstructed semantic motion
        ├── Mixamo adapter
        └── VRM adapter
```

There is no `Mixamo action -> VRM` conversion stage. The Mixamo and VRM targets
are independent consumers of the same reconstructed movement.
