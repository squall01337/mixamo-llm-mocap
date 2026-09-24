# Pitfalls

Every one of these was paid for in real passes across three model
generations working on this rig. If you are an AI operating this
pipeline: read this file before your first pass and again before
touching a clip a human has partially signed off.

## Space and measurement

1. **Blender IK explodes this rig.** `POSE_IK` on the legs sent feet to
   −81 m. The skeleton is FK-only; plant feet via hip-height search +
   Z-only flattening. If FK plant looks wrong, fix the spec or the
   lift — never "just add IK for the landing".
2. **World-space aim is 100× off.** Composing a pose matrix through
   `arm.matrix_world` (scale 0.01) and assigning it sends hands to
   z ≈ 10 m. Aim in armature centimeters (`aim_bone` does).
3. **`pb.head` lies on a posed rig.** Only
   `(arm.matrix_world @ pb.matrix).to_translation()` after a depsgraph
   update is truth.
4. **Stale location keys survive rewrites.** A failed pass that keyed
   locations on non-hip bones will haunt a rotations-only rewrite.
   `apply_mixamo_fk` deletes and rebuilds the action and re-keys zero
   locations everywhere for this reason.

## Reading the video

5. **Never decide a limb from a still.** Facing the camera,
   viewer-left = character-RIGHT. Three separate incidents: a kick
   built on the wrong leg (full wasted pass), a "left knee" jump that
   was numerically right-knee, a "chambered fist" that was a
   foreshortened punch toward the camera. `analyze_landmarks.py` output
   is the only admissible evidence for the beat sheet.
6. **Gen-video does not follow its prompt.** Prompted left-knee jump →
   right knee; prompted extended front kick → high-knee snap; prompted
   A-pose → the model's own idea of one. Retarget what was filmed;
   never author toward the prompt.
7. **The estimator loses occluded limbs.** An arm behind the torso in
   3/4 view came back as garbage (lateral-back instead of chambered at
   the chest). When a human's read of the video contradicts the
   estimator, the human is right — encode it as a spec `arm_override`
   with ramps, and leave the estimator's other limbs alone.

## Feet and ground

8. **Per-frame hip-centered landmarks make planted feet skate.** The
   support foot wandered 0.30 m during a kick. Fix is structural (the
   lift pins the support ankle's XZ and translates the whole pose;
   pelvis sways over the foot, which is correct physics) — do not
   patch it per-frame in the apply.
9. **A degenerate aim target bends toes randomly.** Aiming ToeBase at
   its own head position is a near-zero-length target. Toe aims need a
   point several cm FORWARD along the foot's own heading.
10. **Foot heading follows the foot/hips, never a world axis.**
    "Straightening" toes toward world −Y on a yawed stance turns them
    outward on screen. Keep estimator XZ; constrain heights only.
11. **Never snap the acting limb to the floor.** A floor snap that kept
    running while a kick lifted pinned the foot for one frame and made
    the next frame pop. The plant schedule exists so ground logic
    knows whose foot is acting.

## Blender / tooling

12. **Viewport screenshots capture black** when the Blender window is
    occluded or minimized. Stills go through a temporary camera +
    Workbench render (`run_stills_render`), always.
13. **The long apply blocked Blender's UI.** 300 frames ≈ 2–4 minutes of
    frozen window while the socket request executed — about 100
    depsgraph evaluations per frame. The apply now solves in numpy and
    writes the keys in bulk (seconds); only `apply --legacy` still
    blocks like that. Results are also written to disk
    (`apply_result.json`) so a dropped socket loses nothing.
14. **mcp SDK 2.x breaks the official Blender MCP server** (it imports
    `mcp.server.fastmcp`, removed in 2.0). Pin `mcp[cli]<2` when
    registering the server.
15. **Stale stills burn passes.** After every re-key, delete and
    re-render the stills before judging anything. An old PNG has
    cost this project at least two decisions.
16. **The live scene holds ONE action — the last one applied.** A
    preview rendered after working on another clip silently shows the
    wrong animation (a source-vs-retarget showcase then looks
    "desynchronized" when it is in fact a different clip).
    `render_preview.py` binds the spec's action before rendering for
    exactly this reason; anything else that renders the scene must do
    the same. So must anything that reads the pose back: `curves`,
    `stills` and `contact` now bind too. Before that, dumping clip A's
    curves after applying clip B silently wrote B's motion (measured on
    a test rig: hands 0.22 m off), and every QA and compare figure
    downstream described the wrong clip.

17. **Head orientation must come from the estimator, or the gaze is a
    lie.** Joint positions cannot describe where a head looks (the head
    joint sits inside the skull), so face landmarks must be read from
    real mesh vertices — synthesizing nose/ears from the torso basis
    makes the ear line *identical* to the shoulder line and locks the
    character's gaze to its chest for the whole clip. Symptoms: the head
    swings wide whenever the body twists, and "head vs chest yaw"
    measures exactly 0.000 at every frame — a number too clean to be
    real. Compounding it: the FK apply aims only the skull axis (+Y), so
    the FACE axis (head local +Z) inherits torso lean and yaw. Fix at
    the source (estimator emits `gaze`), orient the head to it in the
    apply, and let `qa_clip.py` flag a chest-locked head (gaze-vs-chest
    std < 2°) and sky-staring (mean elevation > 12°). Three passes were
    burned "fixing" this downstream before the estimator was suspected.
18. **Smoothing eats strikes.** A `smooth` window wide enough to calm a
    fast flurry (9 frames) also averages away its extension peaks —
    punches visibly shrink. Keep flurry windows narrow (5) and restore
    amplitude with `reach`.
19. **Judge limb height against the HEAD, and check proportions before
    correcting.** The eye reads hands against the face, so a guard that
    measures fine relative to the shoulders can still sit at face level
    on a character whose head/shoulder proportions differ from the
    performer's. Measure both, and split the error: on one clip 4.6 cm
    of a 14.8 cm gap was pure proportion (the character's head sits
    lower on its shoulders) and only the remainder was pose worth
    correcting.
20. **Correct a limb by ROTATING the chain, never by translating its
    joints.** Offsetting elbow/wrist/hand and re-normalizing bone
    lengths is fine for a centimetre or two and destroys the pose at ten:
    the re-normalization drags the hands toward the midline and folds the
    wrists (measured: hand separation 0.25 m -> 0.08 m, wrist bend 7° ->
    123°). A rigid rotation about the shoulder socket cannot distort the
    chain — it preserves elbow bend, wrist alignment and the distance
    between the hands. Express the target in metres (`arm_pose.drop_m` /
    `widen_m`) and solve the angle per frame; a fixed angle over- or
    under-shoots as the arm swings.
21. **A "Hand" bone's world position is the WRIST.** Blender reports a
    bone's head; `mixamorig:*Hand` sits at the wrist, so dividing its
    distance-from-shoulder by (arm+fore+hand) understates extension by
    ~15%. Measure the wrist against arm+fore, or the fingertip bone.

## Process

22. **Beat sheet before any spec.** The 16-pass punch and the 4-pass
    kick differ by exactly this discipline; the two shipped clips took
    1–2 passes each.
23. **One constraint per pass; signed-off regions are frozen.** The
    single historical regression came from "improving" feet while
    polishing something else after the upper body was approved.
24. **Map human notes to the owning system before editing.** "Frames
    18–50, punching side" once meant the *guard*, not the punch — a
    pass was wasted editing the wrong function. Convert the frame range
    to src frames, find the beat, find the spec field; only then edit.
25. **QA passing ≠ done.** Numbers catch explosions, pops, skate and
    drift; they cannot see a wrong silhouette. Play the clip, read the
    stills, ask the human.

## Two characters in one scene

26. **Two performers cannot be compared in the estimator's `world`
    frame.** Each one is normalised so that *their own* frame-0 facing
    becomes forward, so two performers arrive in two frames that differ
    by however much their opening yaws differed. Composing their limbs
    onto a shared root looks reasonable and is quietly wrong: it put a
    kicking foot 0.18 m from the other fighter's skull when the video
    and an independent image-space measurement both said 0.29 m — the
    difference between a clean miss and a foot through a head. Anything
    that measures two people against each other must read the CAMERA
    frame (`incam`), the only frame they actually share.
27. **The camera frame's "up" is not gravity.** Heights read straight
    out of camera space carry the camera's pitch — 4.2° on the duel
    plate, which inflated a head-high kick's peak by 0.14 m and
    manufactured a "the kick is too low" finding that did not exist.
    The estimator emits both a gravity-aligned frame and a camera frame;
    fit the rotation between them and push the world's up axis through
    it, then measure in that.
28. **The world-frame root trajectory drifts; the in-camera one does
    not.** Both describe where the performer stood. Measured on the same
    plate: the gravity-aligned global root under-reported a 0.92 m
    step-in as 0.68 m and left the performer 0.16 m off his mark at the
    closing T-pose, while the in-camera root tracked an independent
    image-space measurement (hip pixels ÷ body pixels) to within 2 cm on
    all 241 frames. Drive `root_motion` from `incam`, and trust only its
    lateral component — depth is the ill-conditioned axis of a
    single-camera fit, and one deep crouch moved it 0.22 m in a frame.
29. **Matched proxies, or the comparison is fiction.** Three separate
    false findings in one session, all from comparing a quantity to a
    different quantity: a performer's *nose* against a character's Head
    bone (which sits at the skull base, ~10 cm behind the face); a
    reference *shoulder line* against `mixamorig:Neck` (5–10 cm higher,
    so the torso capsule reached out and "caught" limbs that missed);
    and limb ENDPOINTS instead of whole segments (a shin can pass
    straight through a skull with both its knee and its toe outside).
    Before believing any cross-domain number, name the anatomical point
    on both sides and check they are the same one.
30. **In-place is a solo assumption.** Mixamo clips are in-place by
    convention and every solo plate here is retargeted that way. Two
    fighters are the opposite case: the distance between them IS the
    scene. On the duel plate the attacker closed 1.95 m → 0.88 m and
    back; in place, every punch lands a metre short of the other
    character. One stage, one scale — the initial separation and both
    performers' travel must share a scale factor or the geometry never
    closes.
31. **Flattening the toe direction is right for a planted foot and
    wrong for a kicking one.** The lift used to project heel→toe onto
    the ground plane, which drags a foot pointed 40° up back to
    horizontal: the toe drops and is thrown forward into whatever the
    foot is aimed at. Grounded feet do not need it — the FK apply
    re-aims anything below 0.20 m along its own heading anyway.
32. **The landmark prefilter eats strikes too.** PITFALL #18 is about
    the spec's `smooth` windows; the same thing happens one stage
    earlier in the lift's default 7-frame Savitzky-Golay prefilter. On a
    roundhouse peaking over ~6 source frames it cost 5° of shin angle
    (0.04 m of foot height). `prefilter_window: 5` costs 1.4°.
33. **Capsule proxies are optimistic; only the MESHES decide.** The
    cheap collision model in `compare_pair.py` (a 0.16 m torso cylinder,
    a 0.11 m head sphere) runs outside Blender and is right about
    separation and reach. It is *not* the character's silhouette: a
    hood, a shoulder, a glove and a boot all live outside those
    surfaces. On the duel plate it scored a kick as clearing by 4 mm
    while the actual meshes intersected across 4 frames with up to 230
    overlapping face pairs — visible as a foot through a head. If a
    human says two characters touch and your numbers say otherwise,
    the numbers are measuring the wrong thing. Run
    `run_in_blender.py contact` (BVH overlap on the evaluated, skinned
    meshes) before believing any clearance. The capsules themselves are no
    longer guessed: rig profiles carry radii measured on each character's
    mesh (95th percentile of each part's skin), limbs included — they used
    to be zero-thickness lines.
34. **Check the WHOLE clip, not the beat you are working on.** Chasing
    the kick, the mesh check was run over frames 165–205 and reported
    it clean. Run over all 301 it found four more intersections nobody
    had looked at — including one at the closing T-pose, in a part of
    the clip that had been "finished" for two passes.
35. **A faithful retarget of two people can still be impossible.** Two
    Mixamo characters are not two humans: their limbs, heads and hands
    are thicker. The duel plate's roundhouse clears the defender's
    guard hand by 0.111 m centre-to-centre — with real forearms that is
    a 2 cm miss, and with these characters' meshes it is a collision.
    The retarget reproduced that 0.111 m to the centimetre and was
    *therefore* wrong on screen. When geometry and fidelity conflict,
    fidelity loses: the showcase has to read clean. Buy the clearance
    with the smallest measured departure, declare it in the spec, and
    let the comparator keep reporting it (it prints
    `[declared root_offset]`) so the cost stays visible instead of
    becoming folklore.
36. **Clearance must come from the stage, not from the poses.** Every
    pose-level lever was tried on this collision and each one moved the
    problem: raising the kick swept the foot through the guard hand;
    lowering the defender dropped his hands into the rising foot;
    curling his spine swung his arms forward into it; leaning him away
    lifted his head into the arc; tucking his chin did nothing because
    the contact was his shoulder. Only distance helped every frame at
    once. Requirements that flip between ADJACENT frames (dest 180
    wanted the foot lower, dest 181 wanted it higher) are the signature
    of a grazing contact that no single pose correction can satisfy.
37. **Ramp a stage offset through motion the character already has.**
    0.20 m of clearance applied while the support foot is planted is
    0.20 m of skate. On this plate the attacker already retreats 0.24 m
    between the punches and the kick, so the offset was ramped across
    exactly those frames (src 116–139) and out again while he resettles
    (162–185): nothing slides, and it reads as making range before a
    head kick — which is what a fighter does.
38. **Fist-into-glove is not a bug.** The punches in this plate land on
    the defender's block, and two rigid hands cannot deform, so the
    contact frames show deep face overlap (1016 pairs at the worst).
    That is the strike landing, exactly as in the video. Do not "fix"
    the frames where contact is the point — separate an intended
    contact from an unintended one by asking what the source video
    does at that frame.

## Caches and blind spots

39. **The estimator's cache was keyed by file name.** GVHMR caches
    detections, keypoints, features and poses per plate NAME, and the
    only staleness test was the frame count. Gen-video models emit
    fixed durations (every plate here is 241 frames), so a plate
    regenerated under the same name reused the previous take's poses,
    with nothing on screen to show it. The cache is now stamped with
    the plate's content hash (`source.sha1`). `--fresh` forces a
    rebuild, and a cache that predates the stamp is adopted once,
    with a note saying so.
40. **QA never measured skate in double support.** The plant check
    reports XZ wander only in single-support windows, which are the
    frames the lift pins. In `both` windows nothing pins either foot,
    and hip-centred landmarks turn every sway of the pelvis over planted
    feet into feet sliding under a still pelvis. A synthetic weight
    shift (feet fixed in the ground truth, pelvis ±8 cm) came out with
    the feet sliding over a 16 cm range, and the verdict was PASS. The
    frames right after a pinned window slide too, while the pin's offset
    decays. `qa_clip.py` now reports grounded skate over the whole clip,
    and the lift pins every planted foot, not only the single-support
    one (`pin: "contacts"`, the default): on the same weight shift the
    feet now hold within 1 mm and the pelvis sways its 16 cm.

## Torso and hands

41. **One torso frame turns the hips with the shoulders.** The lift used to
    build a single torso frame, mostly from the shoulder line, and a
    straight spine along it. A performer turning the shoulders 45 degrees
    over square hips came out with the HIPS turned 39 degrees (the leg
    sockets swung round with them) and the chest 6 degrees short; a
    30-degree spine flexion came out as a 15-degree tilt of the whole
    torso about the hips. Punches live on exactly that hip/shoulder
    separation. The pelvis now follows the hip line, the chest the
    collars, and the estimator's spine joints bend the spine
    (`torso: "twist"`, docs/PIPELINE.md 4.1).
42. **Joint positions cannot say how a hand is turned — the same lesson as
    #17, one limb down.** The hand's direction was extrapolated from the
    forearm and its roll was whatever the shortest-arc aim left: an
    80-degree supination came out 153 degrees wrong. The knuckles on the
    estimator's mesh carry the wrist's real orientation; the lift passes
    the index-to-pinky line and the solver rolls forearm and hand to it.
