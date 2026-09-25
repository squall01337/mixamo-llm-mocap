"""Old pipeline vs new, on a clip you already signed off — lift, solve, QA
and compare in one command, no Blender, no GPU.

  python pipeline\\ab_test.py --spec action_specs\\spin.json
  python pipeline\\ab_test.py --spec action_specs\\spin.json --landmarks plates\\spin\\landmarks_v2.json
  python pipeline\\ab_test.py --spec action_specs\\duel_ybot.json --spec action_specs\\duel_ninja.json

Every spec is copied twice, under clips/ab/<name>/:

  old/<spec>/spec.json   "torso": "rigid", "pin": "single" — the single torso
                         frame and the support-ankle-only pin the clip was
                         signed off with (the lift output is identical to the
                         lift before these modes existed)
  new/<spec>/spec.json   "torso": "twist", "pin": "contacts" — the defaults now

Both copies keep every corrector of the original spec (reach, boost, smooth,
arm_follow, ...), so what differs is the pipeline, not the tuning. Both read
the same landmarks: the spec's own, or --landmarks (one per --spec, in the
same order) — a re-estimate that also carries the estimator's `body` and
`static` blocks, which only the new modes read. Each copy has its own
clip_dir, joints_out and action_name, so nothing the original spec produced
is touched, and both go straight to `run_in_blender.py all` and
`render_preview.py --showcase` for the side-by-side look.

Each copy runs lift -> fk_solve -> qa_clip -> compare_reference (and, for
two specs, compare_pair); the reports are saved beside the copies. The
script prints the two reports' numbers side by side and the frame windows
where the two motions differ most — the beats to look at first.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PIPELINE = REPO / "pipeline"

VARIANTS = {"old": {"torso": "rigid", "pin": "single"},
            "new": {"torso": "twist", "pin": "contacts"}}
STAGES = [("lift", "lift_to_mixamo.py"), ("solve", "fk_solve.py"),
          ("qa", "qa_clip.py"), ("compare", "compare_reference.py")]
# The bones whose world position is what a viewer reads in a pose.
WATCH = ["Head", "Spine2", "LeftHand", "RightHand", "LeftForeArm", "RightForeArm",
         "LeftFoot", "RightFoot", "LeftLeg", "RightLeg", "LeftToeBase", "RightToeBase"]
DIFF_TOL = 0.03     # m: a difference between the two motions worth a look
BEAT_SEP = 15       # dest frames (half a second) between two listed beats
TOP = 8             # beats listed per character
MISSING = {"body": "the `body` block (spine, collars, knuckles, toe tips)",
           "static": "the `static` block (the estimator's contact confidence)"}


def rpath(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def rel(p: Path) -> str:
    """Repo-relative with forward slashes (what the specs use), else absolute."""
    p = Path(p).resolve()
    try:
        return p.relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def run(script: str, args: list, log: Path) -> tuple[bool, str]:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, str(PIPELINE / script), *args], cwd=REPO, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    text = r.stdout + (("\n" + r.stderr) if r.stderr.strip() else "")
    log.write_text(text, encoding="utf-8")
    return r.returncode == 0, text


def landmark_blocks(path: Path) -> list:
    frames = json.loads(path.read_text(encoding="utf-8"))["frames"]
    return [k for k in ("body", "static") if frames and k in frames[0]]


def qa_numbers(text: str) -> dict:
    lines = text.splitlines()
    verdict = next((ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("verdict:")), "none (crashed?)")
    skate = {m.group(1): float(m.group(2))
             for m in re.finditer(r"grounded skate (\w+): ([\d.]+) m of slide", text)}
    return {"verdict": verdict.split(" (")[0],
            "FAIL": sum(ln.startswith("[FAIL]") for ln in lines),
            "WARN": sum(ln.startswith("[WARN]") for ln in lines),
            **{f"skate {k} (m)": v for k, v in sorted(skate.items())}}


def qa_checks(text: str) -> dict:
    """QA check lines by check name ("grounded skate LeftFoot", "plant both
    dest 1-40 LeftFoot", ...) -> "[TAG] result"."""
    out = {}
    for ln in text.splitlines():
        m = re.match(r"(\[\w+\]) ([^:]+):(.*)", ln.rstrip())
        if m:
            out[" ".join(m.group(2).split())] = f"{m.group(1)}{m.group(3)}"
    return out


def findings(text: str) -> list | None:
    lines = text.splitlines()
    start = next((k for k, ln in enumerate(lines) if ln.startswith("FINDINGS")), None)
    if start is None:
        return None
    out = []
    for ln in lines[start + 1:]:
        if not ln.strip() or ln.startswith("CLOSEST"):
            break
        if ln.lstrip().startswith("src"):
            out.append(ln.strip())
    return out


def where_they_differ(old_dir: Path, new_dir: Path, src_fps: float, dst_fps: float) -> dict:
    """Per frame, the largest world-position gap between the two motions over
    the WATCH bones. The twist torso and the contact pin shift most frames a
    little, so rather than windows this lists the beats where the gap peaks,
    largest first, at least BEAT_SEP frames apart."""
    fo = json.loads((old_dir / "curves.json").read_text(encoding="utf-8"))["frames"]
    fn = json.loads((new_dir / "curves.json").read_text(encoding="utf-8"))["frames"]
    n = min(len(fo), len(fn))
    bones = [b for b in WATCH if f"mixamorig:{b}" in fo[0]["bones"] and f"mixamorig:{b}" in fn[0]["bones"]]

    def track(frames, b):
        return np.array([f["bones"][f"mixamorig:{b}"]["world_location"] for f in frames[:n]], float)

    gap = np.stack([np.linalg.norm(track(fn, b) - track(fo, b), axis=1) for b in bones], axis=1)
    peak, which = gap.max(axis=1), gap.argmax(axis=1)
    dest = np.array([f["frame"] for f in fo[:n]])

    def src(d):
        return (d - 1) * src_fps / dst_fps + 1

    beats = []
    for k in np.argsort(-peak, kind="stable"):
        if peak[k] <= DIFF_TOL or len(beats) == TOP:
            break
        if all(abs(int(k) - b) >= BEAT_SEP for b in beats):
            beats.append(int(k))
    return {"median": float(np.median(peak)), "max": float(peak.max()),
            "frames_over": int((peak > DIFF_TOL).sum()), "frames": n,
            "beats": [{"dest": int(dest[k]), "src": src(dest[k]), "gap": float(peak[k]),
                       "bone": bones[which[k]]} for k in beats]}


def side_by_side(title: str, a: dict, b: dict) -> None:
    print(f"  {title:34s} {'old':>12s} {'new':>12s}")
    for key in list(dict.fromkeys([*a, *b])):
        va, vb = a.get(key, "-"), b.get(key, "-")
        fa = f"{va:.3f}" if isinstance(va, float) else str(va)
        fb = f"{vb:.3f}" if isinstance(vb, float) else str(vb)
        print(f"    {key:32s} {fa:>12s} {fb:>12s}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description="Old pipeline vs new on one clip (see the module docstring).")
    ap.add_argument("--spec", action="append", required=True,
                    help="the clip's spec; twice for a two-character clip")
    ap.add_argument("--landmarks", action="append", default=[],
                    help="landmarks both copies read instead of the spec's own (one per --spec, same order)")
    ap.add_argument("--out", help="output folder (default clips/ab/<first spec's name>)")
    args = ap.parse_args()
    if len(args.spec) > 2:
        raise SystemExit("one spec, or two for a two-character clip")
    if args.landmarks and len(args.landmarks) != len(args.spec):
        raise SystemExit("--landmarks: give one per --spec, in the same order")

    specs = [json.loads(rpath(s).read_text(encoding="utf-8")) for s in args.spec]
    if len({s["name"] for s in specs}) != len(specs):
        raise SystemExit("the two specs share a name")
    out = rpath(args.out) if args.out else REPO / "clips" / "ab" / specs[0]["name"]

    lms = []
    for i, spec in enumerate(specs):
        lm = rpath(args.landmarks[i]) if args.landmarks else rpath(spec["landmarks"])
        if not lm.exists():
            raise SystemExit(f"landmarks not found: {lm}")
        lms.append(lm)
        missing = [MISSING[k] for k in MISSING if k not in landmark_blocks(lm)]
        print(f"{spec['name']}: landmarks {rel(lm)}")
        if missing:
            print(f"  NOTE: these landmarks predate {' and '.join(missing)}, so the new modes run without"
                  " them. Re-estimate the plate with the current estimate_pose_gvhmr.py into a new file"
                  " (the GVHMR cache is reused) and pass it with --landmarks for the full comparison.")

    paths = {v: [] for v in VARIANTS}
    for spec, lm in zip(specs, lms):
        for v, modes in VARIANTS.items():
            d = out / v / spec["name"]
            d.mkdir(parents=True, exist_ok=True)
            s = copy.deepcopy(spec)
            s.update(modes)
            s.update({"landmarks": rel(lm), "clip_dir": rel(d), "joints_out": rel(d / "joints_mixamo.json"),
                      "action_name": f"{spec['action_name']}_{v.upper()}",
                      "_ab_test": f"{v} copy of {spec['name']}, written by pipeline/ab_test.py"})
            p = d / "spec.json"
            p.write_text(json.dumps(s, indent=1), encoding="utf-8")
            paths[v].append(p)

    reports = {v: {} for v in VARIANTS}
    solved, failed = set(), []
    for v in VARIANTS:
        for p, spec in zip(paths[v], specs):
            for stage, script in STAGES:
                ok, text = run(script, ["--spec", rel(p)], p.parent / f"{stage}.txt")
                reports[v][spec["name"], stage] = text
                if not ok:
                    failed.append(f"{v}/{spec['name']}: {script} (log {rel(p.parent / (stage + '.txt'))})")
                    if stage in ("lift", "solve"):
                        break
                if stage == "solve":
                    solved.add((v, spec["name"]))
        if len(specs) == 2 and all((v, s["name"]) in solved for s in specs):
            ok, text = run("compare_pair.py", ["--spec", rel(paths[v][0]), "--spec", rel(paths[v][1])],
                           out / v / "compare_pair.txt")
            reports[v]["pair", "compare_pair"] = text
            if not ok:
                failed.append(f"{v}: compare_pair.py (log {rel(out / v / 'compare_pair.txt')})")

    for spec in specs:
        name = spec["name"]
        print(f"\n=== {name}")
        qa = {v: reports[v].get((name, "qa")) for v in VARIANTS}
        if all(qa.values()):
            side_by_side("QA (qa_clip.py)", qa_numbers(qa["old"]), qa_numbers(qa["new"]))
            # Every check moves a little between the two; list the ones whose
            # verdict changed, and the ground-contact ones the pin is about.
            co, cn = qa_checks(qa["old"]), qa_checks(qa["new"])
            changed = [c for c in dict.fromkeys([*co, *cn]) if co.get(c) != cn.get(c)]
            shown = [c for c in changed if co.get(c, "").split("]")[0] != cn.get(c, "").split("]")[0]
                     or c.startswith(("grounded skate", "plant", "airborne"))]
            if shown:
                print("  QA checks that changed:")
                for c in shown:
                    print(f"    {c}")
                    for v, checks in (("old", co), ("new", cn)):
                        print(f"      {v} {checks.get(c, '(no such check)')}")
            if len(changed) > len(shown):
                print(f"  ({len(changed) - len(shown)} other checks moved without changing verdict: "
                      f"{rel(out / 'old' / name / 'qa.txt')}, {rel(out / 'new' / name / 'qa.txt')})")
        cmp_ = {v: findings(reports[v].get((name, "compare"), "")) for v in VARIANTS}
        if all(c is not None for c in cmp_.values()):
            print(f"  compare_reference.py windows off the video: old {len(cmp_['old'])}, new {len(cmp_['new'])}")
            for v in VARIANTS:
                for ln in cmp_[v]:
                    print(f"    {v}: {ln}")
        if all((v, name) in solved for v in VARIANTS):
            w = where_they_differ(out / "old" / name, out / "new" / name,
                                  float(spec.get("src_fps", 24)), float(spec.get("dst_fps", 30)))
            print(f"  old vs new motion: gap median {w['median']:.3f} m, max {w['max']:.3f} m; "
                  f"{w['frames_over']} of {w['frames']} frames apart by more than {DIFF_TOL:.2f} m")
            if w["beats"]:
                print("  where they differ most (look here first):")
                for x in w["beats"]:
                    print(f"    src {x['src']:5.1f} (dest {x['dest']:3d})  {x['gap']:.3f} m  {x['bone']}")
                frames = ",".join(str(x["dest"]) for x in sorted(w["beats"], key=lambda x: x["dest"]))
                print("  stills at those beats, once both copies are applied:")
                for v in VARIANTS:
                    print(f"    python pipeline\\run_in_blender.py stills {rel(out / v / name / 'spec.json')} "
                          f"--frames {frames}")
    if len(specs) == 2:
        pair = {v: findings(reports[v].get(("pair", "compare_pair"), "")) for v in VARIANTS}
        if all(c is not None for c in pair.values()):
            print(f"\n=== pair: compare_pair.py findings: old {len(pair['old'])}, new {len(pair['new'])}")
            for v in VARIANTS:
                for ln in pair[v]:
                    print(f"    {v}: {ln}")

    print(f"\nreports and spec copies: {rel(out)}")
    if failed:
        print("\nFAILED:\n  " + "\n  ".join(failed))
        sys.exit(1)
    print("next, the look (docs/PIPELINE.md section 12), Blender open on the clip's scene:")
    for v in VARIANTS:
        for p in paths[v]:
            print(f"  python pipeline\\run_in_blender.py all {rel(p)}")
    for v in VARIANTS:
        also = "".join(f" --also {rel(p)}" for p in paths[v][1:])
        print(f"  tools\\GVHMR\\.venv\\Scripts\\python.exe pipeline\\render_preview.py "
              f"{rel(paths[v][0])}{also} --showcase")


if __name__ == "__main__":
    main()
