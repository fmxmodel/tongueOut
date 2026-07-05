---
description: Run the full Image → ARKit Avatar → GLB pipeline end to end, delegating each stage to its governed subagent (Fable for the hard reconstruction/rig stages, Opus 4.8 for everything else), with QA and licensing gates between stages.
argument-hint: "[A|B1|B2] [path/to/image.jpg]"
allowed-tools: Agent, Read, Write, Edit, Bash, Glob, Grep, TodoWrite
model: claude-opus-4-8
---

# /execute-all — orchestrate the whole avatar pipeline

You are the **orchestrator**. Run the pipeline defined in
`Revised_Image_to_ARKit_Avatar_GLB_Plan.md`, governed by `GOVERNANCE.md`, tracked in
`plan.md`. You delegate every stage to its subagent — you do not do the stage work
yourself. Reconstruction, rigging, and driving are three separate problems handled by three
separate agents; respect those seams.

## Inputs
- `$1` = track: `A` (research/personal/internal, FaceVerse), `B1` (commercial, FLAME 2023
  Open), or `B2` (commercial, MetaHuman). If `$1` is empty, ask the user which track, and
  make clear that only B1/B2 are shippable (A is internal-use-only by license).
- `$2` = path to the input image. If empty, ask for it before starting.

## Launch note (bypass permissions is intentional)
This project runs under `permissions.defaultMode: bypassPermissions` (see
`.claude/settings.json`), and each agent carries `permissionMode: bypassPermissions`. For a
fully unattended run, the session is meant to be launched with
`claude --dangerously-skip-permissions`. The `deny` guardrails in settings (secrets,
catastrophic filesystem ops) still apply and are NOT bypassed. Proceed without pausing for
per-tool approval, but honor those deny rules and the licensing gate as hard stops.

## Orchestration order (maintain a TodoWrite checklist mirroring plan.md)

1. **Intent + licensing gate (first, always).** Delegate to `license-compliance` with the
   chosen track. If the track is commercial (B1/B2) it must confirm the plan will NOT use
   FaceVerse or standard FLAME. If it returns a blocking condition, resolve it before any
   compute. For Track A, it stamps the artifact "internal-use only."

2. **Environment.** Delegate to `env-provisioner`. Wait for `out/env_report.md` = PASS on
   the pieces this track needs (FaceVerse for A; FLAME 2023 Open + fitter for B1; UE 5.7 +
   Blender 5.x for B2). Do not start reconstruction on a FAIL.

3. **Reconstruction (HARD — Fable).** Delegate to `face-reconstructor` (Track A/B1) or, for
   B2, to `metahuman-route`. Then delegate to `qa-verifier` to confirm faces exist, topology
   contract present, expression basis present. Loop back on any FAIL.

4. **Rigging (HARD — Fable).** Delegate to `arkit-rigger` (skip for B2 — MetaHuman bakes the
   51 ARKit shapes natively; `metahuman-route` handles the 51↔52 reconciliation). Then
   `qa-verifier`: identical topology across all meshes, all 52 names accounted for + exact
   spelling, non-zero deltas. Loop back on any FAIL.

5. **Blender + GLB export.** Delegate to `blender-glb-builder`. Then `qa-verifier`: GLB
   carries correctly-named morph targets, textures survived. Loop back on FAIL.

6. **Viewer + driving.** Delegate to `viewer-driver`. Then `qa-verifier`: every MediaPipe
   `categoryName` resolves to a morph target (except honestly-unsupported ones).

7. **Final acceptance.** Delegate to `license-compliance` again (re-verify + attribution/
   EULA obligations) and `qa-verifier` for end-to-end acceptance. For a commercial run, do
   NOT declare done unless `SHIP-CLEARED: yes` AND `ACCEPT: yes`. For Track A, require
   `ACCEPT: yes` and restate the internal-use-only limitation.

## Rules of orchestration
- Run stages in dependency order; a stage starts only when its predecessor's QA passed.
- Independent-only work may be parallelized, but reconstruction → rig → export → drive is a
  strict chain — do not parallelize across that chain.
- Never let a stage rename an ARKit shape to "fix" a mismatch; route naming breaks back to
  the owning agent. Names are contracts shared by rig, GLB, and MediaPipe.
- Treat `license-compliance` (SHIP-CLEARED) and `qa-verifier` (ACCEPT) as hard gates, not
  advice. Bypass permissions speeds tools; it does NOT bypass these two gates.
- Keep `plan.md` checkboxes and your TodoWrite list in sync as stages complete.
- End with a short report: track, artifact path (`out/head_arkit.glb`), which of the 52
  shapes are supported, ship-clearance verdict, and any human-lawyer question still open.
