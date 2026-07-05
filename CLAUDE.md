# CLAUDE.md — project context for the avatar pipeline

Single image → 3D head → rigged with 52 ARKit blendshapes → exported as GLB → viewable in
three.js and drivable by MediaPipe. This repo is the **Claude Code orchestration layer** that
executes `Revised_Image_to_ARKit_Avatar_GLB_Plan.md`.

## How this project is run
- Entry point: the **`/execute-all`** slash command orchestrates all stages.
- It delegates each stage to a governed **subagent** in `.claude/agents/`.
- Authority, model tiers, gates, and escalation live in **`GOVERNANCE.md`**.
- The live checklist is **`plan.md`**; the full technical spec is the `Revised_...Plan.md`.

## Model policy (per the directive)
- **Hard stages → Fable (`claude-fable-5`)**: `face-reconstructor`, `arkit-rigger`.
- **Everything else → Opus 4.8 (`claude-opus-4-8`)**: env, blender/GLB, viewer, licensing,
  QA, MetaHuman route, and the orchestrator.

## Permissions
Runs under `bypassPermissions` (`.claude/settings.json` + per-agent `permissionMode` +
`--dangerously-skip-permissions` launch). `deny` guardrails (secrets, catastrophic ops) and
the two governance gates (licensing `SHIP-CLEARED`, QA `ACCEPT`) are **never** bypassed.

## Invariants every agent must hold
1. Reconstruction, rigging, driving are **separate problems** — stay in your lane.
2. The 52 ARKit names are **contracts** shared by the rig, the GLB morph targets, and
   MediaPipe — never rename to hide a mismatch; exact spelling, case-sensitive.
3. One **topology** across all meshes (`out/recon/faces.npy`); if it drifts, STOP.
4. Prove success by **measurement** (face counts, name diffs), not by claiming it.
5. **License gates ship**: FaceVerse / standard FLAME are non-commercial; FLAME 2023 Open
   (CC-BY-4.0) is the commercial base; MetaHuman EULA is a human-lawyer question.

## Artifact bus
All stages read/write under `out/` (see `GOVERNANCE.md` §6). Each stage emits a `*_report.md`
so the next gate can verify independently.
