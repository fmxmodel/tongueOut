# Governance Manifest — Image → ARKit Avatar → GLB agent fleet

> The constitution for the Claude Code agents that execute
> `Revised_Image_to_ARKit_Avatar_GLB_Plan.md`. It defines who does what, on which model,
> with what authority, under which gates. The orchestrator (`/execute-all`) and every
> subagent are bound by this document.

---

## 1. Governing principles

1. **Three separate problems.** Reconstruction (build the head), rigging (add the 52 ARKit
   blendshapes), and driving (animate them) are distinct and owned by distinct agents. No
   agent reaches across a seam it doesn't own.
2. **Names are contracts.** The 52 ARKit strings are simultaneously glTF morph-target names,
   MediaPipe `categoryName`s, and three.js driver keys. No agent may rename a shape to paper
   over a mismatch — naming breaks route back to the owner.
3. **Topology is sacred.** Every mesh shares one topology (`faces.npy`). The moment that
   breaks, the pipeline stops; it is not patched downstream.
4. **Silent failure is the enemy.** This whole plan exists to prevent artifacts that look
   done but are subtly wrong. Success is proven by measurement, never assumed.
5. **Licensing, not code, is the constraint.** A commercial artifact ships only when every
   component is independently license-cleared.

---

## 2. Model assignment — "hard part → Fable, the rest → Opus 4.8"

Model tiers are assigned by **difficulty**, per the directive. The plan itself names the two
hard stages: reconstruction ("the hard, separate part") and rigging ("where most of the real
work in this plan lives"). Those get **Fable**; everything else gets **Opus 4.8**.

| Agent | Role | Model | Why this tier |
|---|---|---|---|
| `face-reconstructor` | Reconstruction (Track A/B1) | **`claude-fable-5`** | HARD: adapt to FaceVerse v4's under-documented API / FLAME fitting; photo-faithful, topology-locked output. |
| `arkit-rigger` | Rigging → 52 ARKit deltas | **`claude-fable-5`** | HARD: the connective tissue — expression basis → exact 52 ARKit deltas, same topology, naming discipline. |
| `env-provisioner` | Toolchain setup | `claude-opus-4-8` | Mechanical, idempotent provisioning. |
| `blender-glb-builder` | Shape keys + GLB export | `claude-opus-4-8` | Mechanical given topology-consistent input. |
| `viewer-driver` | three.js + MediaPipe driving | `claude-opus-4-8` | Well-trodden web integration; name-lookup driver. |
| `license-compliance` | Licensing gate | `claude-opus-4-8` | Reasoning + verification, not research-hard coding. |
| `qa-verifier` | Anti-silent-failure gate | `claude-opus-4-8` | Adversarial measurement of invariants. |
| `metahuman-route` | Track B2 (MetaHuman→GLB) | `claude-opus-4-8` | Tool orchestration (UE 5.7 + Blender); EULA deferred to compliance. |
| orchestrator (`/execute-all`) | Sequencing + gates | `claude-opus-4-8` | Coordination, not stage work. |

*Model-field notes:* Claude Code's `model:` frontmatter accepts full IDs (used here for
precision) and the aliases `fable` / `opus`. If a full ID is ever rejected in your version,
fall back to the aliases `fable` (hard stages) and `opus` (the rest) — the intent is
pinned by role, not by string.

---

## 3. Permissions & the bypass-permissions directive

- The project runs under **`permissions.defaultMode: "bypassPermissions"`**
  (`.claude/settings.json`), and every agent carries **`permissionMode: bypassPermissions`**.
- Unattended runs launch with **`claude --dangerously-skip-permissions`**. Three layers
  (settings default mode + per-agent mode + launch flag) make the bypass robust across
  versions.
- **Bypass is not lawlessness.** `deny` rules in settings — secrets (`.env`, `~/.ssh`,
  `~/.aws`), catastrophic filesystem ops (`rm -rf /`, `sudo rm`), pipe-to-shell — are
  **always enforced** and are *not* bypassed. `git push` still routes through `ask`.
- Bypass speeds *tool execution*. It does **not** bypass the two governance gates (§4).
- Every `Write`/`Edit` is appended to `.claude/audit.log` via a PostToolUse hook; a
  SessionStart hook restates this governance on every launch.

---

## 4. The two hard gates (cannot be bypassed)

**Gate 1 — Licensing (`license-compliance`, blocking authority).**
- On a **commercial** run (B1/B2) it BLOCKS if the reconstruction base is FaceVerse or
  standard FLAME, if any fitter/reference asset is uncleared, or if the MetaHuman EULA
  question for standalone-GLB shipping is unresolved.
- Emits `out/compliance_report.md` ending in `SHIP-CLEARED: yes|no`. A `no` is a hard stop
  for commercial shipping. Track A is permitted but stamped **internal-use only**.
- EULA-scope judgments (Epic MetaHuman) are explicitly a **human-lawyer** decision; no agent
  self-clears them.

**Gate 2 — QA (`qa-verifier`, read-only, adversarial).**
- Runs after every stage and as final acceptance. Confirms: faces exist (N>0), identical
  topology across all 52 meshes, all 52 ARKit names present + exactly spelled, GLB carries
  named morph targets, MediaPipe categories resolve.
- Emits `out/qa_report.md` ending in `ACCEPT: yes|no`. `no` blocks completion.

**Done means:** Track A → `ACCEPT: yes` (internal-only). Commercial → `ACCEPT: yes` **and**
`SHIP-CLEARED: yes`, with attribution/EULA obligations satisfied.

---

## 5. Escalation & routing

- A QA FAIL routes back **by name** to the owning agent (e.g. topology drift → `arkit-rigger`;
  vertex-only mesh → `face-reconstructor`; missing morph target → `blender-glb-builder`).
- A licensing block routes to `face-reconstructor`/`metahuman-route` to switch track/base.
- A blocker an agent cannot resolve (no GPU, no UE 5.7, EULA scope) is surfaced to the user
  by the orchestrator — never faked or worked around.
- No agent silently downgrades scope (e.g. shipping FaceVerse output commercially, or
  fabricating an unsupported blendshape). Honesty over a green checkmark.

---

## 6. Data / artifact contract (the shared bus)

```
out/
├── env_report.md          # env-provisioner   → PASS/FAIL per tool
├── recon/                 # face-reconstructor → neutral.ply, id_params.npz, faces.npy,
│                          #                       exp_name_list.json (or FLAME basis), recon_report.md
├── shapes/                # arkit-rigger       → neutral.ply, expr_<arkitName>.ply×N,
│                          #                       arkit_manifest.json, rig_report.md
├── head_arkit.glb         # blender-glb-builder → final GLB, 52 named morph targets
├── glb_report.md          # blender-glb-builder
├── viewer/                # viewer-driver      → three.js + MediaPipe app
├── viewer_report.md       # viewer-driver
├── compliance_report.md   # license-compliance → SHIP-CLEARED: yes|no
└── qa_report.md           # qa-verifier        → ACCEPT: yes|no
```

Each agent reads its predecessor's artifacts, never reaches around them, and writes its own
report so the next gate can verify independently.
