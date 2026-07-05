# MetaHuman Route Report — Track B2 (MetaHuman → GLB)

> Owner: `metahuman-route` (Opus 4.8) · Date: 2026-07-04 · Track: **B2** · Run: **COMMERCIAL**
> Mode this session: **SCAFFOLD-ONLY**. No GPU installed, **no UE/Blender/inference compute
> run, no GLB fabricated**. Input image **deferred** (not provided this session). This document
> is a turnkey runbook for the later Windows+NVIDIA GPU box, plus the ARKit naming contract.
>
> Cross-refs (do not contradict): `out/env_report.md`, `out/gpu_requirements.md`,
> `out/compliance_report.md`. Machine-readable name map: `out/arkit_51_52_map.json`.

**What this repo is (stated precisely, so the review-screenshot error is not repeated):**
`smorchj/metahuman-to-glb` is **NOT** "image → rigged avatar." It takes an **already-built
UE 5.7 `MetaHumanCharacter`** and exports it to a web GLB with 51 ARKit blendshapes. It is the
**back half** of B2. The **front half** (getting a MetaHuman that resembles a photo) is a
separate, non-trivial UE step and is not a one-click photo import.

---

## 1. Environment reality — route is NOT runnable here (deferred to a GPU box)

**Verdict: B2 is NOT runnable on this box. Blocked at the front (no UE/MetaHuman) AND the back
(no UE/Blender/GPU).** Nothing was faked; nothing was run.

| Requirement (from repo + `gpu_requirements.md`) | Present here? | Needed for |
|---|---|---|
| NVIDIA RTX-class GPU, ≥8 GB VRAM, DX12 | **No** (Intel UHD integrated only) | UE 5.7 renderer + RigLogic + Sequencer bake |
| Unreal Engine **5.7** + `UnrealEditor.exe` / `UnrealEditor-Cmd.exe` | **No** | stages 00, 01 (front-half build + back-half UE export) |
| A `.uproject` with the **MetaHumanCharacter** plugin + RigLogic | **No** | opening/assembling the character |
| **A built `MetaHumanCharacter` under `/Game/<Name>/`** | **No** | the actual input to the whole converter |
| **Blender 5.x** + `blender.exe` (glTF exporter + shape keys, both built-in) | **No** | stages 02, 03 |
| Stage launchers are **PowerShell `.ps1`** → `UnrealEditor-Cmd.exe` / `blender.exe` | Windows-native | whole pipeline as shipped |
| `smorchj/metahuman-to-glb` cloned (MIT scripts) | **Yes** — `vendor/metahuman-to-glb` | (documented; no compute) |
| Viewer scaffold (three 0.170.0 + tasks-vision 0.10.14) | **Yes** — `out/viewer/` | Phase 5 (`viewer-driver`) |

- This Linux laptop has **no NVIDIA GPU, no CUDA, no UE 5.7, no Blender 5.x, and no
  MetaHumanCharacter asset**. The `.ps1` launchers target Windows + `C:/Program Files/Epic
  Games/UE_5.7/...` and `C:/Program Files/Blender Foundation/...`.
- **Scope note (do not over-provision):** B2 needs **no PyTorch/CUDA model inference** (unlike
  Track A / B1). The GPU powers Unreal's renderer + RigLogic, not a Python DL stack. Do not
  install torch/CUDA for B2 (per `gpu_requirements.md` §0/§2).
- **Migration target:** Windows 10/11 x64 + NVIDIA RTX (≥8 GB VRAM), UE 5.7, Blender 5.x.
  A Linux GPU box is possible but requires porting the `.ps1` launchers to bash/python and
  re-pathing the two UE stages to Linux `UnrealEditor-Cmd` — non-trivial; budget for it.

**Per the run's intent, if a resembling MetaHuman cannot be produced, prefer the alternatives:**
Track **A (FaceVerse)** or **B1 (FLAME 2023 Open, CC-BY-4.0)**. B1 is the commercially-clearable
default with **no Epic EULA dependency** (see §5 and `out/compliance_report.md`).

---

## 2. Front half — Image → MetaHuman (the honest, non-trivial step)

This is the hard part and it is **owned here**, upstream of the converter. There is **no
one-click "photo → MetaHuman"** for an arbitrary face. The deferred input image feeds this step
on the GPU box; the operator produces a `MetaHumanCharacter` that resembles the subject, saved
under `/Game/<Name>/`. Only then does the back-half converter (§3) run.

**Where the deferred image goes (GPU stage):** drop the source photo at
`vendor/metahuman-to-glb/5.7/native-glb/characters/<id>/source/` (the converter's own per-
character `source/` folder; `bootstrap_character.py` also writes a provenance README there).
It is a **reference for the human doing the sculpt/mesh fit — it is not auto-consumed** by any
script. Record the filename in that folder's README.

**Approach (Epic Mesh-to-MetaHuman / MetaHuman Creator), step by step:**
1. **Get a face mesh of the subject.** Mesh-to-MetaHuman needs geometry, not a bare JPEG:
   - photogrammetry / multi-view capture, or a scan, or
   - a single-image face reconstruction (e.g. a FLAME/DECA-style fit) exported as a neutral
     mesh, or a manual sculpt. **A single arbitrary photo alone is not sufficient** for a
     faithful identity; set this expectation with the operator.
2. In UE 5.7 (MetaHuman plugin enabled): import the mesh, run **Mesh-to-MetaHuman** to fit the
   MetaHuman topology + identity to it (**Identity → Promote Frame → Track → MetaHuman
   Identity → Mesh to MetaHuman**), or start from **MetaHuman Creator** and sculpt toward the
   reference photo. Iterate until the likeness is acceptable against the photo.
3. Materialize a **`MetaHumanCharacter`** asset in the project and save it under
   `/Game/<Name>/` (e.g. `/Game/Ada/MHC_Ada`). This asset — **not the photo** — is the input to
   the back half.
4. **Likeness is a human judgment.** There is no automated identity-match gate in this repo;
   the operator eyeballs the MetaHuman against the photo before spending the back-half compute.

> If step 1–2 cannot yield a usable resembling MetaHuman, **B2 is blocked at the front** —
> report that and route to Track A or B1 rather than shipping a generic MetaHuman.
>
> **AI-restriction caveat (for compliance, not to self-decide):** feeding a user photo through
> Mesh-to-MetaHuman may implicate Epic's rule that MetaHumans "may not be used to train or
> enhance the AI models themselves." This is EULA sub-question 3 in `out/compliance_report.md`.

---

## 3. Back half — MetaHuman → GLB (turnkey commands for the GPU box)

The converter's active pipeline is `5.7/native-glb/` (5 stages). Bakes ARKit shapes **natively**
via UE Sequencer + RigLogic, then transfers them onto the GLB face mesh by **KDTree position
match** in Blender. `browInnerUp` is a single bilateral shape; `tongueOut` is not baked by the
default rig (see §4).

**One-time setup on the GPU box:**
```powershell
cd <repo>\vendor\metahuman-to-glb
cp _config\pipeline.example.yaml _config\pipeline.yaml   # gitignored
# Edit the four placeholders in _config/pipeline.yaml:
#   ue_by_version."5.7".project_path : C:/.../MH.uproject       (the .uproject from §2)
#   ue_by_version."5.7".editor_cmd   : C:/Program Files/Epic Games/UE_5.7/Engine/Binaries/Win64/UnrealEditor-Cmd.exe
#   blender_exe                      : C:/Program Files/Blender Foundation/Blender 5.0/blender.exe
# (RUN.md can auto-detect these by glob if UE/Blender are installed in default locations.)
```

**Bootstrap the character folder (derives `<id>`, `mh_folder`, `output_name` from the asset
path; writes `characters/<id>/manifest.json` with every stage `pending`):**
```powershell
python 5.7\native-glb\tools\bootstrap_character.py --asset /Game/<Name>/MHC_<Name>
# e.g. --asset /Game/Ada/MHC_Ada   -> id=ada, mh_folder=/Game/Ada, output_name=Ada
```

**Run the 5 stages in order (each waits for the previous; `<id>` is the bootstrapped id):**
```powershell
# Stage 00 — UE assemble: build/save the MetaHuman SkeletalMesh + textures under /Game/<Name>/.
#   Launches UnrealEditor.exe (GUI, needs Slate ticks) -unattended; editor must be CLOSED first.
.\5.7\native-glb\stages\00-unreal-assemble\tools\run_assemble.ps1 -Char <id>      # ~1-2 min

# Stage 01 — UE GLB export + Sequencer ARKit bake:
#   UnrealEditor-Cmd.exe -run=pythonscript -AllowCommandletRendering. Emits per-mesh .glb +
#   LS_arkit_full.fbx (Sequencer bake at 24fps, 1 frame per ARKit pose) + arkit_pose_names.json
#   into characters/<id>/01-glb/.
.\5.7\native-glb\stages\01-unreal-glb-export\tools\run_export.ps1  -Char <id>      # ~1-2 min

# Stage 02 — Blender assemble: import the .glb(s), replay LS_arkit_full.fbx, capture each pose's
#   deformed mesh, transfer 51 named ARKit shape keys onto the GLB face mesh by KDTree match;
#   propagate the same-named keys onto eyebrow/beard/mustache cards via k=4 IDW.
#   -> characters/<id>/02-blend/<id>.blend
.\5.7\native-glb\stages\02-blender-assemble\tools\run_assemble.ps1 -Char <id>      # ~60-90 s

# Stage 03 — Blender GLB export: Draco-compressed, ≤1024px textures, ~40 MB.
#   -> characters/<id>/03-glb/<id>.glb  (+ glb_manifest.json, mh_materials.json, textures/)
.\5.7\native-glb\stages\03-export-to-glb\tools\run_export.ps1      -Char <id>      # ~30-60 s

# Stage 04 — three.js viewer build: copies GLB into docs/ and renders a preview.
#   -> docs/characters/<id>/<id>.glb
.\5.7\native-glb\stages\04-webview-build\tools\run_site.ps1        -Char <id>      # ~5 s
```
(Alternatively, one orchestrator entry: follow `5.7/native-glb/RUN.md`, which bootstraps and
dispatches all five stages.)

**Where the GLB lands, and the Phase-5 handoff:**
- Converter output: `vendor/metahuman-to-glb/5.7/native-glb/characters/<id>/03-glb/<id>.glb`
  (canonical, with the 51 ARKit morph targets), also published to
  `.../5.7/native-glb/docs/characters/<id>/<id>.glb`.
- **Phase 5 requires the artifact at `out/head_arkit.glb`.** After stage 03 succeeds on the GPU
  box, copy the canonical GLB into the pipeline bus:
  ```powershell
  Copy-Item vendor\metahuman-to-glb\5.7\native-glb\characters\<id>\03-glb\<id>.glb out\head_arkit.glb
  ```
  Then hand `out/head_arkit.glb` to `viewer-driver` (§3.5 driving is identical) and `qa-verifier`.
  **Not produced this session** (no compute; no GLB fabricated) — this copy step is part of the
  deferred GPU run.

**Known back-half fidelity gaps to carry to `qa-verifier`** (from the repo README): FBX bake
captures bone deformation only — fine morph correctives (lip squash, wrinkle deltas ~5–10 mm)
are lost; `browInnerUp` L/R asymmetry lost; eyelash/eye-occlusion/clothing-colour issues open.

---

## 4. The 51 ↔ 52 ARKit naming contract (the load-bearing part)

This is the **name contract** shared by the GLB morph targets, the MediaPipe driver, and QA.
CLAUDE.md invariant 2: the 52 ARKit names are contracts — **never rename to hide a mismatch;
exact spelling, case-sensitive.** The gap is recorded honestly below and in
`out/arkit_51_52_map.json` (generated by reading the converter's own reference data —
measurement, not transcription; verified `51` baked, `52` canonical, the only missing name is
`tongueOut`, `browInnerUp` present).

**There are TWO distinct facts, and the review framing conflates them. Both are recorded:**

**(a) The numeric 51-vs-52 gap is `tongueOut`, not `browInnerUp`.**
The default MetaHuman face rig does **not** produce a `tongueOut` pose (`tongueOut` "needs a
non-default rig variant" — repo README; and the reference `arkit52_deltas.json` note: "tongueOut
is absent from this reference FBX (51 head keys, not 52)"). So stage 02 bakes **51** of Apple's
**52**; the one omitted is **`tongueOut`**.

**(b) The fold is `browInnerUp`.** MetaHuman's raw facial rig can drive left/right inner-brow
raise independently, but `AS_MetaHuman_ARKit_Mapping` **folds them into the single Apple-standard
`browInnerUp`** (repo README known gap; issues #17/#18). Crucially, **Apple's ARKit-52 is ALSO a
single `browInnerUp`** (Apple never split it L/R), **and MediaPipe emits a single `browInnerUp`**
— so on the *name* contract this resolves **1:1 with no remap**. The only thing lost is left/right
inner-brow **asymmetry** (a fidelity limitation), **not** a missing or renamed name.

**Apple canonical 52 vs. MetaHuman-baked 51 (exact, case-sensitive):**
The 51 baked names are Apple's 52 **minus `tongueOut`**. Grouped:

- **Eyes (14):** `eyeBlinkLeft` `eyeLookDownLeft` `eyeLookInLeft` `eyeLookOutLeft` `eyeLookUpLeft`
  `eyeSquintLeft` `eyeWideLeft` `eyeBlinkRight` `eyeLookDownRight` `eyeLookInRight`
  `eyeLookOutRight` `eyeLookUpRight` `eyeSquintRight` `eyeWideRight`
- **Jaw (4):** `jawForward` `jawLeft` `jawRight` `jawOpen`
- **Mouth (23):** `mouthClose` `mouthFunnel` `mouthPucker` `mouthRight` `mouthLeft`
  `mouthSmileLeft` `mouthSmileRight` `mouthFrownLeft` `mouthFrownRight` `mouthDimpleLeft`
  `mouthDimpleRight` `mouthStretchLeft` `mouthStretchRight` `mouthRollLower` `mouthRollUpper`
  `mouthShrugLower` `mouthShrugUpper` `mouthPressLeft` `mouthPressRight` `mouthLowerDownLeft`
  `mouthLowerDownRight` `mouthUpperUpLeft` `mouthUpperUpRight`
- **Brows (5):** `browDownLeft` `browDownRight` **`browInnerUp`** (← the single folded shape)
  `browOuterUpLeft` `browOuterUpRight`
- **Cheeks (3):** `cheekPuff` `cheekSquintLeft` `cheekSquintRight`
- **Nose (2):** `noseSneerLeft` `noseSneerRight`
- **Tongue (1): `tongueOut` — Apple's 52nd; NOT baked (default rig). This is the entire count gap.**

**How the Phase-5 driver still resolves everything (no renames):**
- The viewer writes MediaPipe `categoryName` **straight into `morphTargetInfluences` by exact
  name** — `viewer.js` `setInfluence`: `const idx = dict[keyName]; if (idx !== undefined)
  influences[idx] = v;`. **No remap table.** So exact case-sensitive spelling is mandatory.
- **`browInnerUp`:** present in the GLB (1 of the 51), present in Apple's 52, emitted by
  MediaPipe → **resolves 1:1**. The fold already happened upstream in the MetaHuman ARKit
  mapping; the driver never sees L/R inner brow. Nothing to reconcile at drive time.
- **`tongueOut`:** MediaPipe FaceLandmarker v2 does **not emit it** (its output = `_neutral` +
  51 ARKit names, no tongueOut), and the GLB has **no `tongueOut` morph target**. `setInfluence`
  therefore **no-ops** it (`dict['tongueOut']` is `undefined`); the manual slider "just won't
  render" (`viewer.js` comment). **No crash, no rename, no silent aliasing.**
- **Convergence:** the 51 baked shapes == the 51 ARKit names MediaPipe emits, matched by exact
  name. `_neutral` from MediaPipe is explicitly skipped by the viewer.
- **If a tongue-rig variant is ever used** (baking 52), `tongueOut` then resolves 1:1 too — but
  MediaPipe still cannot drive it (no tongueOut channel). Update `out/arkit_51_52_map.json` to
  52 in that case.

**QA assertions (in `out/arkit_51_52_map.json`, for `qa-verifier`):** GLB face-mesh
`morphTargetDictionary` must contain all 51 names (exact case) and must **not** contain
`tongueOut` (default rig); `browInnerUp` present exactly once (not split L/R); groom card meshes
carry the same 51 names; no name renamed/aliased to mask the gap.

---

## 5. The exact EULA question — handed to `license-compliance` + a human lawyer (I do not self-clear)

`out/compliance_report.md` is **`SHIP-CLEARED: no`** (pending human-lawyer review). B2's
conversion **scripts are MIT**, but the **MetaHuman asset that passes through them is governed by
Epic's MetaHuman license / Content EULA / UE EULA** — MIT does **not** launder the Epic-licensed
mesh. I surface the question and stop; a human lawyer decides. Restated verbatim from the
compliance report:

> **Under the current Epic MetaHuman license (`metahuman.com/license`, folded into the standard
> Unreal Engine EULA and the Epic Content EULA as liberalized in June 2025), may we export a
> MetaHuman-derived head as a standalone GLB file and distribute that GLB to end users' web
> browsers inside a commercial web product — where the raw mesh, ARKit morph targets, and
> textures are downloadable and extractable by the end user on a standalone basis — OR does the
> Content EULA prohibition on redistributing Content "on a standalone basis" (and the requirement
> that a Project "add value beyond the value of the Licensed Content") prohibit that specific
> web-delivery mode?**

Lawyer must confirm in writing (see `compliance_report.md` for full text): (1) standalone vs.
incorporated for a browser-downloadable extractable GLB; (2) which instrument controls
(June-2025 MetaHuman permission vs. general Content-EULA standalone clause); (3) "MetaHuman-
derived" status of a Mesh-to-MetaHuman output **and** whether feeding a user photo through
Mesh-to-MetaHuman implicates the **AI-training restriction**; (4) the $1M/yr free threshold and
whether a ~$1,850/yr UE seat is required at scale; (5) any attribution/labeling obligation.

**Do not spend GPU on a B2-specific shippable asset before this resolves.** If counsel says no,
switch the reconstruction base to **FLAME 2023 Open (CC-BY-4.0) — Track B1**, which is
commercially clearable today with attribution and no Epic EULA dependency, and the B2 Epic
dependency disappears entirely.

---

## Deliverables produced this session (no compute, no GLB)
- `out/metahuman_report.md` — this runbook + naming contract + EULA hand-off.
- `out/arkit_51_52_map.json` — machine-readable 51↔52 name contract for `viewer-driver` + `qa-verifier`.
- **No GLB produced** (deferred to the GPU box; `out/head_arkit.glb` will be the copy target after stage 03).
