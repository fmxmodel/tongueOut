# GPU-Migration Requirements Manifest — Track B2 (MetaHuman → GLB)

> **⚠ SUPERSEDED FOR THE ACTIVE RUN:** the run pivoted to **Track B1 (FLAME 2023 Open)**.
> The governing GPU manifest is now **`out/gpu_requirements_b1.md`** (Linux RTX 6000 Ada pod).
> This B2 document is retained as historical record.

> Handoff document. Everything below is **DEFERRED to the GPU/migration box** by explicit
> user constraint. This provisioning session installed **no GPU stack** and ran **no model
> or numeric compute** — see `out/env_report.md` for what is READY locally right now.
>
> Generated: 2026-07-04 · owner: `env-provisioner` (Opus 4.8) · pipeline: `smorchj/metahuman-to-glb` `5.7/native-glb`

---

## 0. Why this exists

The B2 back-half converter (`vendor/metahuman-to-glb`) drives **Unreal Engine 5.7** and
**Blender 5.x**. Both need a real GPU (Unreal's renderer + MetaHuman RigLogic; Blender's
GLB/shape-key stages). This provisioning box is a laptop with **Intel UHD (Raptor Lake-P)
integrated graphics only — no NVIDIA GPU, no CUDA, no driver**. Per the run constraints we
prepared every file that can exist without a GPU and recorded the rest here, unfaked.

**Important scope note:** B2 needs **no PyTorch / CUDA model inference**. Unlike Track A
(FaceVerse) or B1 (FLAME + DECA/EMOCA), the B2 "reconstruction" is Epic's **Mesh-to-MetaHuman /
MetaHuman Creator**, not a torch model. The GPU here powers Unreal's renderer and RigLogic,
**not** a Python DL stack. Do not install torch/CUDA-python for B2.

---

## 1. Host / OS for the GPU box

The converter's stage launchers are **PowerShell (`.ps1`)** and its config points at
`UnrealEditor-Cmd.exe` and `blender.exe`. The pipeline is **Windows-native as shipped**.

- **Recommended migration host:** Windows 10/11 x64 with an NVIDIA GPU.
- If a Linux GPU box is mandatory, the `.ps1` launchers must be ported to `bash`/`python`
  and the two Unreal stages (00, 01) re-pathed to `UnrealEditor-Cmd` (Linux) — non-trivial;
  UE 5.7 + MetaHuman tooling is best-supported on Windows. Budget for this if going Linux.

## 2. GPU / driver / CUDA

| Item | Requirement | Notes |
|---|---|---|
| GPU | DirectX-12-capable **NVIDIA** GPU, **RTX-class, ≥8 GB VRAM** recommended | For UE 5.7 editor + MetaHuman rendering/RigLogic. Verify against Epic's UE 5.7 system requirements at migration time. |
| GPU driver | Latest NVIDIA Studio/Game Ready driver supported by UE 5.7 | Installed on the GPU box, not here. |
| CUDA toolkit | **Not required for B2.** | No torch/CUDA inference in this track. Do not install for B2. |
| VRAM | ≥8 GB (more for high-res MetaHuman + textures) | MetaHuman assets + Sequencer bake are memory-heavy. |

## 3. Unreal Engine 5.7  — DEFERRED (GPU stage)

- **Version:** UE **5.7** (exact; the `5.7/native-glb` pipeline is version-pinned; a `5.6.1`
  legacy path also exists but is not the default).
- **Install location on GPU box:** e.g. `C:/Program Files/Epic Games/UE_5.7/` via Epic Games
  Launcher (do **not** attempt a source build on this laptop).
- **Binary the pipeline calls:** `.../Engine/Binaries/Win64/UnrealEditor-Cmd.exe`.
- **Plugins/assets:** MetaHuman plugin + **RigLogic** (ships with UE 5.7). Stages 00/01 run
  inside UnrealEditor-Cmd's **embedded Python** — no separate pip environment needed.
- **Project prerequisite:** a `.uproject` containing an **already-built MetaHumanCharacter**
  under `/Game/<Name>/`. Producing that MetaHuman from the source image (Epic
  **Mesh-to-MetaHuman / MetaHuman Creator**) is the **front half** and is itself GPU/UE work —
  owned by `metahuman-route`, not this converter.
- **Status here:** UE 5.7 **not present** and not installable without a GPU. DEFERRED.

## 4. Blender 5.x  — DEFERRED (GPU stage)

- **Version:** Blender **5.x** (README/config say "5.x"; used by stages **02** and **03**).
- **Built-ins required (no add-ons to install):** glTF 2.0 exporter (`io_scene_gltf2`,
  Draco compression) and the **shape-key** system — both ship with Blender.
- **Python deps:** stages 02/03 run inside **Blender's bundled Python** and use only
  `bpy`, `mathutils.kdtree`, and `numpy` (all bundled). **No pip install needed.**
- **Binary the pipeline calls:** `blender.exe` (path set in `_config/pipeline.yaml`).
- **Status here:** no Blender on PATH (checked PATH, `/opt`, `/usr/local`, flatpak, snap,
  apt). Do **not** install a GPU-dependent build on this box. DEFERRED.

## 5. Python (non-GPU) — informational

- **No `requirements.txt` / conda env in the repo.** B2 needs no standalone DL Python env.
- Stage 04 `build_site.py` uses **standard-library only** (argparse, json, shutil, pathlib) —
  runs under any Python 3.x on the GPU box. (Local Python 3.12.3 is available and would
  suffice, but stage 04 is downstream of the GPU stages, so it stays deferred in sequence.)

## 6. Config to fill in on the GPU box

`vendor/metahuman-to-glb/_config/pipeline.yaml` (copy from `pipeline.example.yaml`, gitignored):

```
ue_version: "5.7"
ue_by_version:
  "5.7":
    project_path: "<...>/MH.uproject"
    editor_cmd:   "C:/Program Files/Epic Games/UE_5.7/Engine/Binaries/Win64/UnrealEditor-Cmd.exe"
blender_exe:      "C:/Program Files/Blender Foundation/Blender 5.0/blender.exe"
```

## 7. Exact commands to run on the GPU box (in order)

```powershell
# one-time: config
cp _config/pipeline.example.yaml _config/pipeline.yaml   # then edit the <...> paths

# front half (metahuman-route): build a MetaHuman from the image via Epic
#   Mesh-to-MetaHuman / MetaHuman Creator, saved under /Game/<Name>/ in the .uproject.

# back half: 5 stages (from repo README) — <Char> e.g. ada
./5.7/native-glb/stages/00-unreal-assemble/tools/run_assemble.ps1 -Char <Char>   # UE assemble (~1-2 min)
./5.7/native-glb/stages/01-unreal-glb-export/tools/run_export.ps1   -Char <Char>   # UE Sequencer ARKit bake + GLB (~1-2 min)
./5.7/native-glb/stages/02-blender-assemble/tools/run_assemble.ps1  -Char <Char>   # Blender: ARKit shape keys + groom (~60-90s)
./5.7/native-glb/stages/03-export-to-glb/tools/run_export.ps1       -Char <Char>   # Blender: Draco GLB (~30-60s)
./5.7/native-glb/stages/04-webview-build/tools/run_site.ps1         -Char <Char>   # three.js viewer build (~5s)
# output: docs/characters/<Char>/<Char>.glb  (~40 MB, 51 ARKit blendshapes)
```

## 8. Naming / QA carry-over for downstream gates

- Converter bakes **51** ARKit blendshapes (`browInnerUp` is a single bilateral shape;
  `tongueOut` needs a non-default rig variant). Phase 5 must reconcile **51 ↔ 52** (fold/flag),
  per plan.md line 70 and `qa-verifier`.

## 9. License / EULA — DEFERRED to `license-compliance` (human-lawyer gate)

- Converter scripts are **MIT** (`vendor/metahuman-to-glb/LICENSE`), but **MetaHuman assets
  are governed by Epic's MetaHuman/Unreal EULA**. Shipping a **standalone GLB outside Unreal**
  is an unresolved **lawyer question** — MIT on the scripts does not clear it. This is Gate 1
  (`SHIP-CLEARED`), not an env question, but flagged here so the GPU box doesn't ship blind.
```
