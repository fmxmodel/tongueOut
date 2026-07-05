# models/ — Track B1 (FLAME 2023 Open, COMMERCIAL) model assets

This directory documents **what the operator must download onto the GPU pod** and
**exactly where it goes**. Nothing here is redistributed by this repo. The large model
files are downloaded ON THE POD, not on the authoring box.

> Governing licensing: `out/compliance_report.md` (TRACK B1 section).
> Setup automation: `scripts/pod_setup_b1.sh`.

Target layout on the pod:

```
/workspace/models/
  flame2023_open/      <- FLAME 2023 Open (CC-BY-4.0) — MANUAL, license-gated download
  mediapipe/           <- face_landmarker.task (Apache-2.0) — auto-downloaded by the setup script
/workspace/inputs/
  random-person.jpeg   <- the single input photo the fit + albedo bake consume
```

---

## 1. FLAME 2023 **Open** — shape/geometry model  (CC-BY-4.0)  → `flame2023_open/`

**This is a SHAPE/GEOMETRY model only — and the Open release is PKL-ONLY.**
**MEASURED ON THE POD (2026-07-05):** the downloaded FLAME 2023 (Open) package contains
**only `flame2023.pkl`** (plus a readme). Confirmed pickle contents: `f (9976,3)`,
`v_template (5023,3)`, `shapedirs (5023,3,400)` (300 id + 100 expr), `posedirs (5023,3,36)`,
`weights (5023,5)`, `J_regressor (5,5023)`, `kintree_table`, `J`, `bs_style`, `bs_type`,
`supr_expression_metadata`. It ships **NO UV coordinates and NO landmark embedding**
(the Open readme points to FLAME-Universe for extras — which are NC-licensed and barred
here). It also does **NOT** contain a texture/albedo model — intentional and correct for
B1 (we bake a per-subject albedo from the input photo instead; see §4).

### Not redistributable by us — operator downloads it
1. Register and log in at **https://flame.is.tue.mpg.de/**.
2. Accept the **CC-BY-4.0** license for the **"FLAME 2023 (Open)"** release.
3. Download the **"FLAME 2023 (Open)"** package **only**.
4. Unpack into **`/workspace/models/flame2023_open/`** (staged file:
   `/workspace/models/flame2023_open/flame2023.pkl`).

### What the B1 fitter uses (single required file + two clean-room substitutes)
| Role | Source | Used for |
|---|---|---|
| Shape/expr/pose LBS model | **`flame2023.pkl`** — the ONLY file from the release | the FLAME mesh generator that the optimizer fits; also the sole source of the topology contract `out/recon/faces.npy` |
| Static landmark embedding | **NOT in the Open release.** **SELF-AUTHORED clean-room** by `recon/flame_landmarks.py`: iBUG static-51 anchors derived *geometrically* from the pkl's own `v_template`/`weights`/`J_regressor` (eyeball joints, jaw-weight lip seam, midline nose profile, anthropometric brow stations). Persisted to `out/recon/lmk_embedding_static51.npz`; debug dump in `out/recon/flame_landmarks_selfauthored.{json,png}`. A staged `landmark_embedding*.npy/.pkl` acts as an *optional override* | 51-pt FLAME anchors for the landmark fit **and** the rig's landmark solves (same file, by construction) |
| UV layout | **NOT in the Open release.** **GENERATED clean-room** by `recon/uv_unwrap.py`: deterministic **xatlas (MIT)** unwrap of the pkl's `v_template`+`f`, persisted to `out/recon/uv_coords.npz` (with `uv_source` provenance field). A staged UV-bearing `head_template.obj` acts as an *optional override* (topology-checked, hard STOP on mismatch) | the UV layout the photo albedo is baked onto (§4) and the GLB's UV assignment |
| Dynamic contour embedding | **Not available and not fabricated** — the fit's jaw-contour term simply stays disabled (static-51 only) | — |

> **Clean-room provenance note (for `license-compliance`):** the landmark anchors and the
> UV layout are **not copied from any FLAME package or repo**. No NC source
> (DECA / EMOCA / TF_FLAME / flame-fitting / smplx / MPI texture package) was read or
> consulted for them. The anchors are vertex indices into the fixed CC-BY topology, chosen
> by deterministic geometric rules documented in `recon/flame_landmarks.py`; the UV is a
> fresh xatlas parameterization of CC-BY geometry. The only FLAME data in the pipeline is
> the CC-BY-4.0 `flame2023.pkl` itself.

> **CC-BY-4.0 obligations (enforced at ship by `license-compliance`):** in-product credit
> screen naming "FLAME 2023 (Open), Max Planck Institute for Intelligent Systems", a link to
> the CC-BY-4.0 license, a "changes were made" statement (the mesh is fit + rigged +
> retextured), and the FLAME paper citation. See `out/compliance_report.md` §B1-1.

### ⛔ DO **NOT** download the FLAME texture / albedo package — or NC "extras"
The **MPI FLAME texture space / albedo** package (`FLAME_texture.npz`, `TextureSpace.mat`,
albedo bases) is **CC-BY-NC-SA-4.0 = NON-COMMERCIAL** and is **BARRED** from this commercial
run (`out/compliance_report.md` §B1-2). Likewise, do **NOT** pull `head_template.obj` or a
landmark embedding from DECA / EMOCA / TF_FLAME / flame-fitting or any other NC FLAME repo —
that would re-taint the CC-BY base (§B1-1 caveat). Since the Open release ships neither,
both are **self-authored/generated clean-room** as described in the table above.

---

## 2. MediaPipe FaceLandmarker model  (Apache-2.0)  → `mediapipe/face_landmarker.task`

- Auto-downloaded by `scripts/pod_setup_b1.sh` from:
  `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`
- Produces the **478 landmarks** (and optional 52 ARKit blendshapes) that the optimization
  fit targets. Apache-2.0 — reproduce the license text + NOTICE at ship.

---

## 3. MediaPipe ↔ FLAME landmark correspondence  → **fitter asset, NOT a FLAME download**

The mapping from MediaPipe's 478 landmark indices to FLAME vertices/barycentric points is
**not part of the FLAME download**. It is authored/selected by `face-reconstructor`
(self-authored index list, or a permissively-licensed correspondence file). Its provenance
must be cleared for commercial use — a CC-BY FLAME model does **not** launder a non-commercial
correspondence file. Flagged to `face-reconstructor` and `license-compliance`.

---

## 4. Texture — NO model file  (per-subject bake from the input photo)

There is deliberately **no albedo/texture model** in this directory. The B1 texture is baked
per-subject: project `/workspace/inputs/random-person.jpeg` onto the FLAME 2023 Open UV using
the fitted camera, complete occluded regions by **mirror symmetry + classical inpainting**
(`cv2.inpaint`, Apache-2.0). **No statistical albedo prior** — explicitly no
`FLAME_albedo_from_BFM.npz`, no AlbedoMM, no Basel BFM, no MPI FLAME texture space
(all non-commercial / BARRED, `out/compliance_report.md` §B1-2/§B1-3).

---

## 5. Input image  → `/workspace/inputs/random-person.jpeg`

Copy this repo's `random-person.jpeg` to the pod at `/workspace/inputs/random-person.jpeg`.
It is the sole subject for both the landmark fit and the albedo bake. (Rights to the photo
itself are a separate legal question owned by `license-compliance`, not an env question.)
