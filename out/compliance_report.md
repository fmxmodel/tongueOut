# Compliance Report — COMMERCIAL run · Track B1 (FLAME 2023 Open) [ACTIVE] + Track B2 (MetaHuman) [prior record]

> **PIVOT (2026-07-04):** The run has moved from **B2 (MetaHuman)** to **Track B1
> (FLAME 2023 Open reconstruction, Arc2Avatar/DECA-style)** for the commercial single-image →
> ARKit avatar build (target: Linux 6000-Ada GPU pod). The **governing verdict for this run is
> now the Track B1 section below.** The B2 analysis is retained as historical record; B2 remains
> `SHIP-CLEARED: no` pending its Epic-EULA lawyer question. **This session is pure license
> verification — no GPU, no compute, nothing built or shipped.**

- **Gate:** `license-compliance` (Opus 4.8), blocking authority (GOVERNANCE.md §4 Gate 1)
- **Run intent:** Commercial — shippable, web-viewable rigged GLB avatar
- **Active track:** **B1 (FLAME 2023 Open shape → per-subject textured FLAME mesh → 52 ARKit → GLB → three.js)**
- **Run mode this session:** LICENSE VERIFICATION ONLY. No GPU install, no CPU inference/compute,
  nothing published or distributed.
- **Verified:** 2026-07-04 (live re-check this run — licenses change; see per-component sources)

> Governing rule (CLAUDE.md invariant 5 / GOVERNANCE.md §1.5): a commercial artifact ships
> only when **every** component is independently license-cleared. FaceVerse and standard FLAME
> are non-commercial and must never reach a shippable artifact. EULA-scope judgments are a
> **human-lawyer** decision; this gate does not self-clear them.

---

## Per-component verdicts

### 1. `smorchj/metahuman-to-glb` — conversion scripts
- **License:** MIT (confirmed: repo README states "MIT. See LICENSE"). Verified 2026-07-04.
- **Commercial verdict:** ✅ YES **for the scripts/code only.**
- **Obligation:** Preserve the MIT copyright + permission notice in any distribution that
  includes the scripts.
- **CRITICAL CAVEAT (laundering trap):** MIT covers ONLY the conversion pipeline code. It does
  **NOT** cover the MetaHuman asset that passes through it. The repo README does not warn about
  Epic's EULA. An MIT tool does not clear the Epic-licensed mesh it converts. The output GLB's
  license is governed entirely by component #2, not by this MIT grant.

### 2. MetaHuman assets (the exported head) — Epic MetaHuman license + Epic Content EULA + UE EULA  ⛔ CENTRAL B2 RISK
- **License instruments (all apply):**
  - Epic MetaHuman license (`metahuman.com/license`) — folded into the **standard Unreal
    Engine EULA** as of the June 2025 out-of-early-access launch.
  - **Epic Content EULA** (`unrealengine.com/eula/content`) — governs the asset/Content.
  - Verified 2026-07-04.
- **June 2025 liberalization (material, in our favor):** MetaHuman left early access; the
  toolset is now under the standard UE EULA. MetaHuman characters may be used in **any engine
  or DCC** (Unity, Godot, etc.), may be **sold on online marketplaces**, and are classed as
  **"non-engine products"** (no 5% UE royalty when used outside Unreal). **Free** for entities
  under **$1,000,000/yr** revenue; above that, UE **seat licenses (~$1,850/yr)** are required.
  **AI restriction:** MetaHumans may be used in AI-incorporating workflows but **may not be
  used to train or enhance the AI models themselves.**
- **The unresolved conflict:** The general **Content EULA still prohibits redistributing
  Content "on a standalone basis"** and requires that a Project **"add value beyond the value
  of the Licensed Content"** — i.e. you may not "resell, redistribute, or share the Content as
  a standalone product, in whole or in part." A GLB shipped to a **web browser is delivered as
  a raw, downloadable, trivially-extractable mesh** (geometry + 51 baked ARKit morph targets + textures).
  Whether that web-delivery mode counts as permitted "incorporation into a value-adding
  product" or prohibited "standalone redistribution / obtaining the Content separately from a
  Product" is **not resolved** by the public terms. The June 2025 marketplace-sale permission
  (selling the asset to another *creator*) is not obviously the same as **end-user delivery of
  the raw asset inside a running web app.**
- **Commercial verdict:** ⚠️ **UNRESOLVED — HUMAN-LAWYER DECISION.** This gate does NOT
  self-clear it. See "Exact EULA question" below.

### 3. Unreal Engine 5.7 EULA — used to *produce* the asset
- **License:** Standard Unreal Engine EULA. **UE 5.7 released November 2025.** Verified 2026-07-04.
- **Commercial verdict:** ✅ YES to *build/produce* the asset with UE (free under $1M/yr revenue;
  ~$1,850/yr seat above $1M). Producing the mesh is **not** the risk.
- **Obligation / boundary:** The EULA constrains **distribution** of the resulting Content, not
  its creation. The risk lives entirely at the **ship** step (component #2), not the build step.

### 4. MediaPipe FaceLandmarker (`@mediapipe/tasks-vision`) — driving side
- **License:** Apache-2.0. Verified 2026-07-04. Blendshape `categoryName`s are ARKit-named (52).
- **Commercial verdict:** ✅ YES.
- **Obligation:** Include a copy of the Apache-2.0 license text AND reproduce the contents of
  MediaPipe's **NOTICE** file in the shipped product's attributions/credits (Apache-2.0 §4).

### 5. three.js — viewer
- **License:** MIT. **Exact string in the installed artifact** (`node_modules/three@0.170.0/LICENSE`):
  "The MIT License / Copyright © 2010-2024 three.js authors". Verified 2026-07-04 against the
  installed package (earlier draft said "2010–2026"; the pinned 0.170.0 LICENSE reads 2010-2024 —
  reproduce whatever the shipped version's LICENSE actually says).
- **Commercial verdict:** ✅ YES.
- **Obligation:** Include the MIT copyright + permission notice verbatim in the distribution
  (credits/licenses file bundled with the web app).

### 6. FLAME 2023 Open — the B1 fallback base (NOT used in B2, recorded as the safe alternative)
- **License:** **CC-BY-4.0**, FLAME 2023 **Open** model, released **Nov 25 2025** (model
  license page confirms CC-BY-4.0; standard FLAME and all other FLAME variants remain
  MPI non-commercial). Verified 2026-07-04.
- **Commercial verdict:** ✅ YES with attribution — the clean, self-contained, EULA-free
  commercial reconstruction base.
- **Obligation if adopted:** CC-BY-4.0 requires appropriate credit, a link to the license, and
  indication that changes were made; plus cite the FLAME paper (Li, Bolkart, Black, Li, Romero,
  "Learning a model of facial shape and expression from 4D scans," SIGGRAPH Asia 2017).
- **Note:** If the B2 EULA question comes back "no," the pipeline switches base to FLAME 2023
  Open (plan §B1) via `face-reconstructor` — and B2's Epic dependency disappears entirely.
  **DECA/EMOCA fitter licenses must be independently cleared under B1** (a CC-BY model does not
  launder a non-commercial fitter) — flag to `face-reconstructor` if B1 is chosen.

---

# ===== TRACK B1 (FLAME 2023 Open) — GOVERNING SECTION FOR THIS RUN =====

Commercial single-image → FLAME reconstruction. The user raised a specific, correct distinction
that this section verifies and turns into a viability verdict: **the FLAME shape/geometry model
is commercial-OK, but the MPI FLAME *texture space* is a stricter NON-commercial license**, so a
commercial build may not use MPI's default FLAME textures and must source textures from a
commercially-clean origin.

## B1-1. FLAME 2023 Open — SHAPE / GEOMETRY model
- **License:** **CC-BY-4.0** (FLAME 2023 **Open** model). Source: `flame.is.tue.mpg.de/modellicense.html`
  (live-verified 2026-07-04): "FLAME 2023 Open" is CC-BY-4.0; 2017/2019/2020/standard FLAME remain
  the MPI **Academic** (non-commercial) license. Released Nov 2025.
- **Commercial verdict:** ✅ **YES with attribution.** This is the clean commercial reconstruction base.
- **Obligation (CC-BY-4.0):** in-product credit screen with (a) appropriate credit, (b) a link to
  the CC-BY-4.0 license, (c) a statement that changes were made (the mesh is fit + rigged +
  re-textured), and (d) citation of the FLAME paper.
  - **Exact attribution text to place in the credits screen:**
    > "This product uses the FLAME 2023 (Open) head model by the Max Planck Institute for
    > Intelligent Systems, licensed under CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/).
    > The model was fit to an input image, re-textured, and rigged; these are modifications of the
    > original. FLAME: Tianye Li, Timo Bolkart, Michael J. Black, Hao Li, Javier Romero,
    > 'Learning a model of facial shape and expression from 4D scans,' ACM Transactions on Graphics
    > (SIGGRAPH Asia) 2017."
- **Asset-provenance caveat (must confirm before ship):** the FLAME **UV layout** and the
  **landmark-embedding** files (`landmark_embedding.npy` / `flame_static_embedding.pkl` / the head
  template UVs) that a fitter needs must be taken from the **CC-BY Open release**, NOT from the
  CC-BY-NC-SA texture package (B1-2). Confirm each asset file ships under the Open license before use.

## B1-2. MPI FLAME TEXTURE / APPEARANCE space (albedo)  ⛔ BARRED FROM COMMERCIAL
- **License:** **CC-BY-NC-SA-4.0** (NON-commercial, ShareAlike). Source:
  `flame.is.tue.mpg.de/texturelicense.html` (live-verified 2026-07-04): "The Texture … is under a
  Creative Commons BY-NC-SA 4.0 license … redistribute and adapt it for **non-commercial purposes**
  … distribute derivative works under the same license." Built on the **FFHQ** dataset; derived via
  `HavenFeng/photometric_optimization`.
- **Commercial verdict:** ❌ **NO — BARRED from the commercial B1 build.** The user's constraint is
  **CONFIRMED CORRECT.** The MPI FLAME default texture/albedo must **not** appear in any shippable
  artifact. (Also note the ShareAlike term would try to force the whole derivative under NC — a
  second reason to keep it out entirely.)
- **Consequence:** the commercial build must obtain its texture from a commercially-clean source
  (B1-3). This is the crux the user asked us to resolve.

## B1-3. Commercially-clean TEXTURE strategy — "is this viable?"  → **YES, VIABLE (one clean path)**

### (a) Image-baked per-subject albedo  → ✅ RECOMMENDED, commercially clean
- **Method:** project/sample the **input photo** (`random-person.jpeg`) onto the FLAME 2023 Open UV
  to produce a **per-subject** texture (the standard DECA/EMOCA/Arc2Avatar image-to-UV bake), filling
  occluded/unseen regions by **mirror symmetry + classical inpainting**.
- **Why clean:** the resulting texture derives only from (i) the FLAME 2023 Open UV layout (CC-BY,
  commercial-OK per B1-1) and (ii) the user's own input image. No MPI texture basis, no third-party
  albedo model. ✅ commercially clean **as to third-party IP**.
- **HARD CAVEAT — do NOT use a statistical albedo prior:** DECA/EMOCA "albedo" is normally produced by
  **photometric optimization regularized against `FLAME_albedo_from_BFM.npz` (Basel-derived) or the MPI
  FLAME texture basis.** Using that basis **reintroduces non-commercial IP** and re-poisons the output.
  The clean bake must be **pure image projection + geometric inpainting**, with **no** BFM/AlbedoMM/MPI
  statistical albedo model in the loop.
- **SEPARATE, UN-ADJUDICATED question (flag, do NOT decide):** whether the user holds the rights to the
  specific input photo (`random-person.jpeg`) — ownership/licensing of the image, plus subject consent
  and biometric/publicity/GDPR-type rights in a real person's likeness. This gate does **not** clear
  that; it is a **human-legal** question about the input, distinct from the FLAME/texture IP chain.

### (b) Alternative "cleared texture packages" — evaluated; essentially all are BARRED
- **MPI FLAME texture space** → CC-BY-NC-SA-4.0 → ❌ BARRED (B1-2).
- **DECA `FLAME_albedo_from_BFM.npz`** → derived from the **Basel Face Model (BFM)** → University of
  Basel academic/non-commercial license → ❌ BARRED. (Source: DECA setup instructions +
  `TimoBolkart/BFM_to_FLAME`; live-verified 2026-07-04.)
- **AlbedoMM (`waps101/AlbedoMM`, "A Morphable Face Albedo Model")** → "academic research purposes only;
  contact William Smith for commercial use" → ❌ BARRED. (Basel/3DMD-derived.)
- **Basel BFM itself** → non-commercial academic → ❌ BARRED.
- **FaceVerse textures** → Tsinghua non-commercial → ❌ BARRED.
- **Finding:** there is **NO drop-in, commercially-clean statistical face-albedo model** in the
  FLAME ecosystem — the common ones are all Basel-BFM- or FFHQ-derived and non-commercial. Note too
  that **FLAME 2023 Open is a shape/geometry model only** — the CC-BY Open release does **not** bundle
  a commercial texture. So "use the other package's texture" does **not** resolve to any off-the-shelf
  package; the only commercially-clean texture is the **per-subject image bake (a)**.

### (c) Recommended path + exact requirements
- **Recommendation:** per-subject **image-baked albedo** from the input photo onto the FLAME 2023 Open
  UV, occlusion-filled by symmetry + classical inpainting; **no statistical albedo prior**.
- **Requires:** (1) a FLAME fit (pose/shape/camera) to drive the projection — from a **commercially-clean
  fitter** (see B1-4); (2) the FLAME UV + landmark assets taken from the **CC-BY Open** release, not the
  NC texture package (B1-1 caveat); (3) confirmation of input-photo rights (a, separate legal question).

## B1-4. Fitter CODE + PRETRAINED WEIGHTS — a CC-BY shape model does NOT launder these  ⛔ OPEN
The reconstruction fitter is a **separate license** from the FLAME model. Weights especially.
- **DECA (`yfeng95/DECA`)** — code + `deca_model.tar` weights → **NON-COMMERCIAL** MPI-style LICENSE
  ("sole purpose of … non-commercial scientific research/education/artistic … any use for commercial …
  is prohibited, including incorporation in a commercial product"). Source: `github.com/yfeng95/DECA/blob/master/LICENSE`,
  live-verified 2026-07-04. → ❌ **BARRED for commercial B1.**
- **EMOCA (`radekd91/emoca`, `emoca.is.tue.mpg.de/license.html`)** — code + pretrained models → **NON-COMMERCIAL**
  MPI license; "For commercial uses … email ps-license@tue.mpg.de." Live-verified 2026-07-04. → ❌ **BARRED.**
- **Arc2Avatar (`dimgerogiannis/Arc2Avatar`)** — **own code is MIT** (permits commercial), BUT it is
  **not commercially clean as-run** because of its dependency stack (MIT on the wrapper does **not**
  launder them):
  - **ID guidance requires InsightFace / ArcFace pretrained recognition models (antelopev2)** →
    InsightFace **code is MIT but its PRETRAINED MODELS are "non-commercial research only; contact for
    commercial licensing."** → ❌ blocks commercial as-run. (Source: `insightface.ai` licensing; live-verified 2026-07-04.)
  - **Arc2Face** guidance model (foivospar) — model card claims MIT weights, but trained on a
    **CC-BY-NC-SA-4.0 dataset**; provenance needs its own review before commercial reliance.
  - **Stable Diffusion base** (SDS) → **CreativeML OpenRAIL-M** — commercial permitted but with
    use-restrictions; a separate flag, not a clean MIT.
  - **Output format mismatch:** Arc2Avatar emits **3D Gaussian Splatting**, not a FLAME-topology
    UV-textured mesh — a technical break with the pipeline's single-FLAME-topology + GLB-morph-target
    invariant (separate from licensing, but relevant to viability).
  - → ⚠️ **BARRED as-run**; would require a **commercial InsightFace model license** + SD-base review +
    Arc2Face provenance review to become viable.
- **What IS commercially usable for the fit:** an **optimization-based landmark fit to FLAME 2023 Open
  with NO non-commercial pretrained weights** — e.g. differentiable/energy-based fitting of FLAME to
  **MediaPipe FaceLandmarker** landmarks (Apache-2.0) using the CC-BY FLAME model + CC-BY landmark
  embedding, with **permissively-licensed or self-authored** fitting code. This avoids DECA/EMOCA/ArcFace
  weights entirely. It is the recommended commercially-clean B1 reconstruction route.
- **Bottom line:** **none of the three named turnkey fitters (DECA, EMOCA, Arc2Avatar-as-run) is
  commercially clean out of the box.** The commercial B1 fit must be the optimization-based FLAME-2023
  route above, OR each non-commercial weight (DECA/EMOCA/InsightFace) must be separately commercially
  licensed. **This is the single largest open B1 clearance — flag to `face-reconstructor`.**

## B1-5. Driving + viewer (unchanged, already commercial-clean)
- **MediaPipe FaceLandmarker** → Apache-2.0 ✅ (reproduce license text + NOTICE).
- **three.js** → MIT ✅ (reproduce copyright + permission notice).

## B1 per-component summary
| Component | License (verified 2026-07-04) | Commercial? | Obligation / action |
|---|---|---|---|
| FLAME 2023 Open (shape/geometry) | CC-BY-4.0 | ✅ | Credit screen + license link + "changes made" + FLAME paper cite |
| MPI FLAME texture space (albedo) | CC-BY-NC-SA-4.0 | ❌ **BARRED** | Must not appear in shipped artifact |
| Chosen texture: per-subject image bake | derived from input photo + CC-BY UV | ✅ (IP-clean) | No statistical albedo prior; confirm input-photo rights (separate legal Q) |
| DECA `FLAME_albedo_from_BFM` / AlbedoMM / BFM | Basel non-commercial | ❌ **BARRED** | Do not use any Basel-derived albedo |
| DECA (code+weights) | MPI non-commercial | ❌ **BARRED** | Do not use for commercial |
| EMOCA (code+weights) | MPI non-commercial | ❌ **BARRED** | Do not use for commercial |
| Arc2Avatar (own code) | MIT | ⚠️ | Clean wrapper, but see deps below |
| InsightFace/ArcFace pretrained (Arc2Avatar dep) | models non-commercial (code MIT) | ❌ **BARRED as-run** | Needs commercial InsightFace license |
| Stable Diffusion base (Arc2Avatar dep) | CreativeML OpenRAIL-M | ⚠️ | Commercial w/ use-restrictions — review |
| Optimization fit to FLAME 2023 + MediaPipe landmarks (recommended) | permissive/self-authored + Apache + CC-BY | ✅ | The commercially-clean fit route |
| MediaPipe FaceLandmarker | Apache-2.0 | ✅ | License text + NOTICE |
| three.js | MIT | ✅ | Copyright + permission notice |

## B1 verdict
- **(a) Is a commercial B1 build viable WITHOUT MPI textures?** **YES — viable.** The texture problem
  is cleanly solvable via per-subject image-baked albedo (no MPI/Basel albedo). MPI FLAME texture is
  confirmed CC-BY-NC-SA-4.0 and is correctly BARRED.
- **(b) Recommended commercially-clean texture source:** **per-subject image bake** — project the input
  photo onto the FLAME 2023 Open UV, occlusion-fill by symmetry + classical inpainting, **no statistical
  albedo prior**.
- **(c) Components still to clear before commercial ship (the CC-BY shape model does NOT clear these):**
  1. **The FITTER + its pretrained weights** — DECA (NC) and EMOCA (NC) are BARRED; Arc2Avatar is
     BARRED as-run via InsightFace/ArcFace NC weights (+ SD OpenRAIL, + 3DGS format mismatch). Adopt the
     **optimization-based FLAME-2023 + MediaPipe** fit, OR commercially license each NC weight. **[biggest gap]**
  2. **FLAME UV + landmark-embedding asset provenance** — confirm they come from the **CC-BY Open**
     release, not the CC-BY-NC-SA texture package.
  3. **No Basel/AlbedoMM/BFM albedo** anywhere in the texture pipeline (enforce the "no statistical
     prior" rule at the reconstruction stage).
  4. **Input-photo rights** — ownership/consent/biometric-publicity in `random-person.jpeg` — a separate
     **human-legal** question; flagged, **not** adjudicated here.
  5. **Attribution wiring** — FLAME 2023 Open CC-BY credit + FLAME paper cite; MediaPipe Apache NOTICE;
     three.js MIT notice.

**Because component (c)(1) — the fitter weights — is not yet cleared, Track B1 is `SHIP-CLEARED: no`
today.** It flips to yes once the fit is a commercially-clean one (optimization route or licensed
weights), the texture is confirmed image-baked with no statistical prior, and asset provenance + notices
are wired. The texture question the user raised is **resolved and viable**; the fitter is the remaining blocker.

---

# ===== FINAL ACCEPTANCE — B1 AUTHORED-CODE LICENSE SCAN (2026-07-04) =====

The prior B1 verdicts were written before the scaffold was authored. This pass **read the actual
authored code** and confirms the recommended commercially-clean route was the one BUILT — the
biggest open item (c)(1) was resolved *in the code itself* by choosing the optimization fit and
never importing a non-commercial fitter. Method: direct read + dependency grep, not re-assertion.

## AC-1. Dependency scan of the authored pipeline — CLEAN ✅
Every real `import`/`from` in `recon/`, `rig/`, `blender_build_rig.py`, plus `requirements-b1.txt`,
`scripts/pod_setup_b1.sh`, and `out/viewer/` was enumerated. The only heavyweight runtime deps are
commercially permissive:
- **PyTorch / torchvision** (BSD-style) — pod compute (`pod_setup_b1.sh` installs `torch==2.4.1+cu121`).
- **PyTorch3D 0.7.8** — **BSD-3-Clause** (rasterization/camera for fit + bake).
- **MediaPipe 0.10.14** — **Apache-2.0** (landmarks + the driving side).
- **OpenCV (opencv-python)** — library **Apache-2.0** (`cv2.inpaint` = classical inpainting).
- **trimesh** — **MIT**; **numpy** (<2, mediapipe C-ABI) / **scipy** — **BSD**; **Pillow** — HPND;
  **imageio** — BSD-2; **scikit-image** — BSD-3; **fvcore** — Apache-2.0; **iopath** — MIT; **ninja** — Apache-2.0.
- Viewer: **three@0.170.0** — **MIT**; **@mediapipe/tasks-vision@0.10.14** — **Apache-2.0**; **vite@6** (+ rollup/esbuild/postcss/nanoid/tinyglobby/fdir/picomatch/@types/estree = MIT, picocolors = ISC, source-map-js = BSD-3) — all dev-only, permissive.

## AC-2. Confirmed ABSENCE of every non-commercial trap — CLEAN ✅
Grepped the authored tree for each barred item. **Every hit was a NEGATION in a comment/docstring
declaring the item's deliberate absence** (e.g. `flame_model.py`: "does NOT vendor or copy code from
smplx / FLAME_PyTorch / DECA / EMOCA"; `requirements-b1.txt`: "NO deca / emoca … NO nvdiffrast";
`fit_flame.py`: "NO learned reconstruction weights (no DECA/EMOCA/Arc2Avatar/InsightFace), NO
nvdiffrast, NO 3DGS"; `bake_texture.py`: "no FLAME_albedo_from_BFM, no AlbedoMM, no Basel").
**No import, install, download, or data reference to ANY of:** `nvdiffrast`, DECA, EMOCA,
`insightface`/`arcface`/`antelopev2`, `smplx`, `FLAME_PyTorch`, `AlbedoMM`, `FLAME_albedo_from_BFM`,
Basel `BFM`, Stable Diffusion / diffusers / Arc2Face / Arc2Avatar, the MPI FLAME **texture** package,
or 3D Gaussian Splatting. No model weights exist anywhere in the tree (`.pth`/`.ckpt`/`.safetensors`/
`.onnx`/`.tar`/`.uasset`/`deca_model*` = none). **The single largest open B1 item — the fitter — is
resolved in code: the fit is pure optimization (`recon/fit_flame.py`, Adam over FLAME params against
MediaPipe iBUG-68 landmarks), no learned reconstruction network, no non-commercial weights.**

## AC-3. Clean-room FLAME loader — CONFIRMED ✅
`recon/flame_model.py` is a **from-scratch** implementation of the FLAME forward function (LBS +
shape/expression/pose blendshapes) written from the published FLAME/SMPL equations. It vendors/copies
**no** smplx / FLAME_PyTorch / DECA / EMOCA code (all self-authored: `batch_rodrigues`,
`_rigid_transform_chain`, `FlameModel.decode`, a chumpy-free `_StubUnpickler`). It only **loads** the
operator-supplied FLAME 2023 **Open** `.pkl` at runtime. Standard: the FLAME 2023 Open **shape model
data** is CC-BY-4.0 (commercial-OK); the traps were the *code* and the *texture/albedo* — both avoided.

## AC-4. Texture path — CONFIRMED pure image bake, no statistical prior ✅
`recon/bake_texture.py` projects the input photo onto the FLAME Open UV (self-authored numpy UV
rasterizer + PyTorch3D depth for occlusion), fills unseen texels by **template-space mirror symmetry**
(`cKDTree`, x→−x) then **`cv2.inpaint(TELEA)`**, then mean-color outside UV islands. **No albedo basis,
no BFM/AlbedoMM/MPI texture space, no de-lighting prior** anywhere; the code even records
`"no_statistical_albedo_prior": true` in `bake_summary.json` and honestly flags the map as *shaded*
appearance (de-lighting priors being license-barred). Matches the B1-3(a) mandated method exactly.

## AC-5. Rig / ARKit spec — CONFIRMED self-authored, license-clean ✅
`rig/arkit_spec.py` + `rig/build_arkit_shapes.py`: ARKit-52 targets authored from Apple's **public
textual** blendshape descriptions (facial-anatomy facts, not copyrightable) + iBUG-68 semantics; the
docstring states **no DECA/EMOCA/ARKit-mesh assets, no non-commercial FLAME→ARKit coefficient tables,
no MetaHuman reference deltas** were consulted or copied. Pose shapes drive FLAME joints via the
reconstructor's own LBS; expression shapes are a ridge least-squares over the CC-BY FLAME expression
basis. Only model data touched = FLAME 2023 Open (CC-BY) + `out/recon/` artifacts.

## AC-6. GLB builder + viewer — CONFIRMED clean ✅ (with a live NOTICE gap)
- `blender_build_rig.py` assembles the GLB via Blender's `bpy` (headless glTF export). **Note:**
  Blender is **GPL** but is a **build tool** — it processes data; its GPL does not reach the GLB output
  (settled, like a compiler). `pod_setup_b1.sh` installs a portable Blender 4.2.3 on the pod. **Do not
  redistribute the Blender binary** as part of the shipped web product (it isn't — only the GLB + viewer ship).
- Viewer: no external code CDN; the only external URL is Google's MediaPipe **model** endpoint
  (`storage.googleapis.com/.../face_landmarker.task`, Apache-2.0) as an offline-first fallback.
- **The built `out/viewer/dist/` already redistributes** three.js + the MediaPipe WASM binaries.
  three.js MIT notice **survives** in the minified bundle (banner "Copyright 2010-2024 Three.js Authors
  / @license" present in `dist/assets/index-*.js`) — good. **But `dist/` bundles NO Apache-2.0 LICENSE
  and NO MediaPipe NOTICE, and no FLAME CC-BY credit** — the attribution obligation is **NOT yet wired**
  (grep for credit/attribution/CC-BY/NOTICE in `out/viewer/src`+`index.html` = zero hits). Not a breach
  (nothing published this session), but a hard pre-ship item (see checklist AC-9(b)).

## AC-7. No shippable artifact produced — SHIP gate honored ✅
`out/head_arkit.glb` does **not** exist. `out/recon/` and `out/shapes/` hold only reports/JSON
(`recon_report.md`, `arkit_manifest.json`, `rig_report.md`) — no meshes, no textures, no weights.
`models/` holds only `README.md` (the FLAME 2023 Open download is a **gated CC-BY pod download**, not
vendored). No FLAME model/UV/embedding data is present locally. Nothing was built or shipped this session.

## AC-8. Verdict of the authored-code scan
**The authored code stays license-clean: YES.** The scaffold implements exactly the commercially-clean
B1 route this gate recommended — optimization fit (no NC fitter/weights), image-baked texture (no
statistical albedo prior), clean-room FLAME loader, self-authored ARKit spec, permissive deps only.
The former "biggest gap" (the fitter) is **resolved in code**. What remains before ship is **not code**
— it is operator/asset/legal provenance + attribution wiring, enumerated next.

## AC-9. The exact pre-ship checklist that flips B1 `SHIP-CLEARED: no → yes` (all required)
These are pod/ship-stage acts, not code changes. Nothing in the authored code blocks ship.
- **(a) FLAME 2023 Open provenance (operator, on the pod).** Download the **"FLAME 2023 (Open)"**
  release (CC-BY-4.0) after registering + accepting the license at `flame.is.tue.mpg.de`.
  - **RESOLVE the landmark-embedding provenance flag** (raised by `face-reconstructor`): the UV
    template (`head_template.obj` with `vt`) **and** the landmark-embedding file
    (`landmark_embedding.npy` / `flame_static_embedding.pkl`) must come **from the CC-BY Open release
    itself**, NOT from the CC-BY-NC-SA texture package and NOT from a DECA/EMOCA mirror. Confirm each
    file's origin in writing (record the download URL + license per file in `recon_report.md`). If the
    Open release does not ship a landmark embedding, a commercially-clean embedding must be
    **self-authored** (pick barycentric landmark points on the Open template) — do NOT import the NC one.
    `fit_flame.py` already asserts the UV template carries `vt` and matches topology; it cannot verify
    *license origin* — that is the operator's written confirmation.
- **(b) Attribution wiring into the shipped product (currently absent).** Add a credits/licenses
  surface to the viewer AND a bundled `licenses/` (or `THIRD-PARTY-NOTICES.txt`) in `dist/`:
  - **FLAME 2023 Open — CC-BY-4.0:** the exact credit + FLAME-paper citation from B1-1 (Max Planck
    Institute; link `https://creativecommons.org/licenses/by/4.0/`; state "the model was fit,
    re-textured, and rigged — changes were made"; cite Li, Bolkart, Black, Li, Romero, "Learning a
    model of facial shape and expression from 4D scans," ACM TOG (SIGGRAPH Asia) 2017).
  - **MediaPipe (`@mediapipe/tasks-vision`, Apache-2.0):** bundle the full **Apache-2.0 license text**
    **and** MediaPipe's **NOTICE** — both **from the upstream `google-ai-edge/mediapipe` repo** (the npm
    tarball ships neither). Required because `dist/mediapipe/wasm/*` redistributes the binaries.
  - **three.js (MIT):** the MIT copyright + permission notice — banner already preserved in-bundle
    ("Copyright 2010-2024 Three.js Authors"); also list it in the notices file.
  - **PyTorch3D (BSD-3-Clause):** source-side only — reproduce the BSD-3 copyright + disclaimer **if the
    pod pipeline code/binaries are redistributed** (not triggered by shipping just the GLB + web viewer).
  - **OpenCV (Apache-2.0):** same source-side trigger — include the Apache LICENSE + OpenCV NOTICE only
    if OpenCV binaries are redistributed (not triggered by the web product; relevant only if the pod
    image is shipped).
- **(c) Texture = image bake only, verified on the pod.** `bake_summary.json` must show
  `no_statistical_albedo_prior: true` and the pod install must contain no BFM/AlbedoMM/MPI-texture data
  (enforced by `requirements-b1.txt` absences; re-confirm nothing was side-loaded).
- **(d) Input-photo rights — HUMAN-LEGAL, flag not adjudicated.** Confirm rights to
  `random-person.jpeg`: (i) ownership/licensing of the image file, (ii) the depicted person's
  **consent** to create and distribute a biometric 3D likeness, and (iii) publicity/biometric/GDPR/BIPA
  exposure in shipping a real person's face. This gate does **not** clear it; qualified counsel must.
  Independent of the FLAME/texture IP chain — a clean model does not clean a wrongfully-used photo.

---

## The exact human-lawyer EULA question (do NOT let any agent self-answer this)

> **Under the current Epic MetaHuman license (`metahuman.com/license`, folded into the standard
> Unreal Engine EULA and the Epic Content EULA as liberalized in June 2025), may we export a
> MetaHuman-derived head as a standalone GLB file and distribute that GLB to end users' web
> browsers inside a commercial web product — where the raw mesh, 51 baked ARKit morph targets, and
> textures are downloadable and extractable by the end user on a standalone basis — OR does the
> Content EULA prohibition on redistributing Content "on a standalone basis" (and the
> requirement that a Project "add value beyond the value of the Licensed Content") prohibit that
> specific web-delivery mode?**

Sub-questions a lawyer must confirm in writing:
1. **Standalone vs. incorporated:** Is a client-downloadable, browser-delivered GLB treated as
   "incorporation into a value-adding Product" (permitted) or "distribution of the Content
   separately from / on a standalone basis from a Product" (prohibited)? The web transport makes
   the raw asset extractable — does that extractability itself breach the standalone clause?
2. **Which instrument controls:** Does the June 2025 MetaHuman-specific permission override the
   general Content EULA standalone-redistribution clause for MetaHuman-derived meshes, and to
   what scope — marketplace sale of the asset to another creator vs. end-user runtime delivery
   in a web app?
3. **"MetaHuman-derived" status:** The B2 front half is "Image → MetaHuman via Epic
   Mesh-to-MetaHuman." Does the resulting head remain Epic "Content"/"MetaHuman" bound by the
   EULA (assume yes), and does feeding a user photo through Mesh-to-MetaHuman implicate the AI
   restriction ("may not use MetaHumans to train or enhance the AI models themselves")?
4. **Revenue scale:** Confirm the $1M/yr free threshold and whether a UE seat license
   (~$1,850/yr) is required for our distribution model at scale.
5. **Attribution/labeling:** Any obligation to label or attribute the mesh as MetaHuman-derived
   in the shipped product?

Preferred resolution path: obtain **Epic's own written confirmation** (or a negotiated license)
for the web-GLB-standalone case, reviewed by qualified counsel. "Probably fine" is not clearance.

---

## FINAL ACCEPTANCE PASS — verification against the completed local scaffold (2026-07-04)

This section confirms the SHIP-CLEARED verdict still holds after the scaffold was fully built.
Method: direct scan of built files, not re-assertion.

### A. No shippable artifact was produced (SHIP gate honored) ✅
- `out/head_arkit.glb` **does NOT exist** (`ls` → No such file or directory). The pipeline
  produced **no** MetaHuman-derived shippable GLB this session.
- `out/recon/` and `out/shapes/` are **empty** (no meshes, no model weights). Consistent with
  the no-compute / scaffold-only mode.
- **Determination:** Because no B2 asset was built, **nothing Epic-EULA-governed left the box.**
  The `SHIP-CLEARED: no` gate was respected — no ship, no artifact, no EULA breach this session.

### B. Installed viewer dependencies are all commercial-safe ✅
Scan of `out/viewer/node_modules` (top-level license field of every installed package):
- **Runtime deps:** `three@0.170.0` → **MIT**; `@mediapipe/tasks-vision@0.10.14` → **Apache-2.0**.
- **Build/dev deps (not shipped in the runtime bundle):** `vite@6.4.3`, `rollup@4.62.2`,
  `esbuild@0.25.12`, `postcss@8.5.16`, `nanoid@3.3.15`, `tinyglobby@0.2.17`, `fdir@6.5.0`,
  `picomatch@4.0.5` → **MIT**; `picocolors@1.1.1` → **ISC**; `source-map-js@1.2.1` → **BSD-3-Clause**.
- All permissive (MIT / ISC / BSD-3-Clause / Apache-2.0). **No copyleft, no non-commercial,
  no source-available-only** dep present. `out/viewer/package.json` pins only the two runtime
  deps above + `vite` (dev). ✅
- **NOTICE gap to close before ship:** the `@mediapipe/tasks-vision` **npm package ships no
  LICENSE and no NOTICE file** (only README). Apache-2.0 §4 still requires that, when the
  MediaPipe binaries are redistributed, the product include a copy of the **Apache-2.0 license
  text** AND reproduce MediaPipe's **NOTICE** — these must be sourced from the upstream
  `google-ai-edge/mediapipe` repo, not from the npm tarball. Tracked in the checklist below.

### C. No non-commercial / poisoning component is present in the tree ✅ (with ONE flag)
- **No FaceVerse** anywhere in built output (only referenced as a rejected option in `.md` docs).
- **No standard FLAME**, **no study-only reference FBX**, **no `.pkl`/`.pth`/`.ckpt` model
  weights**, **no `.uasset`** anywhere under the repo (excluding `.git`/`node_modules`).
- Vendored code is exactly one tree: `vendor/metahuman-to-glb` — **MIT** (`LICENSE`:
  "Copyright (c) 2026 smorchj"), scripts only, consistent with component #1.

### D. ⚠️ FLAG — Epic MetaHuman-derived demo GLBs ARE baked into the vendored repo (latent trap)
The premise "the vendor tree contains NO MetaHuman assets" is **NOT accurate.** The upstream
smorchj repo bundles its GitHub-Pages demo characters — **four standalone MetaHuman-derived GLBs
plus full texture sets**:
- `vendor/metahuman-to-glb/docs/characters/taro/taro.glb` (~80 MB) + `textures/` + `mh_materials.json`
- `vendor/metahuman-to-glb/docs/characters/ada/ada.glb` (~53 MB) + `textures/` + `mh_materials.json`
- `vendor/metahuman-to-glb/docs/5.7/characters/bruce/bruce.glb` (~79 MB)
- `vendor/metahuman-to-glb/docs/5.7/characters/bo/bo.glb` (~42 MB)

"Ada / Taro / Bo / Bruce" are **Epic MetaHuman preset characters**; the README hero image is
literally "Ada, MetaHuman rendered in the three.js gallery." These GLBs are **Epic-EULA-governed
Content** — the same laundering trap as component #1: the repo's **MIT LICENSE covers the software
only, NOT these MetaHuman assets**, and the README makes no Epic-EULA mention. They are precisely
the "standalone GLB redistribution of MetaHuman Content" case that is the unresolved lawyer question.

**Blast radius this session — contained:**
- The viewer does **NOT** reference them: `out/viewer/src/main.js` loads a config-driven
  `CONFIG.glbUrl`; no `taro/ada/bruce/bo` reference anywhere in `out/viewer/{src,scripts,index.html}`.
- **No GLB was copied into** `out/viewer/` (public/dist/src) — the viewer ships no mesh.
- So these demo GLBs did **not** enter the pipeline output and did **not** ship. No breach occurred.

**Obligation before ANY commercial ship (latent, must-do):**
- **Do NOT redistribute, bundle, deploy, or copy these four demo GLBs (or their textures) into
  the shipped web product or any deployment bundle.** If `vendor/` is ever packaged for
  deployment, **exclude `vendor/metahuman-to-glb/docs/`** (add a deploy-exclude / `.gitignore`
  or vendor only the scripts, not the `docs/` demo assets).
- Redistributing them would trigger the **same** Epic MetaHuman/Content-EULA standalone-
  redistribution question as B2 — do not ship them on this gate's authority.

---

## Attribution / notice obligations for the shipped product (record now, wire in before ship)
- **three.js (MIT):** reproduce the MIT copyright + permission notice verbatim. Exact copyright
  line in the pinned artifact: **"Copyright © 2010-2024 three.js authors"** (from
  `node_modules/three@0.170.0/LICENSE`; copy the LICENSE of whatever version ships).
- **MediaPipe (`@mediapipe/tasks-vision`, Apache-2.0):** include the full **Apache-2.0 license
  text** AND reproduce MediaPipe's **NOTICE** file. NOTE: the **npm package ships neither** — pull
  both from the upstream `google-ai-edge/mediapipe` source repo before ship.
- **smorchj/metahuman-to-glb (MIT):** include the MIT notice **"Copyright (c) 2026 smorchj"** +
  permission text **only if the scripts are redistributed**. This MIT grant covers the scripts
  ONLY — it does not clear the four bundled demo MetaHuman GLBs (see Final Acceptance Pass §D),
  which must not be redistributed.
- **If B1 fallback adopted — FLAME 2023 Open (CC-BY-4.0):** in-product credit screen with
  appropriate credit + link to the CC-BY-4.0 license + "changes were made" indication + FLAME
  paper citation.
- **If B2 ships (only after lawyer clearance):** whatever MetaHuman attribution/labeling counsel
  requires per sub-question 5.

---

## This-session (local scaffold-only) determination
- Nothing is published, distributed, or shipped this session; no GPU/compute is run.
- The licensing gate blocks **SHIP**, not local file-building. **Local scaffolding may proceed.**
- **Stamp:** No artifact produced by this pipeline may be shipped, published, or delivered to any
  external/end user until EITHER (a) written human-lawyer clearance of the B2 EULA question above
  is obtained and recorded here, OR (b) the reconstruction base is switched to **FLAME 2023 Open
  (B1)**.
- **Compute-spend guidance:** Resolve the lawyer question BEFORE spending GPU on a B2-specific
  shippable asset — if counsel says no, that compute is wasted. The safe default is to build the
  **B1 (FLAME 2023 Open)** path, which is commercially clearable today with attribution and no
  EULA dependency.

---

## What must change to flip SHIP-CLEARED to "yes"
- **Path B2:** Record written legal confirmation (qualified counsel, ideally with Epic's written
  sign-off) that standalone-GLB-in-web-product delivery is permitted under the current
  MetaHuman/Content/UE EULA — covering the extractable-raw-asset concern (sub-q 1–2), the
  Mesh-to-MetaHuman/AI-restriction concern (sub-q 3), and revenue-threshold licensing (sub-q 4).
  Then update this report to `SHIP-CLEARED: yes` with the obligations from sub-q 5 wired in.
- **Path B1 (recommended safe default):** Route `face-reconstructor` to switch the base to
  **FLAME 2023 Open (CC-BY-4.0)** per plan §B1; independently clear the DECA/EMOCA fitter
  license; wire in the CC-BY + MIT + Apache notices. That path is clearable **without** the Epic
  EULA lawyer question.

---

## Final acceptance determination (this session)
- **Gate honored:** No `out/head_arkit.glb`, empty `out/recon` + `out/shapes` — **no
  Epic-EULA-governed asset was produced or shipped.** The prior `SHIP-CLEARED: no` was respected.
- **Built-file scan:** all installed viewer deps permissive (MIT/ISC/BSD/Apache-2.0); no
  FaceVerse, no standard FLAME, no study FBX, no model weights, no `.uasset` in the tree.
- **One latent flag:** four Epic MetaHuman-derived demo GLBs are bundled inside
  `vendor/metahuman-to-glb/docs/` — not referenced by our viewer, not shipped, but must be
  **excluded from any future deployment bundle** (Final Acceptance Pass §D).

**Governing verdict = Track B1 (active track this run). B2 is prior record.**

**SHIP-CLEARED: no**

- **B1 (active) — blocked by the FITTER, not the texture.** The texture question is RESOLVED and
  VIABLE: MPI FLAME texture is confirmed CC-BY-NC-SA-4.0 (BARRED); the commercial build uses a
  per-subject **image-baked** albedo (clean). What remains open is the **reconstruction fitter +
  its pretrained weights**: DECA (non-commercial) and EMOCA (non-commercial) are BARRED, and
  Arc2Avatar is BARRED as-run because its ID guidance pulls **InsightFace/ArcFace pretrained models
  (non-commercial)** (+ SD OpenRAIL, + 3DGS format mismatch). See the B1 GOVERNING SECTION above.
- **B2 (prior record) — remains blocked** pending human-lawyer Epic MetaHuman/Content/UE EULA review.
- Nothing was built or shipped this session (license-verification only); the gate was honored.

**Authored-code scan (2026-07-04) is COMPLETE — see "FINAL ACCEPTANCE — B1 AUTHORED-CODE LICENSE
SCAN" above.** The code stays license-clean (YES): the fit is pure optimization against MediaPipe
landmarks with **no** DECA/EMOCA/ArcFace/nvdiffrast/SD/3DGS import or weight anywhere; the FLAME
loader is clean-room; the texture is pure image-bake + mirror + `cv2.inpaint` (no statistical prior);
deps are all permissive (PyTorch3D BSD-3, MediaPipe/OpenCV Apache-2.0, trimesh/three MIT, numpy/scipy
BSD). **The former biggest gap — the fitter — is resolved in the code.** What still blocks ship is
provenance + notices + photo-rights, none of which is a code change.

**Conditions that flip Track B1 to `SHIP-CLEARED: yes` (all required) — none is a code fix:**
1. **FLAME 2023 Open provenance + landmark-embedding origin (operator, pod).** Download the CC-BY
   "FLAME 2023 (Open)" release; confirm **in writing** that the UV template AND the landmark-embedding
   file come from the **Open (CC-BY) release itself** — NOT the CC-BY-NC-SA texture package, NOT a
   DECA/EMOCA mirror. If Open ships no embedding, **self-author** a clean one; never import the NC one.
   (Resolves the `face-reconstructor` landmark-embedding flag.) [AC-9(a)]
2. **Attribution wiring — currently ABSENT (the concrete remaining build item).** Add a credits screen
   + bundled `THIRD-PARTY-NOTICES` to the viewer/`dist/`: FLAME 2023 Open CC-BY credit + FLAME paper
   cite (exact text in B1-1); MediaPipe Apache-2.0 **license text + NOTICE** from upstream
   `google-ai-edge/mediapipe` (npm ships neither) — required because `dist/mediapipe/wasm/*` already
   redistributes the binaries; three.js MIT (banner already in-bundle, also list it); PyTorch3D BSD-3
   + OpenCV Apache NOTICE only if the **pod** code/binaries are redistributed (not triggered by the web
   product alone). [AC-9(b)]
3. **Image-baked texture only, verified on pod** — `bake_summary.json.no_statistical_albedo_prior ==
   true`, no BFM/AlbedoMM/MPI-texture data side-loaded. [AC-9(c)]
4. **Input-photo rights — HUMAN-LEGAL, flag not adjudicated** — ownership + subject consent +
   biometric/publicity/GDPR/BIPA in `random-person.jpeg`. Counsel must clear; this gate does not. [AC-9(d)]

**The scaffold is clean; ship is blocked only by provenance + notices + photo-rights, and nothing was
built or shipped this session.** FaceVerse, standard FLAME, the MPI FLAME texture space, DECA/EMOCA
weights, InsightFace/ArcFace models, and Basel/AlbedoMM albedo must **never** reach the shipped artifact.
