# Image → Animatable ARKit Avatar → GLB: Revised Implementation Plan
### Reconciling the FaceVerse mesh workflow with the ARKit-rigging / GLB / commercial goals raised in the review screenshots

> **What changed and why this document exists.** The three review screenshots (Copilot's analysis) reframed the goal from the earlier docs. The target is no longer just "a PLY mesh that opens in Blender" — it is now explicitly **single image → 3D head → rigged with 52 ARKit blendshapes → exported as GLB → usable in Blender / three.js, ideally commercially licensable.** This plan folds that goal into the working FaceVerse pipeline from the previous document, **and corrects several load-bearing claims in the screenshots that do not survive verification.** Every correction below is sourced, because building on a wrong premise (a repo that doesn't do what was claimed, or a license status that's out of date) is exactly the silent-failure mode the original directive documents were written to prevent.

---

## 0. Executive summary — read this first

Three claims in the screenshots are wrong or outdated, and they change the plan:

1. **The recommended repo is misnamed and misunderstood.** The screenshots cite `smorchj/MH-FaceScan-to-glb` as "takes a MetaHuman FaceScan → rigs with 51 ARKit blendshapes → 100% commercial." The actual repo is **`smorchj/metahuman-to-glb`**, and it does **not** take an image or a "FaceScan." It exports an **already-built UE 5.7 MetaHuman** to GLB. It requires **Unreal Engine 5.7, a `.uproject`, and a MetaHumanCharacter you already created.** It is a *MetaHuman → GLB converter*, not an *image → avatar* system. Its MIT license covers the *scripts*, not the MetaHuman assets (those are governed by Epic's MetaHuman EULA). So it is **not a drop-in for "photo → rigged avatar."** It's the *back half* of a pipeline whose *front half* (getting a MetaHuman that looks like your photo) is the hard, separate part.

2. **The FLAME "commercial killer" verdict is out of date.** The screenshots correctly note FLAME's standard license is non-commercial. But as of **November 2025, MPI released the FLAME 2023 Open Model under CC-BY-4.0**, which **is** commercially usable with attribution. So "FLAME blocks commercial use" is no longer categorically true — it depends on which FLAME model you use.

3. **FaceVerse is *also* non-commercial** — a fact the earlier FaceVerse doc got wrong (it claimed BSD-2/MIT). The FaceVerse model and dataset are **Tsinghua University, non-commercial research only.** So FaceVerse is excellent for a research/personal/internal tool, but it is **not** a commercial-clearance path on its own. This matters and is corrected in §6.

**The practical consequence:** there are two clean tracks, and you should pick based on whether this is a *commercial product* or a *research/personal/internal tool*. This plan gives both, and does not pretend one repo does everything.

| | **Track A — Research / Personal / Internal** | **Track B — Commercial product** |
|---|---|---|
| Reconstruction | **FaceVerse v4** (image → mesh, fast, full head) | **DECA/EMOCA-class** fit onto **FLAME 2023 Open (CC-BY-4.0)**, *or* MetaHuman route |
| Rigging to ARKit | FaceVerse's Apple-52 expression mapping → shape keys | FLAME→ARKit shape keys, *or* MetaHuman's native 51 ARKit blendshapes |
| Export | GLB via Blender (§4) | Same GLB export (§4) |
| License to ship? | **No** (FaceVerse non-commercial) | **Yes**, if every component is commercial-cleared (§6) |
| Effort | Low — mostly the previous doc + rigging | High — model licensing + MetaHuman or FLAME-Open fitting |

If you only need it to *work* for yourself, do **Track A**; it's mostly done. If you need to *sell* it, do **Track B** and read §6 carefully first, because the licensing — not the code — is the real constraint.

---

## 1. What the screenshots got right (keep these)

Not everything needs correcting. These points from the review are sound and are incorporated:

- **The end artifact should be GLB**, not just PLY. GLB carries mesh + blendshapes (morph targets) + textures in one file, and is what three.js and most engines consume. PLY was only ever the intermediate geometry. ✅
- **52 ARKit blendshapes is the right rig target** for face-tracking-driven animation (jawOpen, mouthSmile_L/R, browInnerUp, eyeBlink_L/R, etc.). Apple's ARKit defines 52 (the screenshots sometimes say 51 — MetaHuman bakes 51 because `browInnerUp` is a single bilateral shape; the ARKit spec itself lists 52). ✅
- **Blender is the right hub** for the mesh → shape-key → GLB step, via its native shape-key system + glTF exporter. ✅
- **MediaPipe can drive the animation** (its 52 blendshape coefficients map directly to ARKit names) — good for the *animation/driving* side, separate from *building the rig*. ✅
- **three.js is the right viewer** for a web deliverable. ✅

The disagreement is not about the *destination*. It's about which *reconstruction+rig* path actually gets you there given (a) single-image input and (b) license constraints.

---

## 2. The corrected architecture

```
                          ┌─────────────────────────────────────────────┐
                          │              SINGLE INPUT IMAGE              │
                          └───────────────────────┬─────────────────────┘
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 │                                                                   │
        TRACK A (research/personal)                                    TRACK B (commercial)
                 │                                                                   │
   ┌─────────────▼─────────────┐                              ┌──────────────────────▼───────────────────────┐
   │ FaceVerse v4 (ResNet50)   │                              │  Option B1: FLAME 2023 Open (CC-BY-4.0)        │
   │ image → full-head mesh    │                              │   + DECA/EMOCA-style single-image fit          │
   │ (eyes/teeth/tongue)       │                              │   → FLAME mesh + FLAME expression basis        │
   │ + Apple-52 expr mapping   │                              │                                                │
   └─────────────┬─────────────┘                              │  Option B2: MetaHuman route                    │
                 │                                            │   image → MetaHuman (Mesh-to-MetaHuman / MHC)   │
                 │                                            │   → smorchj/metahuman-to-glb (51 ARKit baked)  │
                 │                                            └──────────────────────┬───────────────────────┘
                 │                                                                   │
   ┌─────────────▼───────────────────────────────────────────────────────────────────▼──────────────┐
   │  BLENDER: build 52 ARKit shape keys on the mesh, assign ARKit-standard names, verify deltas       │
   │  (Track A: derive from FaceVerse expr basis;  B1: from FLAME expr basis;  B2: already baked)       │
   └─────────────────────────────────────────────────────┬────────────────────────────────────────────┘
                                                          │
                          ┌───────────────────────────────▼────────────────────────────────┐
                          │  EXPORT GLB (glTF 2.0): mesh + morph targets + textures          │
                          │  Blender glTF exporter, "Shape Keys" + "Shape Key Normals" on,    │
                          │  Draco compression optional                                       │
                          └───────────────────────────────┬────────────────────────────────┘
                                                          │
                 ┌────────────────────────────────────────┴─────────────────────────────────┐
                 │  three.js viewer: load GLB, drive morphTargetInfluences[] from MediaPipe   │
                 │  FaceLandmarker 52 blendshape coefficients (live webcam or video)          │
                 └───────────────────────────────────────────────────────────────────────────┘
```

The critical insight the screenshots blurred: **reconstruction (build the head), rigging (add the 52 blendshapes), and driving (animate them from face tracking) are three separate problems.** MediaPipe solves *driving*. FaceVerse/FLAME/MetaHuman solve *reconstruction*. The *rigging* step — attaching a correctly-named 52-target blendshape set to your specific reconstructed mesh — is the connective tissue, and is where most of the real work in this plan lives (§3).

---

## 3. Track A (recommended default): FaceVerse → ARKit-rigged GLB

This extends the previous document. Steps 0–6 of that doc (environment, model download, `run.py --save_ply True`, mesh verification, axis-fix conversion) are unchanged — do those first. What follows is the *new* rigging + GLB layer on top.

### 3.1 Why FaceVerse already has a head start on ARKit

FaceVerse's v2+ models explicitly **fit their expression components to the 52 Apple blendshapes** — the repo ships an `exp_name_list` in `faceverse_simple_v2.npy` giving the mapping between FaceVerse expression axes and the 52 ARKit names. This is the single biggest reason FaceVerse is the pragmatic default for an *ARKit* target: the model was deliberately aligned to Apple's 52, so you are transcribing an existing correspondence rather than authoring 52 blendshapes from scratch.

### 3.2 Generate the neutral mesh + the 52 expression meshes

The rig is built from **deltas**: for each ARKit blendshape *k*, you need the mesh at full activation of *k* minus the neutral mesh. FaceVerse gives you both because its expression basis maps to the 52.

Procedure (conceptual — adapt to FaceVerse v4's parameter API):

1. Fit the image → get identity/shape parameters + neutral expression. Export the **neutral** mesh (this is your base, from the previous doc's `--save_ply`).
2. For each of the 52 ARKit shapes, set that expression coefficient to its max (per the `exp_name_list` mapping) with all others at 0, keeping identity fixed, and export that mesh.
3. You now have `neutral.ply` + 52 `expr_<arkit_name>.ply`, all in **identical topology** (this is guaranteed because they're the same FaceVerse template — the property the old 3DGS path could never give you).

```python
# build_arkit_shapes.py  (pseudocode against FaceVerse's model API)
# Produces neutral.ply + one mesh per ARKit blendshape, all same topology.
import numpy as np
# 'fv' = loaded FaceVerse model; 'id_params' = fitted identity for THIS image
# 'exp_name_list' from faceverse_simple_v2.npy maps ARKit name -> expression index

ARKIT_52 = [  # Apple's canonical order/names
  "browDownLeft","browDownRight","browInnerUp","browOuterUpLeft","browOuterUpRight",
  "cheekPuff","cheekSquintLeft","cheekSquintRight","eyeBlinkLeft","eyeBlinkRight",
  "eyeLookDownLeft","eyeLookDownRight","eyeLookInLeft","eyeLookInRight","eyeLookOutLeft",
  "eyeLookOutRight","eyeLookUpLeft","eyeLookUpRight","eyeSquintLeft","eyeSquintRight",
  "eyeWideLeft","eyeWideRight","jawForward","jawLeft","jawOpen","jawRight",
  "mouthClose","mouthDimpleLeft","mouthDimpleRight","mouthFrownLeft","mouthFrownRight",
  "mouthFunnel","mouthLeft","mouthLowerDownLeft","mouthLowerDownRight","mouthPressLeft",
  "mouthPressRight","mouthPucker","mouthRight","mouthRollLower","mouthRollUpper",
  "mouthShrugLower","mouthShrugUpper","mouthSmileLeft","mouthSmileRight","mouthStretchLeft",
  "mouthStretchRight","mouthUpperUpLeft","mouthUpperUpRight","noseSneerLeft","noseSneerRight",
  "tongueOut",
]

neutral_verts = fv.build_mesh(id_params, exp=np.zeros(fv.n_exp))     # base
save_ply("out/neutral.ply", neutral_verts, fv.faces, colors=fv.tex(id_params))

for name in ARKIT_52:
    idx = exp_name_list.index(name)          # map ARKit -> FaceVerse expr axis
    exp = np.zeros(fv.n_exp); exp[idx] = 1.0 # full activation of just this shape
    verts = fv.build_mesh(id_params, exp=exp)
    save_ply(f"out/expr_{name}.ply", verts, fv.faces)   # SAME topology as neutral
```

> If a given ARKit name has no clean FaceVerse axis (a few, e.g. `tongueOut`, may be weak or absent depending on model version), record it as "unsupported" and either (a) leave that shape key flat, or (b) borrow a delta from the free MetaHuman-52 reference FBX (study-only license — see §6 caveat). Don't fabricate a shape; a flat unsupported shape is honest and won't break the rig.

### 3.3 Assemble shape keys in Blender (script, headless-capable)

Blender's shape-key system *is* the morph-target system glTF exports. Load neutral as the base mesh, then add each expression mesh as a shape key equal to `expr - neutral`.

```python
# blender_build_rig.py  -- run with:  blender --background --python blender_build_rig.py
import bpy, glob, os

def load_ply(path):
    bpy.ops.wm.ply_import(filepath=path)   # Blender 4.x importer
    return bpy.context.selected_objects[0]

base = load_ply("out/neutral.ply")
base.name = "Head"
base.shape_key_add(name="Basis")           # required base shape key

for path in sorted(glob.glob("out/expr_*.ply")):
    name = os.path.splitext(os.path.basename(path))[0].replace("expr_", "")
    tmp = load_ply(path)
    key = base.shape_key_add(name=name, from_mix=False)  # ARKit-named shape key
    for i, v in enumerate(tmp.data.vertices):            # topology matches -> index-aligned
        key.data[i].co = v.co
    bpy.data.objects.remove(tmp, do_unlink=True)

# name matters: glTF morph target names = shape key names = ARKit names (drivers rely on this)
bpy.ops.wm.save_as_mainfile(filepath="out/head_rigged.blend")
```

The index-alignment (`key.data[i].co = v.co`) is only valid *because every mesh shares topology*. That is the whole payoff of using a template-based model (FaceVerse/FLAME) instead of marching-cubes (MonoNPHM) or splats (the old path): consistent vertex order makes rigging a one-liner instead of a correspondence-solving problem.

### 3.4 Export GLB with morph targets

```python
# append to blender_build_rig.py
bpy.ops.export_scene.gltf(
    filepath="out/head_arkit.glb",
    export_format="GLB",
    export_morph=True,                 # <-- exports shape keys as morph targets
    export_morph_normal=True,          # smoother deformation
    export_draco_mesh_compression_enable=True,   # optional, smaller file
)
```

You now have `head_arkit.glb`: one file, your subject's head, 52 ARKit-named morph targets, vertex-colored/textured. That is the deliverable the screenshots were aiming at — reached via FaceVerse rather than the mis-described MetaHuman repo.

### 3.5 Drive it from MediaPipe in three.js

MediaPipe's FaceLandmarker returns `faceBlendshapes` — 52 categories with the **same ARKit names**. Because your morph targets are named identically, driving is a name lookup:

```javascript
// three.js: mesh.morphTargetDictionary maps name -> index
const results = faceLandmarker.detectForVideo(video, performance.now());
const blends = results.faceBlendshapes[0].categories;   // 52 {categoryName, score}
for (const b of blends) {
  const idx = headMesh.morphTargetDictionary[b.categoryName];
  if (idx !== undefined) headMesh.morphTargetInfluences[idx] = b.score;
}
```

This is the clean payoff of naming discipline: reconstruction (FaceVerse), rig (Blender shape keys), and driver (MediaPipe) all agree on the 52 ARKit strings, so the seams disappear.

---

## 4. Track B: the commercial path (when you need to ship)

If this is a product, FaceVerse's non-commercial license (§6) rules out Track A for shipping. Two commercial-clearable options:

### 4.1 Option B1 — FLAME 2023 Open (CC-BY-4.0) + single-image fit

- Use the **FLAME 2023 Open model** (CC-BY-4.0, commercial OK with attribution). *Not* the standard FLAME model, which is non-commercial.
- Fit it to the image with a **DECA/EMOCA-style** encoder (check each fitter's *own* license — the FLAME model being CC-BY doesn't automatically make a given fitting codebase commercial; DECA's weights and EMOCA have their own terms).
- FLAME has a well-documented **FLAME→ARKit** correspondence effort in the community, and its expression + jaw-pose basis maps to the ARKit shapes similarly to §3.2. Build the 52 shape keys from FLAME's expression basis, then §3.3–3.4 identically.
- **Attribution obligation:** CC-BY-4.0 requires you credit FLAME 2023 Open in your product. Satisfy it (e.g. an in-app credits screen).

### 4.2 Option B2 — MetaHuman route (this is where `metahuman-to-glb` actually fits)

This is the corrected home for the screenshot's repo. The pipeline is:

1. **Image → MetaHuman.** Use Epic's **Mesh-to-MetaHuman** / MetaHuman Creator to get a MetaHuman resembling your subject (this is the hard front-half; it needs a mesh or careful sculpt, and is *not* a one-click photo import for arbitrary faces).
2. **MetaHuman → GLB with 51 ARKit blendshapes** using **`smorchj/metahuman-to-glb`** (UE 5.7, `.uproject`, MetaHumanCharacter under `/Game/<Name>/`, Blender 5.x). Its five stages bake ARKit shapes natively via Sequencer + RigLogic and transfer them to the GLB by KDTree position match.
3. **License reality:** the *scripts* are MIT, but **MetaHuman assets are governed by Epic's MetaHuman/Unreal EULA**, which permits use within Unreal-ecosystem projects but has its own terms about using MetaHuman-derived assets outside Unreal. **Get Epic's EULA reviewed for your specific "export MetaHuman to standalone GLB and ship in a web product" use case** — that is a lawyer question, not a code question, and the MIT license on the conversion scripts does *not* clear it.

**Bottom line for Track B:** B1 (FLAME 2023 Open) is the cleaner *self-contained* commercial path if its fitter licenses check out; B2 (MetaHuman) gives higher visual quality and a ready 51-ARKit rig but drags in Epic's EULA and requires Unreal + a non-trivial image→MetaHuman step. Neither is "one repo, one command," which is the expectation the screenshots created and that this plan corrects.

---

## 5. Why not just use the screenshot's other suggestions

Briefly, so these are evaluated rather than silently dropped:

- **MediaPipe→OBJ converters** (`mediapipe-facemesh-to-obj-blender`, Knorkje's shape-key tool): these convert MediaPipe's 468-point face *mesh* to OBJ or generate shapekeys from MediaPipe. They're useful for the *driving/animation* side but MediaPipe's face mesh is a **coarse tracking mesh, not a personalized head reconstruction** — it won't give you a good-looking avatar of your subject, only a generic low-res face proxy. Fine as an animation source (§3.5), not as the reconstruction. The screenshots partly conflate "get 52 blendshape *coefficients*" (MediaPipe is great) with "get a 52-blendshape *rigged head*" (MediaPipe alone is not).
- **Meshborn** (AI avatar service): a hosted text/image→avatar service. Viable if you want to *outsource* the whole problem and accept a third-party service's terms/quality/pricing, but it's a dependency on someone else's black box, not an open-source pipeline you control. Verify its output actually includes named ARKit morph targets before relying on it; the screenshots themselves flag it as very new and unverified.
- **Ready Player Me**: commercial avatar SaaS with ARKit-compatible morph targets and a real license — a legitimate Track-B alternative if you'd rather license a service than build. Again, a business/terms decision, not a code one.

---

## 6. The licensing table (verified — this is the part that actually gates shipping)

This corrects the earlier FaceVerse doc and the screenshots. **Confirm current terms yourself before shipping; licenses change (FLAME's did in Nov 2025).**

| Component | License (verified) | Commercial? | Note |
|---|---|---|---|
| **FaceVerse model + dataset** | Tsinghua Univ., **non-commercial research only** | ❌ No | Corrects earlier "BSD/MIT" claim. Fine for personal/research/internal. |
| **FLAME (standard)** | MPI non-commercial | ❌ No | Research/education/artistic only. |
| **FLAME 2023 Open** | **CC-BY-4.0** | ✅ Yes (attribution) | Released Nov 2025. This is the commercial-clearable FLAME. |
| **DECA / EMOCA (fitters)** | Each has own terms | ⚠️ Check | CC-BY on the *model* ≠ commercial on the *fitting code/weights*. Verify per repo. |
| **MediaPipe (FaceLandmarker)** | Apache 2.0 | ✅ Yes | Great for driving; blendshape coefficients are ARKit-named. |
| **`smorchj/metahuman-to-glb`** | MIT (scripts) | ⚠️ Scripts yes; **assets no** | MetaHuman assets governed by Epic's EULA — lawyer question. |
| **MetaHuman assets** | Epic MetaHuman/Unreal EULA | ⚠️ Restricted | Terms about use outside Unreal ecosystem — review for your use case. |
| **three.js** | MIT | ✅ Yes | Viewer. |
| **Ready Player Me / Meshborn** | Commercial SaaS terms | ⚠️ Per-service | Read their terms; these are services, not open source. |

**The single most important takeaway:** the screenshots framed licensing as "FLAME bad, MIT repo good." The verified reality is subtler — **FaceVerse (the thing that actually reconstructs from an image well) is the non-commercial one**, the "MIT repo" is MIT-scripts-over-EULA-assets and isn't image-driven anyway, and **FLAME 2023 Open is now the cleanest commercial base**. If you're shipping, that inverts the screenshots' recommendation.

---

## 7. Concrete build order

**If research/personal/internal (Track A):**
1. Do the previous doc's §0–6 (FaceVerse install, `run.py --save_ply True`, verify `element face N>0`, axis-fix). ✅ already specified there
2. `build_arkit_shapes.py` (§3.2) → neutral + 52 expression meshes, same topology
3. `blender_build_rig.py` (§3.3–3.4) → `head_arkit.glb` with 52 named morph targets
4. three.js viewer (§3.5) → drive from MediaPipe webcam
5. Ship *internally only* (license). Done.

**If commercial (Track B):**
1. Decide B1 (FLAME 2023 Open) vs B2 (MetaHuman) — read §6 first
2. **B1:** get FLAME 2023 Open (CC-BY-4.0) + a fitter whose license you've cleared → fit image → FLAME mesh + expression basis → build 52 shape keys → §3.3–3.5
3. **B2:** Mesh-to-MetaHuman → `smorchj/metahuman-to-glb` (UE 5.7 + Blender 5.x) → GLB with 51 ARKit baked → §3.5. **Clear Epic's EULA for standalone-GLB shipping first.**
4. Satisfy attribution (B1: credit FLAME 2023 Open) / EULA (B2) obligations in-product

---

## 8. How this maps back to the Arc2Avatar directive documents

The original Arc2Avatar pipeline (Arc2Face + SDS + 3DGS + FLAME) remains a valid *research* system, but note two things this revision surfaces:

- **Its output is splats, which don't mesh into Blender/GLB** (established in the previous doc). For an *ARKit-rigged GLB*, the FaceVerse/FLAME mesh path here replaces Modules F/H's splat export, exactly as the previous doc argued — and now additionally supplies the 52-blendshape rig the screenshots want.
- **Its FLAME dependency is non-commercial** *unless* you swap to **FLAME 2023 Open (CC-BY-4.0)**. If Arc2Avatar is ever meant to ship, that swap (feasible because Module V's Directive 77–79 already abstracts `MeshModel` behind an interface) is the licensing-critical change — precisely the kind of pluggable-topology swap those directives were designed to allow.

Wire this in behind the same config flag pattern the previous doc suggested: `export_target: gaussian_splat | blender_mesh | arkit_glb`, where `arkit_glb` runs §3's rig+export on the FaceVerse (or FLAME-Open) reconstruction. That keeps one codebase serving research (splats), Blender (mesh), and the new animatable-avatar (GLB) goals without forking.
