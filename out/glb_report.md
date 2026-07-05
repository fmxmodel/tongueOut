# GLB Report — Track B1 (Blender assembly → head_arkit.glb)

> Owner: `blender-glb-builder` (Opus 4.8) · **MEASURED on the GPU pod** by `blender_build_rig.py`.

> Pod-run command: `blender --background --python blender_build_rig.py`


## Topology (byte-locked to the contract)

- vertices: **5023**, faces: **9976**
- faces.npy sha256: `0e1dc70f945eb944daea3a3c23e5700f2b782e0ec302cd3cfb056c677345475f`
- manifest cross-check: faces.npy sha256 matches manifest

## Morph targets (== manifest supported set, exact ARKit spelling)

- morph-target count in GLB: **20** (supported names: 20)
- GLB `extras.targetNames` resolve for three.js `morphTargetDictionary`: **YES**

| # | ARKit morph-target name |
|---|---|
| 0 | `eyeLookDownLeft` |
| 1 | `eyeLookInLeft` |
| 2 | `eyeLookOutLeft` |
| 3 | `eyeLookUpLeft` |
| 4 | `eyeLookDownRight` |
| 5 | `eyeLookInRight` |
| 6 | `eyeLookOutRight` |
| 7 | `eyeLookUpRight` |
| 8 | `jawForward` |
| 9 | `jawLeft` |
| 10 | `jawRight` |
| 11 | `jawOpen` |
| 12 | `mouthFunnel` |
| 13 | `mouthPucker` |
| 14 | `mouthDimpleLeft` |
| 15 | `mouthDimpleRight` |
| 16 | `mouthRollLower` |
| 17 | `mouthRollUpper` |
| 18 | `mouthShrugLower` |
| 19 | `mouthShrugUpper` |

## Unsupported — deliberately NO morph target (viewer no-ops by name)

- a-priori unsupported (4): `cheekPuff`, `cheekSquintLeft`, `cheekSquintRight`, `tongueOut`
- pod-gate-demoted (28): `eyeBlinkLeft`, `eyeSquintLeft`, `eyeWideLeft`, `eyeBlinkRight`, `eyeSquintRight`, `eyeWideRight`, `mouthClose`, `mouthRight`, `mouthLeft`, `mouthSmileLeft`, `mouthSmileRight`, `mouthFrownLeft`, `mouthFrownRight`, `mouthStretchLeft`, `mouthStretchRight`, `mouthPressLeft`, `mouthPressRight`, `mouthLowerDownLeft`, `mouthLowerDownRight`, `mouthUpperUpLeft`, `mouthUpperUpRight`, `browDownLeft`, `browDownRight`, `browInnerUp`, `browOuterUpLeft`, `browOuterUpRight`, `noseSneerLeft`, `noseSneerRight`
- (manifest `demoted_by_gates`: ['eyeBlinkLeft', 'eyeSquintLeft', 'eyeWideLeft', 'eyeBlinkRight', 'eyeSquintRight', 'eyeWideRight', 'mouthClose', 'mouthRight', 'mouthLeft', 'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthPressLeft', 'mouthPressRight', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'noseSneerLeft', 'noseSneerRight'])

## Texture survival

- images in GLB: 1, materials: 1
- baseColor texture present: **True** (baked albedo via FLAME UV, sRGB)

## File / export settings

- `head_arkit.glb`: **16,084,920 bytes**
- `head_rigged.blend`: 2,442,944 bytes
- Draco mesh compression: **OFF** (GLB reports KHR_draco: False; toggle with B1_GLB_DRACO=1)
- exporter flags: `export_morph=True`, `export_morph_normal=True`, `export_format='GLB'`

## Handoff

- `viewer-driver`: load `out/head_arkit.glb`; drive `morphTargetInfluences` by ARKit name — every supported name above resolves 1:1.
- `qa-verifier`: reconcile these morph-target names against `out/shapes/arkit_manifest.json` (supported set) and `out/shapes/shapes_run_manifest.json` (measured verdict). qa-verifier owns the final ACCEPT gate.
