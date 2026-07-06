"""The ARKit-52 name contract + the ICT-FaceKit -> ARKit source mapping.

The 52 strings in ARKIT_52 are LOAD-BEARING: they are simultaneously the
Blender shape-key names, the GLB morph-target names, the MediaPipe
`categoryName` values, and the three.js driver keys. Never edit spelling.

ICT-FaceKit ships its expressions as per-file OBJs named with `_L`/`_R`
suffixes (following ARKit semantics: `_L` = subject's left). Two ARKit names
are bilateral singles that ICT may ship split (browInnerUp, cheekPuff) --
for those we try the single file first, else fold `_L + _R` as a SUMMED
delta. `tongueOut` has no ICT blendshape OBJ, but ICT DOES have real static
tongue geometry, so its delta is SYNTHESIZED numerically from that geometry
(tongue_synth.py) and shipped with source "synthesized-ict-tongue" -- 52/52
supported.

Resolution is data-driven and glob-tolerant: whatever OBJs actually exist in
FaceXModel/ get consumed if mapped, reported if dropped (e.g. pupilDilate_*,
cheekRaiser_* style extras), and any ARKit name with no source is recorded
`unsupported` with a reason -- never fabricated.
"""

ARKIT_52 = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
    "tongueOut",
]
assert len(ARKIT_52) == 52, "ARKit contract must have exactly 52 names"
assert len(set(ARKIT_52)) == 52, "ARKit contract has duplicates"


def _sided(base):
    """ARKit '<base>Left/Right' -> ICT '<base>_L/_R' candidate lists."""
    return {
        base + "Left": [[base + "_L"]],
        base + "Right": [[base + "_R"]],
    }


def _single(name):
    return {name: [[name]]}


def _bilateral_foldable(name):
    """Single ARKit name that ICT may ship as one file OR split _L/_R.
    Candidates tried in order; the split pair is a SUMMED delta."""
    return {name: [[name], [name + "_L", name + "_R"]]}


# ARKit name -> list of candidate source groups (each group = list of ICT OBJ
# stems whose deltas are summed). First group whose files ALL exist wins.
ICT_SOURCE_CANDIDATES = {}
for _b in ["browDown", "browOuterUp", "cheekSquint", "eyeBlink", "eyeLookDown",
           "eyeLookIn", "eyeLookOut", "eyeLookUp", "eyeSquint", "eyeWide",
           "mouthDimple", "mouthFrown", "mouthLowerDown", "mouthPress",
           "mouthSmile", "mouthStretch", "mouthUpperUp", "noseSneer"]:
    ICT_SOURCE_CANDIDATES.update(_sided(_b))
for _s in ["jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
           "mouthFunnel", "mouthLeft", "mouthPucker", "mouthRight",
           "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper"]:
    ICT_SOURCE_CANDIDATES.update(_single(_s))
ICT_SOURCE_CANDIDATES.update(_bilateral_foldable("browInnerUp"))
ICT_SOURCE_CANDIDATES.update(_bilateral_foldable("cheekPuff"))
# tongueOut: no ICT blendshape OBJ exists, so no OBJ candidates -- the delta
# is synthesized from ICT's static tongue geometry instead (SYNTHESIZED_SOURCES).
ICT_SOURCE_CANDIDATES["tongueOut"] = []

assert set(ICT_SOURCE_CANDIDATES) == set(ARKIT_52), (
    "source table must cover exactly the ARKit-52: missing="
    + str(set(ARKIT_52) - set(ICT_SOURCE_CANDIDATES))
    + " extra=" + str(set(ICT_SOURCE_CANDIDATES) - set(ARKIT_52))
)

# ARKit names with no ICT OBJ source whose deltas are SYNTHESIZED numerically
# instead of resolved from expression OBJs. tongueOut is built by
# tongue_synth.synth_tongue_out_delta from ICT's real static tongue geometry
# (the "Gums and tongue" vertex region) -- supported, not fabricated from thin
# air, and gated in s4 (tip must pass the lips; zero delta off the tongue).
SYNTHESIZED_SOURCES = {"tongueOut": "synthesized-ict-tongue"}

# Currently empty: all 52 names have a source (51 ICT OBJs + 1 synthesized).
# Kept so s4 stays honest if a future topology loses a source.
UNSUPPORTED_REASON = {}


def resolve_sources(available_stems):
    """Given the set of expression OBJ stems that actually exist, return
    (mapping, dropped): mapping[arkit_name] = list of stems (empty = unsupported),
    dropped = sorted list of available stems consumed by no ARKit name."""
    mapping = {}
    consumed = set()
    for name in ARKIT_52:
        chosen = []
        for group in ICT_SOURCE_CANDIDATES[name]:
            if all(s in available_stems for s in group):
                chosen = list(group)
                break
        mapping[name] = chosen
        consumed.update(chosen)
    dropped = sorted(set(available_stems) - consumed)
    return mapping, dropped
