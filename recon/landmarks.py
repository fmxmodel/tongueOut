"""Stage 1 -- MediaPipe FaceLandmarker on the input photo.

POD-ONLY (pod_guard). Produces, under out/recon/:
  input_image.png     EXIF-normalized RGB decode of the input photo. This is
                      the SINGLE SOURCE OF PIXELS: fit_flame.py and
                      bake_texture.py consume this file, never the original
                      JPEG, so EXIF rotation can never desynchronize stages.
  landmarks.npz       all 478 landmarks (normalized + pixel), the iBUG-68
                      subset per recon/mp_flame_correspondence.py, image size,
                      MediaPipe's 52 blendshape scores + facial transform.
  landmarks_debug.png the 68 correspondence picks drawn + numbered on the
                      photo -- the REQUIRED human verification of the
                      self-authored MediaPipe<->FLAME correspondence.
  mediapipe_blendshapes_photo.json
                      MediaPipe's ARKit-named blendshape scores FOR THE PHOTO.
                      Reference material only (photo expression state / QA
                      sanity); it says NOTHING about which ARKit shapes the
                      FLAME rig can express.

Run:  python -m recon.landmarks
"""

import json
import sys
import time

import numpy as np

from . import config as C
from .mp_flame_correspondence import MEDIAPIPE_IBUG68
from .pod_guard import require_pod


def load_canonical_image():
    """Decode the input photo with EXIF orientation applied; save the
    canonical PNG that all later stages must use. Returns HxWx3 uint8 RGB."""
    from PIL import Image, ImageOps

    if not C.INPUT_IMAGE.is_file():
        sys.exit(
            f"[landmarks FATAL] input image not found: {C.INPUT_IMAGE}\n"
            "Copy the repo's random-person.jpeg to /workspace/inputs/ "
            "(models/README.md section 5)."
        )
    img = Image.open(C.INPUT_IMAGE)
    img = ImageOps.exif_transpose(img)  # bake EXIF rotation into pixels
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    Image.fromarray(rgb).save(C.CANONICAL_IMAGE)
    print(f"[landmarks] canonical image {rgb.shape[1]}x{rgb.shape[0]} -> {C.CANONICAL_IMAGE}")
    return rgb


def run_facelandmarker(rgb: np.ndarray):
    """Run MediaPipe FaceLandmarker (IMAGE mode, 1 face, blendshapes on)."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not C.MP_TASK_PATH.is_file():
        sys.exit(
            f"[landmarks FATAL] face_landmarker.task not found at {C.MP_TASK_PATH}. "
            "Run scripts/pod_setup_b1.sh (step 5) first."
        )
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(C.MP_TASK_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.5,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        sys.exit(
            "[landmarks FATAL] MediaPipe detected NO face in the input image. "
            "The fit cannot proceed. Check the photo / detection confidence."
        )
    if len(result.face_landmarks) > 1:
        print(f"[landmarks WARN] {len(result.face_landmarks)} faces detected; using face 0.")
    return result


def draw_debug_overlay(rgb: np.ndarray, lmk478_px: np.ndarray) -> None:
    """Draw all 478 points faintly + the 68 correspondence picks numbered."""
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for x, y in lmk478_px:
        cv2.circle(bgr, (int(round(x)), int(round(y))), 1, (90, 90, 90), -1)
    for ibug_idx, mp_idx in enumerate(MEDIAPIPE_IBUG68):
        x, y = lmk478_px[mp_idx]
        p = (int(round(x)), int(round(y)))
        cv2.circle(bgr, p, 3, (0, 255, 0), -1)
        cv2.putText(bgr, str(ibug_idx), (p[0] + 3, p[1] - 3),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(C.LANDMARKS_DEBUG_PNG), bgr)
    print(f"[landmarks] correspondence overlay -> {C.LANDMARKS_DEBUG_PNG}")
    print("[landmarks] ACTION: visually confirm every numbered pick sits on the "
          "right facial feature (self-authored correspondence verification).")


def main() -> None:
    require_pod()
    C.ensure_out_dirs()
    t0 = time.time()

    rgb = load_canonical_image()
    h, w = rgb.shape[:2]
    result = run_facelandmarker(rgb)

    lms = result.face_landmarks[0]
    n = len(lms)
    if n < int(MEDIAPIPE_IBUG68.max()) + 1:
        sys.exit(
            f"[landmarks FATAL] FaceLandmarker returned {n} landmarks; the "
            f"correspondence table needs >= {int(MEDIAPIPE_IBUG68.max()) + 1}. "
            "Wrong .task model variant?"
        )
    print(f"[landmarks] {n} landmarks returned (expected 478 for face_landmarker.task)")

    lmk478_norm = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float64)
    lmk478_px = np.stack([lmk478_norm[:, 0] * w, lmk478_norm[:, 1] * h], axis=1)
    ibug68_px = lmk478_px[MEDIAPIPE_IBUG68]

    # MediaPipe's 52 ARKit-named blendshape scores for the PHOTO (reference only)
    bs_names, bs_scores = [], []
    if result.face_blendshapes:
        for cat in result.face_blendshapes[0]:
            bs_names.append(cat.category_name)
            bs_scores.append(float(cat.score))
    tf = (np.array(result.facial_transformation_matrixes[0], dtype=np.float64)
          if result.facial_transformation_matrixes else np.eye(4))

    np.savez(
        C.LANDMARKS_NPZ,
        lmk478_norm=lmk478_norm,
        lmk478_px=lmk478_px,
        ibug68_px=ibug68_px,
        ibug68_mp_idx=MEDIAPIPE_IBUG68,
        image_hw=np.array([h, w], dtype=np.int64),
        blendshape_names=np.array(bs_names),
        blendshape_scores=np.array(bs_scores, dtype=np.float64),
        facial_transform=tf,
        input_image=str(C.INPUT_IMAGE),
    )
    print(f"[landmarks] landmarks -> {C.LANDMARKS_NPZ}")

    with open(C.BLENDSHAPES_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "note": (
                    "MediaPipe FaceLandmarker blendshape scores for the INPUT PHOTO. "
                    "Photo expression state / QA reference ONLY. NOT a statement of "
                    "FLAME rig coverage -- see expression_basis_notes.json."
                ),
                "scores": dict(zip(bs_names, bs_scores)),
            },
            f,
            indent=2,
        )
    print(f"[landmarks] photo blendshape reference -> {C.BLENDSHAPES_JSON}")

    draw_debug_overlay(rgb, lmk478_px)
    print(f"[landmarks] DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
