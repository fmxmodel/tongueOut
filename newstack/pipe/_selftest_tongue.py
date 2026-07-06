#!/usr/bin/env python3
"""Offline self-test for tongue_synth -- synthetic geometry, NO ICT asset.

Fabricates a full-topology (26719-vert) array where the documented index
ranges hold plausible mouth geometry:
  - teeth [17039:21451]:  cluster in x[-3.2,3.2] y[-2.0,-0.5] z[5.0,10.8]
  - gums  (front of the "Gums and tongue" region): each vert EXACTLY 0.3 cm
    from a tooth vert -> must be EXCLUDED (< 1 cm clearance), deterministically
  - tongue slab (rest of the region): y <= -3.5, i.e. >= 1.5 cm from every
    tooth -> must ALL be selected, deterministically
  - everything else: far background -> must not move at all

Asserts:
  (a) selection == the fabricated tongue slab, exactly; moved verts are a
      subset of it (the single root-plane vert has w=0 by construction)
  (b) EXACTLY 0.0 delta on gums, teeth, face, and everything else
  (c) tip z increases by exactly AMOUNT_Z_CM (target +4-5 cm) and the small
      +y lift equals AMOUNT_Y_CM at the tip
  (d) forward weighting is monotonic non-decreasing in z; root ~0, tip max
  plus: guard paths (mesh smaller than the teeth range) return zero deltas
  without crashing, and stats/no-stats call forms agree.

Run:  python3 newstack/pipe/_selftest_tongue.py   (exit 0 + PASS lines)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import N_VERTS  # noqa: E402
from tongue_synth import (AMOUNT_Y_CM, AMOUNT_Z_CM, GUMS_TONGUE_END,  # noqa: E402
                          GUMS_TONGUE_START, TEETH_END, TEETH_START,
                          TOOTH_CLEARANCE_CM, synth_tongue_out_delta)

N_TONGUE = 777  # fabricated tongue-slab size (~real-world ~760)


def build_synthetic():
    rng = np.random.default_rng(7)
    v = rng.uniform([-8.0, 2.0, -8.0], [8.0, 12.0, 8.0], size=(N_VERTS, 3))

    # teeth cluster (front teeth reach z ~ 10.8)
    n_teeth = TEETH_END - TEETH_START
    v[TEETH_START:TEETH_END] = rng.uniform(
        [-3.2, -2.0, 5.0], [3.2, -0.5, 10.8], size=(n_teeth, 3))

    region = np.arange(GUMS_TONGUE_START, GUMS_TONGUE_END)
    gums_idx = region[: len(region) - N_TONGUE]
    tongue_idx = region[len(region) - N_TONGUE:]

    # gums: EXACTLY 0.3 cm from a tooth vert -> nearest-tooth dist <= 0.3 < 1.0
    src = np.arange(len(gums_idx)) % n_teeth
    v[gums_idx] = v[TEETH_START + src] + np.array([0.0, -0.3, 0.0])

    # tongue slab: y <= -3.5 vs teeth y >= -2.0 -> dist >= 1.5 > 1.0, always
    v[tongue_idx] = rng.uniform(
        [-3.2, -5.4, 1.0], [3.2, -3.5, 9.1], size=(N_TONGUE, 3))
    return v, tongue_idx, gums_idx


def main():
    ok = 0

    def check(cond, msg):
        nonlocal ok
        if not cond:
            print(f"FAIL: {msg}")
            sys.exit(1)
        ok += 1
        print(f"  PASS {msg}")

    v, tongue_idx, gums_idx = build_synthetic()
    delta, st = synth_tongue_out_delta(v, return_stats=True)

    check(delta.shape == (N_VERTS, 3), "delta shape (N,3), same vertex count")
    check(st["available"] and st["n_selected"] == N_TONGUE
          and np.array_equal(st["indices"], tongue_idx),
          f"selection == fabricated tongue slab exactly ({N_TONGUE} verts, "
          f"clearance > {TOOTH_CLEARANCE_CM} cm)")

    moved = np.flatnonzero(np.linalg.norm(delta, axis=1) > 0)
    check(np.isin(moved, tongue_idx).all(),
          f"(a) only in-range tongue verts moved ({len(moved)} of {N_TONGUE}; "
          "root plane w=0)")
    check(len(moved) >= N_TONGUE - 1, "(a) all tongue verts except the exact "
          "z_root vert received displacement")

    outside = np.ones(N_VERTS, dtype=bool)
    outside[tongue_idx] = False
    check(np.all(delta[outside] == 0.0), "(b) EXACTLY 0.0 outside tongue set")
    check(np.all(delta[gums_idx] == 0.0), "(b) gums untouched")
    check(np.all(delta[TEETH_START:TEETH_END] == 0.0), "(b) teeth untouched")
    check(np.all(delta[:GUMS_TONGUE_START] == 0.0)
          and np.all(delta[TEETH_END:] == 0.0),
          "(b) face / eyes / everything else untouched")

    tz = v[tongue_idx, 2]
    i_tip = tongue_idx[int(np.argmax(tz))]
    dz_tip, dy_tip = delta[i_tip, 2], delta[i_tip, 1]
    check(abs(dz_tip - AMOUNT_Z_CM) < 1e-9 and 4.0 <= dz_tip <= 5.0,
          f"(c) tip moved +{dz_tip:.2f} cm forward (target +4-5, w(tip)=1)")
    check(abs((v[i_tip, 2] + dz_tip) - st["tip_final_z"]) < 1e-9,
          f"(c) stats tip_final_z consistent ({st['tip_final_z']:.2f} cm)")
    check(abs(dy_tip - AMOUNT_Y_CM) < 1e-9,
          f"(c) small +y lift at tip = {dy_tip:.2f} cm")
    check(np.all(delta[tongue_idx, 2] >= 0.0)
          and np.all(delta[tongue_idx, 1] >= 0.0),
          "(c) all tongue displacement is forward/up, never backward")

    order = np.argsort(tz)
    dz_sorted = delta[tongue_idx, 2][order]
    check(np.all(np.diff(dz_sorted) >= -1e-12),
          "(d) forward weighting monotonic non-decreasing in z")
    check(dz_sorted[0] < 1e-9 < dz_sorted[-1] and dz_sorted[-1] > dz_sorted[0],
          f"(d) root anchored ({dz_sorted[0]:.3f}) < tip ({dz_sorted[-1]:.3f})")
    mid = dz_sorted[len(dz_sorted) // 2]
    check(dz_sorted[0] <= mid <= dz_sorted[-1],
          f"(d) mid-tongue between root and tip ({mid:.3f} cm)")

    d_nostats = synth_tongue_out_delta(v)
    check(np.array_equal(d_nostats, delta), "no-stats call returns same delta")

    # guard: topology without the tongue/teeth region -> zero delta, no crash
    small = np.random.default_rng(1).uniform(-5, 5, size=(5000, 3))
    d_small, st_small = synth_tongue_out_delta(small, return_stats=True)
    check(d_small.shape == (5000, 3) and not d_small.any()
          and not st_small["available"],
          "guard: 5000-vert mesh (no tongue region) -> zero delta, logged")
    d_edge = synth_tongue_out_delta(
        np.zeros((TEETH_END - 1, 3)))
    check(d_edge.shape == (TEETH_END - 1, 3) and not d_edge.any(),
          f"guard: {TEETH_END - 1}-vert mesh (one short of teeth end) -> zero")

    print(f"\nSELF-TEST PASS: {ok}/{ok} checks "
          f"(selected {st['n_selected']} tongue verts; tip z "
          f"{st['z_tip']:.2f} -> {st['tip_final_z']:.2f} cm)")


if __name__ == "__main__":
    main()
