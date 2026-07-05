# recon/ -- Track B1 (FLAME 2023 Open, COMMERCIAL) reconstruction pipeline.
#
# Authored on the CPU-only box; EXECUTES ONLY ON THE GPU POD (RTX 6000 Ada).
# Every module calls recon.pod_guard.require_pod() before doing any compute.
#
# Stages (run in order via scripts/run_recon_b1.sh):
#   1. recon.landmarks       MediaPipe FaceLandmarker -> out/recon/landmarks.npz
#   2. recon.fit_flame       optimization fit -> neutral.ply / id_params.npz /
#                            faces.npy / expression_basis.npz
#   3. recon.bake_texture    photo -> FLAME UV albedo -> out/recon/albedo.png
#   4. recon.verify_outputs  measured asserts -> out/recon/recon_run_manifest.json
