"""Track B1 ARKit rigging stage (arkit-rigger).

Consumes out/recon/ (face-reconstructor) and produces out/shapes/:
52 ARKit-named blendshape delta meshes in identical FLAME topology,
arkit_manifest.json, rig_report.md. POD-ONLY execution (recon.pod_guard);
nothing in this package computes on the authoring box.
"""
