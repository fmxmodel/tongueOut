"""Contamination guard: refuse to compute anywhere except the GPU pod.

The run directive forbids ANY model/numeric compute on the CPU-only authoring
box. Mirroring scripts/pod_setup_b1.sh, every pipeline entrypoint calls
require_pod() first and dies loudly if no NVIDIA GPU is enumerable.

require_cuda_torch() additionally asserts torch sees the GPU (used by the
torch-based stages; the MediaPipe landmark stage only needs require_pod()).
"""

import shutil
import subprocess
import sys


def require_pod() -> None:
    """Die unless an NVIDIA GPU is present (i.e. we are on the pod)."""
    if shutil.which("nvidia-smi") is None:
        sys.exit(
            "[pod_guard FATAL] no 'nvidia-smi' on PATH. This pipeline runs ONLY on "
            "the GPU pod (RTX 6000 Ada). Refusing to compute on the authoring box "
            "(no-local-compute contamination guard)."
        )
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30
        )
    except Exception as exc:  # noqa: BLE001 - want the guard to be loud, not clever
        sys.exit(f"[pod_guard FATAL] nvidia-smi failed to run: {exc!r}")
    if out.returncode != 0 or "GPU" not in (out.stdout or ""):
        sys.exit(
            "[pod_guard FATAL] nvidia-smi present but no GPU enumerated. "
            f"stdout={out.stdout!r} stderr={out.stderr!r}"
        )
    print(f"[pod_guard] GPU present: {out.stdout.strip().splitlines()[0]}")


def require_cuda_torch():
    """Import torch and die unless torch.cuda.is_available(). Returns torch."""
    require_pod()
    import torch  # deliberately imported here, never at authoring time

    if not torch.cuda.is_available():
        sys.exit(
            "[pod_guard FATAL] torch imported but torch.cuda.is_available() is False. "
            "Check the venv (/workspace/venvs/b1) was built by scripts/pod_setup_b1.sh "
            "with the cu121 wheel."
        )
    print(
        f"[pod_guard] torch {torch.__version__} on "
        f"{torch.cuda.get_device_name(0)} (cc "
        f"{'.'.join(map(str, torch.cuda.get_device_capability(0)))})"
    )
    return torch
