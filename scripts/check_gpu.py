#!/usr/bin/env python3
"""Check that PyTorch can use this machine's GPU, and measure how much faster it is than the CPU.

    python scripts/check_gpu.py

On an Apple Silicon Mac the GPU is used through Metal Performance Shaders ("mps"). The
check confirms the GPU gives the same answers as the CPU, then times a large matrix
multiply and the first layers of a YOLO-style network on 640x640 images. The report is also
saved to logs/gpu_report.txt. Exits with code 1 if no working GPU is found.
"""

import sys
import time

import _bootstrap  # noqa: F401

from ppe_monitor.config import PROJECT_ROOT


def main() -> int:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("PyTorch isn't installed. Run: pip install -r requirements-gpu.txt")
        return 1
    from ppe_monitor.device import gpu_report

    print("Checking the GPU (takes about 30 seconds) ...\n")
    report = gpu_report()
    verdict = (f"PASS - PyTorch runs on the {report.device.upper()} GPU and its results match the CPU."
               if report.gpu_ok else "FAIL - no working GPU for PyTorch on this machine.")
    lines = [f"GPU check - {time.strftime('%Y-%m-%d %H:%M:%S')}", *report.lines, "", verdict]
    print("\n".join(lines[1:]))

    out = PROJECT_ROOT / "logs" / "gpu_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n(report saved to {out.relative_to(PROJECT_ROOT)})")
    return 0 if report.gpu_ok else 1


if __name__ == "__main__":
    sys.exit(main())
