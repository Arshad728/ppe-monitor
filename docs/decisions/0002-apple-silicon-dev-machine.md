# 0002 — Build and run on the Mac, using its Apple Silicon GPU

**Status:** accepted (Phase 0 review, 2026-09-20) · **Book:** Chapters 12, 17, 21 and Appendix C

## Context

The book's reference setup uses an NVIDIA GPU with CUDA and TensorRT. This project is built on
a MacBook Air with Apple Silicon. It has a capable built-in GPU, but CUDA and TensorRT exist
only for NVIDIA hardware. At the Phase 0 review the choice was: **run on the Mac's GPU.**

## Decision

| Work | Runs on |
|---|---|
| Fake cameras (video encoding) | The Mac's hardware video encoder (`h264_videotoolbox`), chosen automatically; falls back to software `libx264` if it fails a test encode |
| Neural networks: training and detection | The Mac's GPU through PyTorch **MPS** (Metal Performance Shaders); `ppe_monitor.device.best_device()` returns `"mps"` |
| Optimised detection (Phase 4) | Core ML / ONNX Runtime on the Mac's GPU and Neural Engine, compared against plain PyTorch MPS |
| TensorRT (Phase 4) | Not used. It is NVIDIA-only. The book's TensorRT chapter stays useful background. |

`scripts/check_gpu.py` confirms that MPS works and gives the same answers as the CPU, and
measures the speed-up. `scripts/mac_setup.sh` refuses to use an Intel (x86_64) Python: under
Rosetta, PyTorch can't see the Apple GPU.

Versions: Python 3.11; PyTorch 2.14.0 on macOS 14+ (2.11.0 is the newest for macOS 11–13);
NumPy 2.3.5; OpenCV 4.14.0.94; MediaMTX v1.21.0. `docs/ENVIRONMENT.md` records the exact state.

## Consequences

- **Training speed.** MPS is several times faster than the Mac's CPU, but slower than a desktop
  NVIDIA GPU. Phase 1 starts with the smallest YOLO model at 640 px and a moderate number of
  epochs, and measures time per epoch before committing to longer runs. A free Colab or Kaggle
  GPU is the fallback if a full run takes too long, and nothing in the code depends on
  where training happens.
- **Heat.** A MacBook Air has no fan and slows down under long, heavy load. Training and
  benchmarks should run with the Mac plugged in, and results should report the *sustained*
  rate after several minutes, not the first minute.
- **Some operations aren't implemented on MPS yet.** For those, PyTorch can fall back to the CPU
  (`PYTORCH_ENABLE_MPS_FALLBACK=1`). Phase 1 will note any operation that needs this.
- **Throughput target.** The book's "8 streams at 10+ FPS on one mid-range GPU" assumes an
  NVIDIA card. Phase 4 measures what this Mac sustains and records that as the project's number.
