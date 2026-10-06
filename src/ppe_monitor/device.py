"""Choosing where neural-network work runs, and checking that the choice actually works.

    cuda  an NVIDIA GPU
    mps   an Apple Silicon GPU, through Metal Performance Shaders (Apple's GPU compute layer)
    cpu   no GPU available to PyTorch

Phase 1 trains and runs the detector on `best_device()`. `gpu_report()` is what
scripts/check_gpu.py prints: is the GPU usable, does it give the same answers as the CPU,
and how much faster is it on work shaped like object detection.
"""

from __future__ import annotations

import platform
import statistics
import subprocess
import time
from dataclasses import dataclass, field


def running_under_rosetta() -> bool:
    """True when an Intel (x86_64) Python runs on an Apple Silicon Mac through Rosetta
    translation. Such a Python can never use the Apple GPU."""
    if platform.system() != "Darwin" or platform.machine() != "x86_64":
        return False
    try:
        out = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() == "1"
    except (OSError, subprocess.SubprocessError):
        return False


def best_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def why_no_gpu() -> str:
    """A plain-language reason PyTorch can't see a GPU here."""
    import torch

    if running_under_rosetta():
        return ("this Python is the Intel (x86_64) build running under Rosetta, which cannot use the "
                "Apple GPU. Use an arm64 Python (e.g. Miniforge/Miniconda for Apple Silicon, or Homebrew's).")
    if platform.system() == "Darwin":
        if not torch.backends.mps.is_built():
            return "this PyTorch was built without MPS support; reinstall it from requirements-gpu.txt."
        return "macOS reports no Metal GPU for PyTorch (MPS needs macOS 12.3 or newer on Apple Silicon)."
    return "no NVIDIA GPU with a CUDA build of PyTorch, and no Apple GPU (this is a CPU-only machine)."


def _sync(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _time_per_call(fn, device: str, warmup: int, iters: int) -> float:
    """Median seconds per call. GPU work is queued asynchronously, so each timing waits for
    the GPU to actually finish (synchronize) before stopping the clock."""
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


@dataclass
class GpuReport:
    torch_version: str
    device: str
    lines: list[str] = field(default_factory=list)
    gpu_ok: bool = False
    speedups: dict[str, float] = field(default_factory=dict)


def gpu_report(matmul_size: int = 2048, image_size: int = 640, batch: int = 4) -> GpuReport:
    import torch
    from torch import nn

    device = best_device()
    report = GpuReport(torch_version=torch.__version__, device=device)
    say = report.lines.append

    say(f"PyTorch          {torch.__version__}")
    say(f"Python           {platform.python_version()} ({platform.machine()})")
    if platform.system() == "Darwin":
        say(f"macOS            {platform.mac_ver()[0]}")
        say(f"MPS built        {torch.backends.mps.is_built()}")
        say(f"MPS available    {torch.backends.mps.is_available()}")
    say(f"CUDA available   {torch.cuda.is_available()}")
    say(f"best device      {device}")

    if device == "cpu":
        say(f"No GPU: {why_no_gpu()}")
        return report

    if device == "mps" and hasattr(torch.mps, "recommended_max_memory"):
        say(f"GPU memory       {torch.mps.recommended_max_memory() / 1024**3:.1f} GB usable by PyTorch "
            "(shared with the CPU on Apple Silicon)")

    # 1. Correctness: the GPU must give the same answer as the CPU (within float32 rounding).
    torch.manual_seed(0)
    a, b = torch.randn(512, 512), torch.randn(512, 512)
    expected = a @ b
    got = (a.to(device) @ b.to(device)).cpu()
    rel_error = float((got - expected).abs().max() / expected.abs().max())
    report.gpu_ok = rel_error < 1e-3
    say(f"correctness      max relative difference GPU vs CPU = {rel_error:.1e} "
        f"({'OK' if report.gpu_ok else 'TOO LARGE - GPU results are wrong'})")
    if not report.gpu_ok:
        return report

    # 2. Raw arithmetic: a large matrix multiply (2 * n^3 floating-point operations).
    n = matmul_size
    flops = 2 * n**3
    results = {}
    for dev, iters in (("cpu", 5), (device, 20)):
        x, y = torch.randn(n, n, device=dev), torch.randn(n, n, device=dev)
        results[dev] = _time_per_call(lambda: x @ y, dev, warmup=2, iters=iters)
    speed = results["cpu"] / results[device]
    report.speedups["matmul"] = speed
    say(f"matmul {n}x{n}   CPU {flops / results['cpu'] / 1e12:.2f} TFLOPS | "
        f"{device.upper()} {flops / results[device] / 1e12:.2f} TFLOPS | {speed:.1f}x faster")

    # 3. Detector-shaped work: the first layers of a YOLO-style backbone on 640x640 images.
    def backbone() -> nn.Module:
        layers, ch = [], 3
        for out_ch in (32, 64, 128, 256):
            layers += [nn.Conv2d(ch, out_ch, 3, stride=2, padding=1), nn.BatchNorm2d(out_ch), nn.SiLU()]
            ch = out_ch
        return nn.Sequential(*layers).eval()

    images_per_s = {}
    with torch.inference_mode():
        for dev, dtype, iters in (("cpu", torch.float32, 3), (device, torch.float32, 10), (device, torch.float16, 10)):
            model = backbone().to(dev, dtype)
            batch_t = torch.randn(batch, 3, image_size, image_size, device=dev, dtype=dtype)
            seconds = _time_per_call(lambda: model(batch_t), dev, warmup=2, iters=iters)
            images_per_s[(dev, dtype)] = batch / seconds
    cpu_ips = images_per_s[("cpu", torch.float32)]
    gpu32 = images_per_s[(device, torch.float32)]
    gpu16 = images_per_s[(device, torch.float16)]
    report.speedups["cnn_fp32"] = gpu32 / cpu_ips
    report.speedups["cnn_fp16"] = gpu16 / cpu_ips
    say(f"CNN {image_size}x{image_size}    CPU {cpu_ips:.0f} img/s | {device.upper()} {gpu32:.0f} img/s (FP32), "
        f"{gpu16:.0f} img/s (FP16) | {gpu32 / cpu_ips:.1f}x / {gpu16 / cpu_ips:.1f}x faster")
    return report
