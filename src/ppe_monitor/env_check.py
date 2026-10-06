"""Record exactly what this machine has installed, so version mismatches are caught in
Phase 0 rather than discovered in Phase 4 (book, Chapter 17).

Required for Phase 0: Python 3.10-3.12, NumPy, OpenCV with its FFmpeg backend, the ffmpeg
command-line tool, and MediaMTX. Everything under "accelerators" is informational until
Phase 1, where it decides whether training and inference run on an NVIDIA GPU (CUDA),
an Apple Silicon GPU (MPS) or the CPU.
"""

from __future__ import annotations

import importlib
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime

from .device import running_under_rosetta
from .executables import find_ffmpeg, find_ffprobe, find_mediamtx, first_line_of, pick_encoder

OK, WARN, MISSING, INFO = "ok", "warn", "MISSING", "info"


@dataclass
class Check:
    group: str
    name: str
    value: str
    status: str
    note: str = ""
    required: bool = False


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:  # noqa: BLE001 - any import failure means "not usable"
        return None
    return getattr(module, "__version__", "installed")


def _system_checks() -> list[Check]:
    checks = []
    os_name = platform.system()
    if os_name == "Darwin":
        os_value = f"macOS {platform.mac_ver()[0]}"
    else:
        os_value = platform.platform(terse=True)
    checks.append(Check("system", "operating system", os_value, INFO))
    arch = platform.machine()
    note = "Apple Silicon" if os_name == "Darwin" and arch == "arm64" else ""
    checks.append(Check("system", "CPU architecture", arch, INFO, note))
    checks.append(Check("system", "CPU cores", str(os.cpu_count()), INFO))
    ram = _total_ram_gb()
    if ram:
        checks.append(Check("system", "memory", f"{ram:.0f} GB", INFO))

    py = sys.version_info
    py_ok = (3, 10) <= (py.major, py.minor) <= (3, 12)
    checks.append(Check("python", "Python", platform.python_version(), OK if py_ok else WARN,
                        "" if py_ok else "use 3.10-3.12 (3.11 recommended) for later-phase libraries",
                        required=True))
    # Judge by the Python actually running, not by shell variables: the setup script runs the
    # env's python directly, so CONDA_DEFAULT_ENV can still say "base".
    is_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    is_conda_env = os.path.isdir(os.path.join(sys.prefix, "conda-meta")) and \
        os.path.basename(os.path.dirname(sys.prefix)) == "envs"
    in_env = is_venv or is_conda_env
    env_name = ""
    if is_conda_env:
        env_name = f"conda: {os.path.basename(sys.prefix)}"
    elif is_venv:
        env_name = f"venv: {os.path.basename(sys.prefix)}"
    checks.append(Check("python", "isolated environment", env_name or "none", OK if in_env else WARN,
                        "" if in_env else "create a venv or conda env so project packages stay separate"))
    if running_under_rosetta():
        checks.append(Check("python", "Python architecture", "x86_64 under Rosetta", WARN,
                            "Intel Python can't use the Apple GPU; use an arm64 Python"))
    return checks


def _total_ram_gb() -> float | None:
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip()) / 1024**3
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def _library_checks() -> list[Check]:
    checks = []
    np_version = _module_version("numpy")
    checks.append(Check("libraries", "numpy", np_version or "not installed",
                        OK if np_version else MISSING, required=True))
    yaml_version = _module_version("yaml")
    checks.append(Check("libraries", "PyYAML", yaml_version or "not installed",
                        OK if yaml_version else MISSING, required=True))

    cv_version = _module_version("cv2")
    if not cv_version:
        checks.append(Check("libraries", "OpenCV", "not installed", MISSING, required=True))
        return checks
    import cv2

    info = cv2.getBuildInformation()
    has_ffmpeg = re.search(r"FFMPEG:\s+YES", info) is not None
    checks.append(Check("libraries", "OpenCV", cv_version, OK if has_ffmpeg else MISSING,
                        "" if has_ffmpeg else "this OpenCV build cannot read RTSP (no FFmpeg backend)",
                        required=True))
    gui = re.search(r"GUI:\s+(\S+)", info)
    gui_name = gui.group(1) if gui else "NONE"
    checks.append(Check("libraries", "OpenCV window support", gui_name, OK if gui_name != "NONE" else WARN,
                        "" if gui_name != "NONE" else "headless build: use view_stream.py --headless"))
    return checks


def _tool_checks() -> list[Check]:
    checks = []
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        line = first_line_of([ffmpeg, "-version"]) or "unknown"
        version = line.split(" Copyright")[0].replace("ffmpeg version ", "")
        checks.append(Check("tools", "ffmpeg", version, OK, ffmpeg, required=True))
        try:
            chosen = pick_encoder(ffmpeg)
        except RuntimeError:
            chosen = None
        hardware = chosen in ("h264_videotoolbox", "h264_nvenc")
        if chosen and chosen != "mpeg4":
            note = "hardware encoder, keeps the CPU free" if hardware else "software encoder"
            checks.append(Check("tools", "video encoder for fake cameras", chosen, OK, note))
        else:
            checks.append(Check("tools", "video encoder for fake cameras", chosen or "none", WARN,
                                "no working H.264 encoder; install ffmpeg from Homebrew"))
    else:
        checks.append(Check("tools", "ffmpeg", "not found", MISSING,
                            "macOS: brew install ffmpeg  |  conda: conda install -c conda-forge ffmpeg",
                            required=True))
    ffprobe = find_ffprobe()
    checks.append(Check("tools", "ffprobe", "found" if ffprobe else "not found", OK if ffprobe else WARN,
                        "" if ffprobe else "ships with ffmpeg; used later for inspecting clips"))

    mediamtx = find_mediamtx()
    if mediamtx:
        version = first_line_of([mediamtx, "--version"]) or "unknown"
        checks.append(Check("tools", "MediaMTX", version, OK, mediamtx, required=True))
    else:
        checks.append(Check("tools", "MediaMTX", "not found", MISSING,
                            "run: python scripts/get_mediamtx.py", required=True))
    docker = shutil.which("docker")
    checks.append(Check("tools", "docker", "found" if docker else "not found", INFO,
                        "optional now; used for packaging in Phase 6"))
    return checks


def _accelerator_checks() -> list[Check]:
    checks = []
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            out = subprocess.run([nvidia_smi], capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        driver = re.search(r"Driver Version:\s*([\d.]+)", out)
        cuda = re.search(r"CUDA Version:\s*([\d.]+)", out)
        gpu = first_line_of([nvidia_smi, "--query-gpu=name", "--format=csv,noheader"]) or "unknown"
        checks.append(Check("accelerators", "NVIDIA GPU", gpu, OK))
        checks.append(Check("accelerators", "NVIDIA driver", driver.group(1) if driver else "unknown", INFO,
                            f"supports CUDA up to {cuda.group(1)}" if cuda else ""))
    else:
        on_mac = platform.system() == "Darwin"
        checks.append(Check("accelerators", "NVIDIA GPU", "none", INFO,
                            "no CUDA/TensorRT on this machine" + ("; expected on a Mac" if on_mac else "")))

    torch_version = _module_version("torch")
    if torch_version:
        import torch

        cuda_ok = torch.cuda.is_available()
        mps_ok = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        checks.append(Check("accelerators", "PyTorch", torch_version, INFO,
                            f"CUDA {torch.version.cuda}" if torch.version.cuda else "CPU/MPS build"))
        checks.append(Check("accelerators", "PyTorch CUDA available", str(cuda_ok), INFO))
        checks.append(Check("accelerators", "PyTorch MPS (Apple GPU) available", str(mps_ok), INFO))
    else:
        checks.append(Check("accelerators", "PyTorch", "not installed", INFO, "for the GPU: pip install -r requirements-gpu.txt"))

    ort_version = _module_version("onnxruntime")
    if ort_version:
        import onnxruntime as ort

        checks.append(Check("accelerators", "ONNX Runtime", ort_version, INFO,
                            "providers: " + ", ".join(ort.get_available_providers())))
    else:
        checks.append(Check("accelerators", "ONNX Runtime", "not installed yet", INFO, "added in Phase 4"))

    trt_version = _module_version("tensorrt")
    if trt_version or platform.system() != "Darwin":
        checks.append(Check("accelerators", "TensorRT", trt_version or "not installed", INFO,
                            "" if trt_version else "NVIDIA-only; used in Phase 4 on NVIDIA machines"))
    return checks


def recommended_device(checks: list[Check]) -> str:
    values = {c.name: c.value for c in checks}
    if values.get("PyTorch CUDA available") == "True":
        return "cuda"
    if values.get("PyTorch MPS (Apple GPU) available") == "True":
        return "mps"
    if values.get("NVIDIA GPU", "none") != "none":
        return "cuda (after installing a CUDA build of PyTorch)"
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mps (Apple GPU, once PyTorch is installed in Phase 1)"
    return "cpu"


def collect() -> list[Check]:
    return _system_checks() + _library_checks() + _tool_checks() + _accelerator_checks()


def missing_required(checks: list[Check]) -> list[Check]:
    return [c for c in checks if c.required and c.status == MISSING]


def format_table(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks) + 2
    lines, group = [], None
    for c in checks:
        if c.group != group:
            group = c.group
            lines.append(f"\n[{group}]")
        mark = {OK: "ok ", WARN: "!! ", MISSING: "XX ", INFO: "   "}[c.status]
        note = f"   ({c.note})" if c.note else ""
        lines.append(f" {mark}{c.name.ljust(width)}{c.value}{note}")
    return "\n".join(lines).lstrip("\n")


def format_markdown(checks: list[Check]) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = "\n".join(
        f"| {c.group} | {c.name} | {c.value.replace('|', '/')} | {c.status} | {c.note.replace('|', '/')} |"
        for c in checks
    )
    return (
        "# Development environment\n\n"
        f"Generated by `python scripts/check_env.py --write` on {stamp}.\n"
        "Re-run it whenever you install or upgrade something, and commit the result, so the\n"
        "exact versions behind every benchmark number are on record.\n\n"
        f"**Recommended compute device for training/inference:** {recommended_device(checks)}\n\n"
        "| Group | Component | Version / value | Status | Note |\n"
        "|---|---|---|---|---|\n"
        f"{rows}\n"
    )
