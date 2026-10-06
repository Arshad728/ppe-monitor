"""The same two models (detector and keypoints), run in different ways (book, Chapters 12 and 21).

    backend        what runs                                         numbers        where
    pytorch        the .pt files, as in Phases 1-3                   FP32           Mac GPU (Metal), or CPU
    pytorch-fp16   the .pt files at half precision                   FP16           Mac GPU (Metal)
    coreml         Apple Core ML export                              FP16           Neural Engine (macOS only)
    coreml-int8    Core ML with 8-bit weights                        INT8 weights,  Neural Engine (macOS only)
                                                                     FP16 maths
    onnx           ONNX Runtime export                               FP32           CPU: works on any machine
    onnx-int8      ONNX Runtime, detector quantized to 8 bits,       INT8           CPU
                   calibrated on the validation split's images

The book's NVIDIA route (TensorRT) doesn't exist on a Mac (decision 0002); Core ML is Apple's
equivalent: it compiles the model for the Mac's Neural Engine (Ultralytics picks CPU + Neural
Engine, avoiding a GPU compiler bug in coremltools 9). ONNX on the CPU is the
fallback for a machine with no usable GPU.

INT8 for the keypoint model: Core ML's 8-bit weights need no calibration, so coreml-int8 has both
models in 8 bits. The ONNX INT8 recipe needs calibration images of the model's own task, which
this project has only for the detector, so onnx-int8 keeps the keypoint model in FP32.

Exports live in models/exported/<model>/<backend>/ and are made by scripts/export_models.py. A
backend whose files are missing can't be loaded; the error says which command makes them.
"""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import PROJECT_ROOT
from .detector import MODELS_DIR, PRETRAINED_DIR, load_model
from .matching import Det
from .pose import Pose, poses_from_result

EXPORT_DIR = MODELS_DIR / "exported"
CALIBRATION_DATA = PROJECT_ROOT / "datasets" / "ppe3" / "data.yaml"   # INT8 calibration uses its val split


@dataclass(frozen=True)
class Spec:
    name: str
    precision: str
    runs_on: str
    fmt: str                          # "pt", "coreml" or "onnx"
    export: dict = field(default_factory=dict)        # YOLO.export() arguments for the detector
    pose_export: dict | None = None                   # ... for the keypoint model (None: same as the detector)
    half: bool = False
    macos_only: bool = False


SPECS = {
    "pytorch": Spec("pytorch", "FP32", "GPU (Metal) or CPU", "pt"),
    "pytorch-fp16": Spec("pytorch-fp16", "FP16", "GPU (Metal)", "pt", half=True),
    "coreml": Spec("coreml", "FP16", "Neural Engine", "coreml", {"format": "coreml", "quantize": 16},
                   macos_only=True),
    "coreml-int8": Spec("coreml-int8", "INT8 weights, FP16 maths", "Neural Engine", "coreml",
                        {"format": "coreml", "quantize": "w8a16"}, macos_only=True),
    "onnx": Spec("onnx", "FP32", "CPU", "onnx", {"format": "onnx", "dynamic": True, "batch": 8}),
    "onnx-int8": Spec("onnx-int8", "INT8 (calibrated)", "CPU", "onnx",
                      {"format": "onnx", "dynamic": True, "batch": 8, "quantize": 8},
                      pose_export={"format": "onnx", "dynamic": True, "batch": 8}),
}
SUFFIX = {"coreml": ".mlpackage", "onnx": ".onnx"}


def is_macos() -> bool:
    return platform.system() == "Darwin"


def exported_path(weights: Path, backend: str) -> Path:
    """Where the export of `weights` for `backend` lives (the .pt itself for PyTorch backends)."""
    spec = SPECS[backend]
    if spec.fmt == "pt":
        return Path(weights)
    return EXPORT_DIR / Path(weights).stem / backend / (Path(weights).stem + SUFFIX[spec.fmt])


def export(weights: Path, backend: str, *, pose: bool = False, imgsz: int = 640) -> Path:
    """Export one model for one backend (slow: seconds to a minute). Returns the exported file."""
    from ultralytics import YOLO

    spec = SPECS[backend]
    if spec.fmt == "pt":
        return Path(weights)
    args = dict(spec.pose_export if (pose and spec.pose_export) else spec.export)
    if args.get("quantize") == 8:
        args["data"] = str(CALIBRATION_DATA)
    target = exported_path(weights, backend)
    work = target.parent
    work.mkdir(parents=True, exist_ok=True)
    local = work / Path(weights).name            # export next to a copy, so the output lands in `work`
    shutil.copy2(weights, local)
    try:
        produced = Path(YOLO(str(local)).export(imgsz=imgsz, **args))
    finally:
        local.unlink(missing_ok=True)
    if produced != target:                       # e.g. INT8 ONNX is written as <name>_int8.onnx
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        produced.rename(target)
    return target


def _max_batch(path: Path, fmt: str) -> int:
    if fmt == "pt":
        return 16
    if fmt == "onnx":
        import onnx

        dim = onnx.load(str(path), load_external_data=False).graph.input[0].type.tensor_type.shape.dim[0]
        return 16 if dim.dim_param else max(1, dim.dim_value)
    return 1                                      # Core ML exports take one image at a time


@dataclass
class Backend:
    """A loaded detector + keypoint model pair, and how to call them on a list of frames."""
    spec: Spec
    detector: object
    pose: object | None
    device: str
    det_batch: int
    pose_batch: int
    files: tuple[Path, Path | None]

    @property
    def name(self) -> str:
        return self.spec.name

    def _predict(self, model, frames: list, batch: int, **kw) -> list:
        kw = {"verbose": False, "device": self.device, **kw}
        if self.spec.half:
            kw["quantize"] = 16
        out = []
        for start in range(0, len(frames), batch):
            out += model.predict(frames[start:start + batch] if batch > 1 else frames[start], **kw)
        return out

    def detect(self, frames: list, *, conf: float, imgsz: int = 640) -> list[list[Det]]:
        results = self._predict(self.detector, frames, self.det_batch, conf=conf, imgsz=imgsz)
        return [[Det(int(c), tuple(float(v) for v in xy), float(s))
                 for c, xy, s in zip(r.boxes.cls.tolist(), r.boxes.xyxyn.tolist(), r.boxes.conf.tolist())]
                for r in results]

    def warmup(self, imgsz: int = 640) -> None:
        """One call of each model on a blank frame. The first call loads the model onto the device
        (and, for Core ML, compiles it for the Neural Engine), which can take seconds."""
        blank = np.full((360, 640, 3), 114, np.uint8)
        self.detect([blank], conf=0.25, imgsz=imgsz)
        self.keypoints([blank], conf=0.25, keypoint_conf=0.5, imgsz=imgsz)

    def keypoints(self, frames: list, *, conf: float, keypoint_conf: float, imgsz: int = 640) -> list[list[Pose]]:
        if self.pose is None or not frames:
            return [[] for _ in frames]
        results = self._predict(self.pose, frames, self.pose_batch, conf=conf, imgsz=imgsz)
        return [poses_from_result(r, f.shape[1] / f.shape[0], keypoint_conf) for r, f in zip(results, frames)]


def available(weights: Path, pose_weights: Path | None = None) -> dict[str, str]:
    """{backend: "" if it can be loaded here, else why not}."""
    out = {}
    for name, spec in SPECS.items():
        if spec.macos_only and not is_macos():
            out[name] = "Core ML runs only on macOS"
            continue
        if spec.half and _device() == "cpu":
            out[name] = "FP16 needs a GPU"
            continue
        missing = [p for p in (exported_path(weights, name),
                               exported_path(pose_weights, name) if pose_weights else None) if p and not p.exists()]
        out[name] = f"not exported yet: python scripts/export_models.py --backend {name}" if missing else ""
    return out


def _device() -> str:
    from ..device import best_device

    return best_device()


def load(backend: str, weights: Path, pose_weights: Path | None = None, *, device: str | None = None) -> Backend:
    spec = SPECS[backend]
    why = available(weights, pose_weights).get(backend, "")
    if why:
        raise RuntimeError(f"backend {backend}: {why}")
    det_file = exported_path(weights, backend)
    pose_file = exported_path(pose_weights, backend) if pose_weights else None
    if spec.fmt == "pt":
        device = device or _device()
    else:
        device = "cpu"         # Core ML and ONNX Runtime choose their own hardware; Ultralytics wants "cpu" here
    det = load_model(det_file)
    pose = load_model(pose_file) if pose_file else None
    b = Backend(spec, det, pose, device, _max_batch(det_file, spec.fmt),
                _max_batch(pose_file, spec.fmt) if pose_file else 1, (det_file, pose_file))
    # so that code taking a plain model (evaluation, rule checks) runs it the same way
    for model, batch in ((det, b.det_batch), (pose, b.pose_batch)):
        if model is not None:
            model.ppe_batch = batch
            model.ppe_predict_args = {"device": device, **({"quantize": 16} if spec.half else {})}
    return b


def default_pose_weights(name: str = "yolo26n-pose.pt") -> Path:
    return PRETRAINED_DIR / name
