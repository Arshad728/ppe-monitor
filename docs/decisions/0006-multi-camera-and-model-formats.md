# 0006 — Several cameras on one Mac: reader threads, one shared clock, Core ML and ONNX instead of TensorRT

**Status:** accepted, 25 Sep 2026 (the user chose 5–8 cameras per Mac; keypoints on 1 frame in 2 added) · **Book:** Chapters 11, 14, 16 and 21

## Context

Phase 4 turns the one-camera chain of Phases 2–3 into a service that watches many cameras. The
book (Chapter 21) asks for:

- one reader thread per camera with a latest-frame queue, reconnection with backoff, RTSP over
  TCP, and a fixed processing rate of 5–10 frames a second;
- frames from several cameras batched through the detector;
- the models exported to TensorRT and measured at FP32, FP16 and INT8: latency, frames per
  second, CPU use and mAP@50, with 1, 2, 4 and 8 cameras.

It is done when there is a throughput-against-accuracy table across precisions and camera counts,
and a camera can drop out and come back without crashing the service or stalling the others.

The development machine is a Mac (decision 0002). It has no NVIDIA GPU, so TensorRT is not
available.

## Decision

1. **One reader thread per camera, newest frame only** (`ingestion/reader.py`).
   - The "queue" has one slot. The reader overwrites it with every frame, and the analysis takes
     whatever is newest when it is ready.
   - Frames nobody took are dropped and counted (`skipped`). A slow machine therefore analyses
     fewer frames, never old ones.
   - A video file standing in for a camera is played at its own frame rate and loops.
2. **Reconnection.**
   - A camera counts as gone when its stream won't open, a read fails, or no frame arrives for
     `read_timeout` (5 s).
   - The reader then closes it and tries again after 0.5 s, 1 s, 2 s, 4 s, 8 s, then every 10 s.
   - The wait goes back to 0.5 s once frames flow again. RTSP always goes over TCP.
   - **Cap of 10 s, not 30 s.** An attempt costs one RTSP request. A safety camera that comes back
     after a long outage should be watched again within seconds, not half a minute.
3. **One clock for all cameras** (`service.py`).
   - `process_fps` times a second (default 10), the newest frame of every camera that has sent a
     new one goes into **one** detector call.
   - The frames with people in them then go into one keypoint call.
   - Each camera keeps its own tracker, rules and events.
   - **Why a shared clock:** the first version gave every camera its own timer. The timers drifted
     apart, and every detector call got a single frame, so nothing was ever batched. The test
     suite caught it (`tests/test_service.py`).
   - **When the machine can't keep up,** every camera slows down equally and is still analysed on
     its newest frame. The lag stays bounded, the frame rate drops, and the status table shows it.
4. **Model formats instead of TensorRT** (`vision/backends.py`, `scripts/export_models.py`):

   | The book (NVIDIA) | Here (Mac) |
   |---|---|
   | PyTorch FP32 | `pytorch`: PyTorch FP32 on the GPU (Metal) |
   | TensorRT FP16 | `pytorch-fp16`: PyTorch FP16 on the GPU; `coreml`: Core ML FP16 on the Neural Engine |
   | TensorRT INT8 | `coreml-int8`: Core ML with 8-bit weights and 16-bit maths; `onnx-int8`: ONNX Runtime INT8 on the CPU |
   | — | `onnx`: ONNX FP32 on the CPU. The portable file, which also runs on a Linux server. |

   - **INT8 calibration** uses the validation photos, never the test photos, which measure it.
   - **The keypoint model stays FP32 in `onnx-int8`.** Calibrating it needs keypoint-labelled
     photos, and the PPE dataset has none. Core ML's 8-bit weights need no calibration, so
     `coreml-int8` shrinks both models.
   - **Core ML's 8-bit option is weights only.** The weights shrink to a quarter and the maths
     stays FP16. Full 8-bit maths is left for later, if the benchmark shows the Neural Engine is
     the bottleneck.
   - **Batch sizes:** ONNX is exported with a variable batch (up to 16 frames a call). Core ML
     exports take one frame per call, so batching helps PyTorch and ONNX only.
5. **`backend: auto`** (`configs/streams.yaml`) picks, in order:
   - Core ML on a Mac, once exported;
   - otherwise PyTorch on a GPU;
   - otherwise ONNX on the CPU;
   - otherwise PyTorch on the CPU.

   **Confirmed by the Mac benchmark (25 Sep 2026).** Core ML is the fastest format on the Mac
   (62 frames a second in total, against 49 for PyTorch on the GPU), with unchanged accuracy
   (mAP@50 0.870 against 0.869). Core ML with 8-bit weights is as fast and as accurate, so either
   can be the default. FP16 stays, because it needs no quantization.
6. **What the benchmark measures** (`scripts/benchmark.py`).
   - **The whole chain, not only the model:** readers, decoding, trackers, rules and events. The
     "cameras per Mac" number is then what a deployment would get.
   - **Accuracy:** the Phase 1 score (mAP@50 on the test photos), plus the Phase 2 rule errors per
     frame, for every format.
   - **Load:** CPU and memory are measured. GPU and Neural Engine load are not, because reading
     them needs administrator rights (`sudo powermetrics`).
7. **What "doesn't stall the others" means** (`scripts/check_reconnect.py`). While one camera is
   gone, every other camera must:
   - keep at least 90 % of its frame rate from before;
   - never go longer than 1 s between two analysed frames (3 frame periods, when the machine
     manages under 3 frames a second per camera).

## Consequences

- **Adding a camera** is one entry in `configs/cameras.yaml`. The service picks it up at start.
- **A camera that is down** costs one connection attempt every 10 s. It is logged in
  `cameras.jsonl` each time it drops out or comes back, and shown in the status table.
- **Accuracy for speed is measured, not assumed.** The format `auto` picks should stay within
  about 0.01 mAP@50 of PyTorch FP32 in the benchmark. On the Mac every format did, within 0.004.
- **One Mac watched 4 cameras at 10 frames a second, and 8 at about 7.7.** The models are the
  limit: Core ML takes one frame per call, at 15 ms per frame for both models.
- **Added 25 Sep (the user plans 5–8 cameras per Mac): the keypoint model runs on 1 frame in 2**
  of each camera (`configs/ppe.yaml`, `pose: every: 2`). Between those frames, each person's last
  keypoints are moved onto their new box.
  - One Mac now watches 8 cameras at 10.0 frames a second with Core ML (80 frames a second in
    total, up from 62).
  - The Phase 2 and 3 test clips give identical events either way. Bending and crouching tests
    pass with keypoints on 1 frame in 1, 2 and 3.
  - `every: 1` puts it back, for example on a machine with few cameras and time to spare.
  - Details: [phase4_results.md](../phase4_results.md).
- **The TensorRT path is one more entry.** On a Linux server with an NVIDIA GPU, it is one more
  entry in `SPECS` (format `engine`, FP16 or INT8). Nothing else changes.
- **Why the 1-second cap.**
  - It is half the shortest dwell time in the rules (2 s).
  - It is also half the time the tracker keeps a hidden person's ID (`lost_seconds: 2`).
  - So a gap that short can delay an event, but can't lose a person's track or split one
    incident into two.
