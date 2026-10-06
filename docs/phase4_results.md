# Phase 4: many cameras at once, and faster model formats

**Book:** Chapter 21, using Chapters 11, 14 and 16 · **Model:** `ppe3_yolo26n_baseline` (Phase 1) and
`yolo26n-pose`; the second Mac run uses the fine-tuned `ppe4_yolo26n_finetune` · **Settings:** `configs/streams.yaml` · **Why it is built this way:**
[decision 0006](decisions/0006-multi-camera-and-model-formats.md)

**Reproduce everything below:** `bash scripts/mac_phase4.sh export` (once, ~5 min), then
`bash scripts/mac_phase4.sh check` (~15 min).

## What was built

```
camera 1 ─ reader thread ─ newest frame ─┐
camera 2 ─ reader thread ─ newest frame ─┼─ every 0.1 s: one batch ─► detector ─► keypoints (frames with people)
   ...                                   │                                 │
camera N ─ reader thread ─ newest frame ─┘                                 ▼
                                          per camera: tracker ─► PPE rule ─► zone rules ─► events ─► events.jsonl + snapshots
```

| Part | File | What it does |
|---|---|---|
| **Reader** | `ingestion/reader.py` | One thread per camera. It keeps only the newest frame. After a drop-out it retries after 0.5, 1, 2, 4 and 8 s, then every 10 s. RTSP always uses TCP. A camera counts as gone when its stream won't open, a read fails, or nothing arrives for 5 s. A video file plays in real time and loops, like a camera. |
| **Multi-camera engine** | `service.py` | One clock for all cameras (10 ticks a second). At each tick, the newest frame of every camera goes to the detector in one call; the frames with people go to the keypoint model in one call. Then each camera's own chain (Phase 3) runs. Frames nobody took are counted as skipped. |
| **Model formats** | `vision/backends.py`, `scripts/export_models.py` | PyTorch FP32 and FP16, Core ML FP16 and 8-bit weights, ONNX FP32 and INT8 (calibrated on the validation photos). They share one interface, so the engine, the evaluation and the rule check all run on any of them. |
| **Service** | `scripts/run_cameras.py` | Watches the cameras in `configs/cameras.yaml` (or video files). Writes events, snapshots and camera drop-outs, and prints a status table every 10 s. `--show` opens one window with every camera. |
| **Benchmark** | `scripts/benchmark.py` | For each format: accuracy on the test photos, time per frame, then the whole chain with 1, 2, 4 and 8 cameras: frames per second, lag, CPU and memory. Each format runs in a fresh process. |
| **Drop-out check** | `scripts/check_reconnect.py` | Four fake RTSP cameras. One is unplugged for 15 s and plugged back in. The others must not stall, and it must come back by itself. |

## 8 cameras at 10 frames a second: keypoints on 1 frame in 2 (25 Sep, evening)

You chose 5–8 cameras per Mac, so the keypoint model now runs on 1 frame in 2 of each camera
(`configs/ppe.yaml`, `pose: every: 2`).
- **Between those frames,** each person's last keypoints are moved onto their new box
  (`vision/pose.py`, `carried`). Posture hardly changes in 0.1 s, and the box follows a walking
  person.
- **A camera with nobody in view** runs the keypoint model as soon as someone appears.
- `every: 1` puts it back on every frame.

**On the Mac** (`mac_phase4.sh check`, 23:03). The model in use is now the fine-tuned
`ppe4_yolo26n_finetune`, with the same size and layers as the Phase 1 model, so the speeds compare.

| Format | 8 cameras: before | now | Frames/s in total (best): before | now | Lag at 8 cameras (95 %): before | now |
|---|---|---|---|---|---|---|
| pytorch | 6.1 | 8.0 | 49 | 64 | 225 ms | 205 ms |
| pytorch-fp16 | 7.1 | **9.2** | 57 | 74 | 205 ms | 185 ms |
| **coreml** (in use) | 7.7 | **10.0** | 62 | 80 | 189 ms | 163 ms |
| coreml-int8 | 7.8 | **9.9** | 63 | 79 | 188 ms | 186 ms |
| onnx | 2.0 | 2.5 | 16 | 20 | 564 ms | 526 ms |
| onnx-int8 | 2.2 | 2.9 | 17 | 24 | 588 ms | 425 ms |

- **Core ML now keeps all 8 cameras at the full 10 frames a second.** 80 frames a second in total
  is exactly what 8 cameras ask for, so the Mac's real limit is higher; it wasn't measured past 8.
- **1, 2 and 4 cameras stay at 10** with every format except ONNX. **Bold** = every camera got at
  least 90 % of the target.
- **The rest of the check passed:** 188 tests. With a camera unplugged for 15 s, the others stayed
  at 10.0 frames a second (longest gap 0.12 s), and it was analysed again 3.7 s after it came back.
- **One number to watch: ONNX INT8 of the fine-tuned model** loses 0.024 mAP@50 (0.859 against
  0.883), where the Phase 1 model lost 0.004, and it judges a bare head "worn" twice as often
  (4.7 % against 2.4 %). Core ML, the format in use, loses 0.001. ONNX INT8 isn't used on the Mac.
  Before it is used on a Linux server (Phase 6), calibrate it on more photos: the export warns that
  the 237 validation photos are fewer than the 300 it recommends.

**Does reusing keypoints cost accuracy? No.** Checked in the cloud on the Phase 2 and 3 test clips
with the same detections (Phase 1 model): keypoints on every frame, against 1 frame in 2.

| Clips | No helmet: caught once | No vest: caught once | False events, helmet / vest | Every frame vs 1 in 2 |
|---|---|---|---|---|
| still | 25/28 | 35/45 | 4 / 6 | identical |
| moving, 15 fps | 42/47 | 55/68 | 5 / 6 | identical |
| moving, 5 fps | 38/47 | 48/68 | 5 / 3 | identical |
| zone clips | 40/47 | 56/68 | 5 / 7 | identical; zone 28/37 caught, 0 false |

- **The lists of mistakes are identical, clip by clip.**
- **Frame by frame,** 33 of 14,730 verdicts differ (0.2 %), and the 1-second vote absorbs every
  one of them.
- **Reused heads land close:** 95 % of them sit within 0.021 frame heights of where the keypoint
  model would have put them.
- **Tests** (`tests/test_motion.py`) cover bending, crouching and brisk walking with keypoints on
  1 frame in 1, 2 and 3. `tests/test_service.py` checks that each camera sends half its frames
  to the keypoint model.

## Results on the Mac (MacBook Air M4, 25 Sep 2026)

- **Tests:** 173 pass.
- **Exports:** all 8 succeeded (4 formats × detector and keypoint model). Core ML FP16 is 4.8 MB
  for the detector and 8-bit weights 2.6 MB, against 5.1 MB for the PyTorch file.

### Speed against accuracy

This is the book's table: frames analysed per second per camera, target 10, next to mAP@50 on
the test photos. **Bold** means every camera got at least 90 % of the target. Each cell is the
whole chain (readers, models, trackers, rules, events), measured for 20 s after a 6 s warm-up.

| Format | Precision | Runs on | mAP@50 | 1 camera | 2 cameras | 4 cameras | 8 cameras | Frames/s in total (best) |
|---|---|---|---|---|---|---|---|---|
| pytorch | FP32 | GPU (Metal) | 0.869 | **10.0** | **10.0** | **10.0** | 6.1 | 49 |
| pytorch-fp16 | FP16 | GPU (Metal) | 0.869 (+0.000) | **10.0** | **10.0** | **10.0** | 7.1 | 57 |
| **coreml** | FP16 | Neural Engine | 0.870 (+0.000) | **10.0** | **10.0** | **10.0** | 7.7 | 62 |
| coreml-int8 | 8-bit weights, FP16 maths | Neural Engine | 0.870 (+0.001) | **10.0** | **10.0** | **10.0** | 7.8 | 63 |
| onnx | FP32 | CPU | 0.869 (+0.000) | **10.0** | 4.3 | 3.1 | 2.0 | 16 |
| onnx-int8 | INT8 (calibrated) | CPU | 0.866 (−0.004) | **10.0** | 5.2 | 3.8 | 2.2 | 17 |

**Accuracy in detail,** with the time for one frame through each model:

| Format | person AP@50 | helmet | vest | Helmet: wearer judged missing | Helmet: bare judged worn | Vest: wearer judged missing | Vest: bare judged worn | Detector ms | Keypoints ms |
|---|---|---|---|---|---|---|---|---|---|
| pytorch | 0.838 | 0.921 | 0.849 | 4.5 % | 2.3 % | 9.5 % | 8.6 % | 12.5 | 14.1 |
| pytorch-fp16 | 0.838 | 0.921 | 0.849 | 4.5 % | 2.3 % | 9.1 % | 8.5 % | 12.7 | 13.5 |
| coreml | 0.840 | 0.922 | 0.846 | 4.5 % | 2.3 % | 9.1 % | 8.6 % | 7.4 | 7.6 |
| coreml-int8 | 0.837 | 0.920 | 0.853 | 3.8 % | 2.4 % | 9.5 % | 8.6 % | 7.4 | 7.7 |
| onnx | 0.838 | 0.921 | 0.849 | 4.5 % | 2.3 % | 9.5 % | 8.6 % | 14.6 | 16.7 |
| onnx-int8 | 0.837 | 0.917 | 0.843 | 4.2 % | 2.6 % | 11.6 % | 8.1 % | 16.9 | 17.7 |

**Lag** (95th percentile: how old the newest frame was when its analysis finished):

| Format | 1 camera | 2 cameras | 4 cameras | 8 cameras |
|---|---|---|---|---|
| pytorch | 150 ms | 162 ms | 161 ms | 225 ms |
| pytorch-fp16 | 128 ms | 161 ms | 144 ms | 205 ms |
| coreml | 104 ms | 113 ms | 134 ms | 189 ms |
| coreml-int8 | 73 ms | 84 ms | 108 ms | 188 ms |
| onnx | 103 ms | 302 ms | 383 ms | 564 ms |
| onnx-int8 | 104 ms | 279 ms | 329 ms | 588 ms |

What this shows:

1. **No format loses accuracy that matters.**
   - Every format is within 0.004 mAP@50 of PyTorch FP32.
   - The rule errors move by at most 0.7 points. The one exception is ONNX INT8's vest rule:
     wearers judged missing go from 9.5 % to 11.6 %.
2. **Core ML is the default, and stays so.** `backend: auto` picks it on the Mac, and the
   numbers back that:
   - It is the fastest per frame: the detector takes 7.4 ms, against 12.5 ms for PyTorch on the GPU.
   - It gets the most frames through in total: 62 a second, against 49.
   - Its lag is lower than PyTorch's at every camera count.
   - Its accuracy is unchanged.
3. **One Mac watches 4 cameras at the full 10 frames a second.** With 8 cameras, each one gets
   7.7 frames a second with Core ML. That is above the book's 5 minimum but under the target
   of 10.
   - **The models are the limit.** Core ML takes one frame per call, and each frame costs
     7.4 ms in the detector plus 7.6 ms in the keypoint model. 8 cameras × 10 frames × 15 ms
     is 1.2 s of model time every second.
   - **The keypoint model is half of that.** Running it on every second frame of each camera
     should bring 8 cameras to about 10 frames a second: a person's posture hardly changes in
     0.1 s. **Done the same evening:** 8 cameras now get 10.0 with Core ML, at no cost in
     accuracy ([above](#8-cameras-at-10-frames-a-second-keypoints-on-1-frame-in-2-25-sep-evening)).
4. **INT8 doesn't pay off on the Mac.**
   - Core ML with 8-bit weights runs at the same speed as FP16. The Neural Engine does the maths
     in FP16 either way, and the smaller weights make no difference for a model this size
     (2.4 million parameters). It is as good a choice as FP16, with files half the size.
   - ONNX INT8 is slower per frame than ONNX FP32 (16.9 against 14.6 ms), and it costs the most
     accuracy.
5. **ONNX on the Mac's CPU is only good for one camera.** With 2 or more cameras it manages
   9–16 frames a second in total. That is far under the ~30 its 31 ms per frame would suggest,
   so batches of several frames seem to be slow in ONNX Runtime on this CPU. It isn't worth
   chasing, because Core ML is 4 times faster. ONNX is the file that runs on a Linux server,
   where it will be measured again (Phase 6).
6. **The Mac has room to spare.** With 8 cameras, the Core ML chain uses 15 % of the 10 cores
   and PyTorch 5 %, and both stay under 1.2 GB of memory.
   - The ONNX figures (1.3–3.1 GB) are inflated. ONNX Runtime keeps the memory it took for the
     accuracy run's batches of 16 photos.
   - The benchmark now measures speed before accuracy, so the next run shows the live chain's
     own memory.

### A camera drops out

Four fake RTSP cameras at 10 frames a second each, with Core ML. `rc2` was unplugged at 10.0 s
and plugged back in at 25.0 s.

| Check | Result | Detail |
|---|---|---|
| nothing crashed | PASS | |
| the other cameras never stalled | PASS | 10.1 frames/s each before, 10.0 while rc2 was gone; longest gap between two analysed frames 0.13 s |
| rc2 came back by itself | PASS | analysed again 3.7 s after it came back |

### Watching the three simulated cameras (`mac_phase4.sh run`)

The three Phase 0 cameras ran through the fake RTSP network with Core ML.

- All three were live within 4 s of starting.
- The status table stayed the same for the first 5 minutes: every camera at 10.0 frames a second,
  lag 85 ms (median) and 114 ms (95th percentile), batches of 3 frames taking 56 ms, and no
  reconnections.
- In the first 30 s, the zone rules fired on cam1 (keep-out, always on) and cam3 (pit, on
  07:00–19:00 Monday to Saturday). Nothing fired on cam2, whose rule is only on 19:00–07:00.
  That is right for 11:16 on a Friday.
- Each loop of a clip brings its worker back as a new person, so the zone event repeats every
  20 s.
- The PPE events on these cartoon clips mean nothing, as in Phase 3. The detector finds the
  cartoon people but not their cartoon helmets.

**Found on the Mac:** the events and the status table reached the Terminal late, in bursts.
Python holds back its output when it goes into a pipe (the script copies everything to the log
file through one). `run_cameras.py` and `check_reconnect.py` now send out every line as it
happens.

## Reference: the cloud check

**This machine has 2 CPU cores and no GPU.** It shows that everything works and how the formats
compare in accuracy. Its speed says nothing about the Mac.

### Speed against accuracy

Target: 10 frames per second per camera. The cells give frames per second per camera; none of
the formats reaches the target on this machine.

| Format | Precision | Runs on | mAP@50 | 1 camera | 2 cameras | 4 cameras |
|---|---|---|---|---|---|---|
| pytorch | FP32 | CPU | 0.869 | 4.7 | 2.0 | 1.0 |
| onnx | FP32 | CPU | 0.869 (+0.000) | 6.1 | 3.2 | 1.7 |
| onnx-int8 | INT8 (calibrated) | CPU | 0.861 (−0.008) | 4.0 | 2.2 | 1.0 |

| Format | person AP@50 | helmet | vest | Helmet: wearer judged missing | Helmet: bare judged worn | Vest: wearer judged missing | Vest: bare judged worn | Detector ms/frame | Keypoints ms/frame |
|---|---|---|---|---|---|---|---|---|---|
| pytorch | 0.838 | 0.921 | 0.849 | 4.5 % | 2.3 % | 9.5 % | 8.6 % | 105 | 130 |
| onnx | 0.838 | 0.921 | 0.849 | 4.5 % | 2.3 % | 9.5 % | 8.6 % | 50 | 60 |
| onnx-int8 | 0.825 | 0.916 | 0.843 | 4.2 % | 2.4 % | 11.1 % | 6.7 % | 97 | 64 |

- **ONNX FP32 gives exactly PyTorch's answers.** Every AP and every rule error is the same. The
  detector takes half the time, and the whole chain analyses 30–70 % more frames.
- **INT8 costs a little accuracy** (−0.008 mAP@50, mostly on people). It also moves the vest rule's
  errors around: more wearers judged missing (9.5 → 11.1 %), fewer bare people judged worn
  (8.6 → 6.7 %).
- **INT8 is slower on this CPU**, not faster, even though the CPU has 8-bit instructions
  (AVX-512 VNNI).
  - Only the convolutions become 8-bit. The activations (88 sigmoids and 95 multiplications)
    stay in floating point.
  - So the INT8 file converts back and forth around every layer: it adds 616 conversion steps to
    the model's 471.
  - For a model this small, those conversions cost more than the 8-bit convolutions save.
- **Batching does not help on a CPU.** With 2 cameras, each gets about half the frames one camera
  got, because one camera already fills the CPU. It is there for the GPU, where one call of N frames costs far less than
  N calls.

### A camera drops out

Four fake RTSP cameras at 5 frames a second each (ONNX). `rc2` was unplugged at 10.5 s and
plugged back in at 22.0 s.

| Check | Result | Detail |
|---|---|---|
| nothing crashed | PASS | |
| the other cameras never stalled | PASS | 1.3 frames/s each before, 2.0 while rc2 was gone (they got its share); longest gap 0.85 s |
| rc2 came back by itself | PASS | analysed again 7.3 s after it came back |

The 7.3 s is mostly the retry schedule. By then the waits between attempts had grown to 8 s.
With the cap at 10 s, a camera is picked up within about 10 s of coming back, plus the second or
two it takes to open the stream.

## Found while building

- **Timers per camera never batch.** The first version gave every camera its own clock, and the
  clocks drifted apart. The test with a counting stand-in model showed every detector call got
  exactly one frame. Now all cameras share one clock.
- **Formats that take one frame at a time.** Core ML exports, and ONNX exported without a
  variable batch, take one frame per call; a list of 16 frames fails. Each format now carries
  its own batch size (ONNX is exported with a variable batch, up to 16).
- **Two exports of the same model overwrote each other.** Core ML FP16 and 8-bit were written to the
  same file name. Every format now has its own folder, `models/exported/<model>/<format>/`.
- **Memory figures were inflated.**
  - Measuring every format in one process mixed their memory, so each format now runs in a
    fresh process.
  - The Mac run still showed ONNX at 1.3–3.1 GB. ONNX Runtime keeps the memory it took for the
    accuracy run's batches of 16 photos.
  - The benchmark now measures speed before accuracy. In the cloud, ONNX with 1 camera then uses
    0.56 GB.
- **The first frames were slow.** A model's first call loads it onto the device (and, for Core ML,
  compiles it for the Neural Engine). The engine now makes that call before the cameras start.
- **A dead camera was noisy.** FFmpeg printed an error at every attempt. The reader's own status
  already says so, so FFmpeg's and OpenCV's messages are turned down.
