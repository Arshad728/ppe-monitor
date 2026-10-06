# Using the PPE monitor

The system watches CCTV cameras for three things:
- a worker **without a helmet**;
- a worker **without a hi-vis vest** (switched off at your site);
- a person inside a **restricted zone**.

When one lasts a few seconds, it sends an alert with a picture to your phone. It also keeps every
event on a dashboard, where you can mark each one "Confirm violation" or "False alarm".

Every command below is run in the Terminal on your Mac, from any folder. The command
`bash ~/Documents/ppe-monitor/scripts/ppe.sh help` lists them all.

## 1. What you have

| | |
|---|---|
| **The model in use** | `models/ppe4caps_yolo26n_caps2.pt`, YOLO26n. It is fine-tuned on public construction PPE photos and on photos of people in caps and hats. It runs as Core ML on the Mac's GPU. `bash scripts/ppe.sh model` shows it and how it was accepted. |
| **What it finds** | three things: person, helmet, vest. Whether a person *wears* a helmet or vest is decided afterwards by a rule: the helmet must sit on the head, the vest on the torso. The rule uses body keypoints, so bending and crouching people are judged too. |
| **Speed** | 8 cameras at 10 frames a second each, on a MacBook Air M4 |
| **Time to alert** | about 4 s after the violation starts, 3 s of which is deliberate: it must last 3 s. On the phone: about 5 s as text only, 7 s with the picture on your connection. |
| **How good it is** | On 266 test clips it catches 83 % of real violations, and 89 % of its alerts are real. On photos of people in caps and hats it wrongly judges 2.5 % "helmet worn". Details: README, "Measured". |

## 2. Try it in five minutes

```bash
bash ~/Documents/ppe-monitor/scripts/ppe.sh photo ~/Desktop/site.jpg     # any photo with people in it
```

The checked photo is written next to the original, as `site_checked.jpg`:
- a **red** box means no helmet;
- a **green** box means a helmet is worn;
- a **grey** box means it can't judge: the person is too small, cut off, or has their head out of sight.

Each person's verdict is also printed. A folder of photos works too.

A photo is a single moment. The live system waits until a violation has lasted 3 s before it
alerts, which smooths out one-frame mistakes. So a photo can occasionally disagree with what the
cameras would do.

```bash
bash ~/Documents/ppe-monitor/scripts/ppe.sh video ~/Desktop/clip.mp4     # any video file
```

This plays the video in a window, with a box on each person:
- **green:** all fine;
- **amber:** a violation is being confirmed;
- **red:** a confirmed violation, which becomes an event.

Press `q` to stop. You get an annotated copy in `runs/videos/`, and the events with their pictures
in `runs/monitor/`. Add `--headless` to skip the window. Add a camera's id after the file name
(`... clip.mp4 gate`) to use that camera's zones and rules.

```bash
bash ~/Documents/ppe-monitor/scripts/ppe.sh start
```

This is the whole system on the cameras in `configs/cameras.yaml`: for now, the three simulated
cameras.
- The **dashboard** opens in your browser at <http://127.0.0.1:8080>.
- Alerts go to your **Telegram** once it is connected: `bash scripts/ppe.sh telegram`, one time
  only.
- **Ctrl+C** stops the cameras, the alerts and the dashboard. The database keeps running;
  `bash scripts/ppe.sh stop` stops it.

### The PPE Monitor app: double-click instead of typing

```bash
bash ~/Documents/ppe-monitor/scripts/ppe.sh app            # once: makes the app
bash ~/Documents/ppe-monitor/scripts/ppe.sh app --login    # ... and starts it whenever you log in
```

This puts **PPE Monitor** (a yellow helmet icon) in your Applications folder and on your Desktop.

**Double-click it:**
- If the monitor isn't running, a Terminal window opens and runs the whole system. The dashboard
  opens in your browser. **Leave that window open; closing it stops the monitor.** You can
  minimise it.
- If the monitor is already running, it just opens the dashboard.

**While it runs, the app looks after it:**
- **Parts that stop are started again.** If the camera service stops, it restarts after 10 s,
  then 20, 40 ... up to 5 minutes if it keeps stopping. So do the alert worker and the dashboard.
  Each restart is written in the window.
- **The Mac stays awake** while plugged in. Closing a laptop's lid still puts it to sleep, so keep
  the lid open.
- **The first time,** macOS asks whether "PPE Monitor" may control Terminal: answer OK.

**Starting on its own after a power cut:** `--login` starts it when you log in. For that to happen
unattended:
- turn on automatic log-in for this user (System Settings > Users & Groups);
- on a Mac mini, also turn on "Start up automatically after a power failure" (System Settings >
  Energy).

Automatic log-in means anyone at the Mac is logged in, so keep the Mac somewhere locked.

**Other options:**
- `ppe.sh app --no-login` stops the start at log-in.
- `ppe.sh app --remove` removes the app; the project is untouched.
- Make the app again (`ppe.sh app`) after moving the project folder or changing the dashboard's
  port.

## 3. Use it on your cameras

### 3.1 Add each camera

The system reads a camera's **RTSP stream**, the address that NVR and viewing apps use. Your Mac
must be on the same network as the cameras.

1. Find the camera's address in its manual or its web page. Use the **sub-stream** (lower
   resolution, about 720p): the model looks at 640 pixels anyway, and it saves network and processing.

   | Make | Sub-stream address |
   |---|---|
   | Hikvision | `rtsp://USER:PASS@IP:554/Streaming/Channels/102` |
   | Dahua, CP Plus | `rtsp://USER:PASS@IP:554/cam/realmonitor?channel=1&subtype=1` |

2. Put the password in `configs/secrets.env`, one line per camera, so it is never written in a
   settings file:

   ```
   CAM_GATE_PASSWORD=the-camera-password
   ```

3. Add the camera to `configs/cameras.yaml`, and delete the simulated ones (or add `enabled: false`
   to each):

   ```yaml
   cameras:
     - id: gate                       # short: lowercase letters, digits, - or _
       name: Main gate                # shown on alerts and the dashboard
       url: rtsp://admin:${CAM_GATE_PASSWORD}@192.168.1.64:554/Streaming/Channels/102
       transport: tcp
   ```

4. Check it: `bash scripts/ppe.sh camera gate` opens a window with the camera's picture and the
   system's boxes. Press `q` to close it.

**Where cameras should be.** These rules come from what the tests showed.
- **High up, looking down at an angle,** in landscape.
- **People large enough:** at least a sixth of the picture's height. Smaller people are not judged
  (grey), and a worker about 90 pixels tall in a 720p picture was missed.
- **The whole of a restricted zone in view,** including where people's feet will be.
- **Machines at the frame's edge can be taken for people.** Frame them out where you can.

### 3.2 Restricted zones and rules

```bash
bash ~/Documents/ppe-monitor/scripts/ppe.sh zones gate      # click the zone's corners on the camera's picture
```

1. Go round the area on the **ground** where people must not stand. The system tests where each
   person's feet are.
2. Press Enter to finish the zone, give it an id and a name, and press `s` to save.
3. Add a rule for it in `configs/rules.yaml`, then check both files:
   `bash scripts/ppe.sh rules`.

```yaml
  - id: gate-keep-out
    type: zone_intrusion
    cameras: [gate]
    zone: keep-out
    severity: high            # low | medium | high | critical
    dwell: 2.0                # seconds inside before it is an event
    active: {hours: "07:00-19:00", days: mon-sat}     # optional
```

- The **helmet rule** covers all cameras already.
- The **vest rule** is off (`enabled: false`). Remove that line to switch it on.

### 3.3 Who gets alerts

These settings are in `configs/server.yaml`:
- **Telegram:** `bash scripts/ppe.sh telegram` connects it. To alert a group, add the bot to the
  group and put the group's chat id in `TELEGRAM_CHAT_ID`; several ids go on one line, separated
  by commas.
- **Email:** fill in the `SMTP_...` and `EMAIL_...` lines of `configs/secrets.env` and set
  `email: enabled: true`.
- **Quiet events:** `min_severity` per channel. Events below it stay on the dashboard only.
- **No pictures:** `photo: false` sends text and a dashboard link instead. It is faster (about
  5 s instead of 7 s), and the workers' pictures don't leave your Mac.
- **Floods:** at most 5 alerts per camera in 10 minutes. The rest arrive as one summary, and
  nothing is lost.
- **Camera down:** a camera silent for 60 s, or the whole camera service, sends its own alert.

### 3.4 Start it

Double-click **PPE Monitor**, or:

```bash
bash ~/Documents/ppe-monitor/scripts/ppe.sh start --keep-running
```

Keep the Mac plugged in, and the Terminal window open. The dashboard is at <http://127.0.0.1:8080>.

**On your phone's browser**, on the same Wi-Fi:
1. Set `dashboard: host: 0.0.0.0` in `configs/server.yaml`.
2. Add `DASHBOARD_PASSWORD=...` to `configs/secrets.env`. The user name is `safety`.
3. Open `http://<the Mac's IP address>:8080`.
4. Also set `dashboard_url` in `configs/server.yaml`, so that alerts link to their event.

## 4. Every day

- **The dashboard** shows each camera's live picture and the counts by hour and camera. It also
  lists the events, each with its evidence picture. Open an event and press **Confirm violation**
  or **False alarm**.
- **Mark every false alarm.** Its clean picture is kept and becomes training material (section 5).
- **An alert** reads like this: *"🟠 HIGH · No helmet · Main gate"*, then *"14:03:12, 29 Sep:
  person #7 without it for 3 s"*, with the picture.
- **Evidence pictures** are kept for 30 days (`storage: keep_days`). Those marked "false alarm" are
  kept longer, for retraining.

## 5. Keeping it accurate

- **Measure it on your own site:** this is the most useful next step.
  - Record a few 1-3 minute clips from your fixed cameras, into `data/clips/`.
  - Mark their violations: `bash scripts/mac_phase6.sh annotate data/clips/<clip>.mp4`.
  - Run `bash scripts/ppe.sh evaluate`. It gives the five headline numbers, including your clips.
- **Learn from its false alarms.** Once at least 10 are marked on real cameras:
  - `bash scripts/ppe.sh feedback` turns them into labelled training pictures.
  - `bash scripts/ppe.sh retrain` fine-tunes, compares and evaluates.
  - `bash scripts/mac_phase6.sh accept` puts the new model into use, but only if it passes every
    check. Test footage is never trained on.
  - If an alert was on something that isn't a person (a machine, a sign), delete that "person" box
    in the training label. `datasets/feedback/review.html` shows every picture with its boxes.
- **Go back to the previous model** if you ever need to:
  - run `touch ~/Documents/ppe-monitor/models/ppe4_yolo26n_finetune.pt`, since the newest `.pt` in
    `models/` is the one in use;
  - set `model: ppe4_yolo26n_finetune` in `configs/ppe.yaml`;
  - restart with `ppe.sh start`.

  Its Core ML files are still in `models/exported/`.
- **Is everything working?** `bash scripts/ppe.sh check` takes about 3 minutes: the tests, and a
  short run through the whole system.

## 6. The model on its own (for developers)

| | |
|---|---|
| **Weights** | `models/ppe4caps_yolo26n_caps2.pt`: Ultralytics YOLO, input 640 |
| **Other formats** | Core ML: `models/exported/ppe4caps_yolo26n_caps2/coreml/…mlpackage` (also 8-bit). ONNX: `…/onnx/…onnx` (batch 8) and `…/onnx-int8/` |
| **Classes** | 0 person, 1 helmet, 2 vest |
| **Thresholds** | person 0.40, helmet 0.25, vest 0.25 (`configs/ppe.yaml`) |
| **Model card** | `models/ppe4caps_yolo26n_caps2.md`: data, training, the overfitting check, results per class |

The detector alone, from Python in the project's environment (`conda activate ppe`):

```python
from ultralytics import YOLO
model = YOLO("models/ppe4caps_yolo26n_caps2.pt")
r = model.predict("photo.jpg", imgsz=640, conf=0.25)[0]
for cls, conf, box in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
    print(model.names[int(cls)], round(conf, 2), [round(v) for v in box])
```

The detector finds helmets. It does not say who is **wearing** one. For that, use the project's
rule, the same one the cameras use:

```python
import cv2, sys
sys.path.insert(0, "src")
from ppe_monitor.pipeline import PPEMonitor
from ppe_monitor.vision.detector import load_model

monitor = PPEMonitor(load_model("models/ppe4caps_yolo26n_caps2.pt"), camera="photo", fps=1.0)
result = monitor.process(cv2.imread("photo.jpg"), t=0.0)
for track in result.tracks:
    v = result.verdicts[track.track_id]
    print(track.track_id, "helmet:", v.helmet, "vest:", v.vest)     # worn / missing / unknown
```

Feed it a video's frames in order, with `t` in seconds, and `result.events` holds the confirmed
events. `scripts/monitor.py` does exactly that.

## 7. On a Linux server, with Docker

```bash
docker compose up -d --build       # the whole system; first time ~3-10 min
docker compose logs secrets        # the dashboard's password (user: safety)
```

- The models, `configs/` and `data/` are shared with the Mac setup.
- For real cameras, edit `configs/cameras.yaml` as in section 3.
- Put Telegram's token and chat id in `configs/secrets.env`.
- Docker on a Mac can't use the Apple GPU, so on the Mac the native `ppe.sh start` is the fast way.

## 8. What it can't do yet

- **It hasn't been measured on your own fixed cameras.** Its numbers come from public photos, 27 s
  of hand-held video from your workshop, and 1.2 minutes from other sites (section 5 fixes this).
- **Small, distant people** aren't judged, and so aren't alerted.
- **Machines at a frame's edge** have been taken for people.
- **Bicycle helmets and hoods** can confuse it: it learnt hard hats.
- **Half-hidden or very small helmets:** since the caps fine-tuning, 5 of 350 on the test photos
  are no longer found, and 2 others now are.
- **Vests:** if you switch the vest rule on, check it on your cameras first. The current model
  judges wearers "no vest" a little more often than the first one did.
- **Licences:** the code is AGPL-3.0 (`LICENSE`), like Ultralytics. The model is for non-commercial
  use: two of its training datasets are non-commercial or have no licence (README, "Licence").

## 9. When something goes wrong

| What you see | What to do |
|---|---|
| "Another … is already running" | A previous command is still running in another window. Let it finish, or stop it there with Ctrl+C. |
| The dashboard doesn't open | Look at `logs/dashboard.log`. Another program may be using port 8080: change `dashboard: port`. |
| No alerts on the phone | Run `bash scripts/ppe.sh telegram` again (in Telegram, the bot must have been started with **Start**). Check `min_severity`, and `logs/alert_worker.log`. |
| A camera stays "connecting" | Check its address with `bash scripts/ppe.sh camera <id>`, and that the Mac and the camera are on the same network. Try the sub-stream. |
| "set neither in the environment nor in configs/secrets.env" | The camera address uses a `${…}` password that isn't in `configs/secrets.env`. |
| Cameras below 10 frames a second | Too many cameras for this Mac. Use fewer, or lower `process_fps` in `configs/streams.yaml` (not below 5). |
| "No Core ML version" after changing the model | `bash scripts/mac_phase4.sh export` |
| Anything else | `bash scripts/ppe.sh check` tells what works. Logs are in `logs/`. |

## 10. Where things are

| | |
|---|---|
| `scripts/ppe.sh` | everyday commands (this guide) |
| `~/Applications/PPE Monitor.app` | the double-click app (`ppe.sh app` makes it again) |
| `configs/` | cameras, zones, rules, alerts, and secrets (`secrets.env`, never shared) |
| `models/` | the model in use (newest `.pt`), older ones, `candidates/`, `exported/` |
| `data/events/` | evidence pictures of events, by day |
| `runs/` | results of checks, evaluations, comparisons, and the videos and photos you checked |
| `docs/` | results of each phase, decisions (`decisions/`), and the book (`book/`) |
| `README.md` | the project: status, measured numbers, limits, every command |
| `scripts/push_to_github.sh` | puts the project on GitHub as a private repository (signs in through your browser the first time); run it again to send later changes. Clips, data, logs and secrets stay on the Mac |
