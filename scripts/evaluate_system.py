#!/usr/bin/env python3
"""Phase 6: Chapter 4's five headline numbers, measured end to end on the test clips.

    python scripts/evaluate_system.py                  # every test clip, 8 at a time: ~10 min on the Mac
    python scripts/evaluate_system.py --sets site      # only your own clips (data/ground_truth/*.yaml)
    python scripts/evaluate_system.py --telegram 5     # also send 5 of the alerts to your phone, to time them
    bash scripts/mac_phase6.sh evaluate                # the same, on the Mac

Each test clip plays ONCE as a camera, at its own speed, up to 8 at a time (like 8 CCTV cameras),
through the real system: the camera service (detector, keypoints, tracker, rules, events), the
database writer, the alert worker, and a stand-in for Telegram on this machine that records when
each alert arrives. Nothing reaches your phone, unless you ask for some with --telegram (then only
alerts from the public test photos, never from your own footage). A throw-away database is used
(`ppe_eval` on the project's PostgreSQL, or a SQLite file), so the dashboard's data is untouched.

Every alert that arrives is then matched to the clip's ground truth (src/ppe_monitor/evaluation/):
the violations really in it, with when they begin and end and where the person is. Clip sets:

    still     datasets/event_clips        86 clips: slow pan and zoom over labelled test photos
    moving    datasets/event_clips_sway   86 clips: the same photos sliding at walking pace
    zone      datasets/event_clips_zone   86 clips: people walking into a restricted zone
    site      data/ground_truth/*.yaml    your own footage, annotated by hand (scripts/annotate_clip.py)

The five numbers (book, Chapter 4 and Appendix B):
    mAP@50            from the model card: held-out test photos neither trained nor tuned on
    throughput        frames analysed per second, per camera, while 8 cameras run at once (this run)
    alert latency     from the moment a violation begins on camera to the alert arriving
    missed violations event recall: the share of real, sustained violations that produced an alert
    false alarms      event precision: the share of alerts that were about a real violation; and
                      false alerts per camera per hour of footage

Rules: the PPE rules in configs/rules.yaml, as they are (a rule switched off there is not
evaluated); on the zone clips, one zone rule with a 2 s dwell. The rate limit is off for the
evaluation (it would hold back alerts that must be timed); everything else is as configured.
Report: runs/evaluation/<time>/report.md (+ metrics.json, alerts.csv, the evidence images).
"""

import argparse
import csv
import json
import platform
import re
import statistics
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from ppe_monitor.backend import localpg
from ppe_monitor.backend.channels import SendError, TelegramChannel
from ppe_monitor.backend.db import alerts as alerts_t
from ppe_monitor.backend.db import events as events_t
from ppe_monitor.backend.settings import ServerSettings
from ppe_monitor.backend.sink import DbSink
from ppe_monitor.backend.standin import StandInTelegram
from ppe_monitor.backend.store import EventStore
from ppe_monitor.backend.worker import AlertWorker
from ppe_monitor.config import PROJECT_ROOT
from ppe_monitor.evaluation.score import AlertRecord, Summary, match_clip
from ppe_monitor.evaluation.truth import TRUTH_DIR, from_photo_clips, load_dir
from ppe_monitor.ingestion.reader import STOPPED
from ppe_monitor.pipeline import PPE_CONFIG, Settings
from ppe_monitor.rules.engine import Rule, RuleSet
from ppe_monitor.rules.zones import CameraZones, Zone
from ppe_monitor.service import CameraSource, MultiCameraMonitor, StreamSettings, choose_backend
from ppe_monitor.vision.backends import load
from ppe_monitor.vision.detector import MODELS_DIR, PRETRAINED_DIR

SETS = {"still": ("st-", PROJECT_ROOT / "datasets" / "event_clips"),
        "moving": ("mv-", PROJECT_ROOT / "datasets" / "event_clips_sway"),
        "zone": ("zn-", PROJECT_ROOT / "datasets" / "event_clips_zone"),
        "site": ("", TRUTH_DIR)}
TARGETS = {"map50": 0.85, "fps": 10.0, "latency_s": 5.0, "recall": 0.90, "precision": 0.90}
ZONE_DWELL = 2.0


class RoutedTelegram:
    """Telegram for the evaluation: the stand-in on this machine, except the first `real_n` alerts
    from public (not private) clips, which go to the real Telegram so their delivery is timed too."""
    name = "telegram"

    def __init__(self, stand_in: TelegramChannel, real: TelegramChannel | None, real_n: int, private: set[str]):
        self.stand_in, self.real, self.left, self.private = stand_in, real, real_n if real else 0, private
        self.real_ids: set[str] = set()
        self._lock = threading.Lock()

    def missing(self) -> str:
        return self.stand_in.missing()

    def ping(self) -> float:
        """Like the live worker, keep the real connection warm too (once a minute when idle), so a
        real alert isn't timed with a fresh connection that the live system wouldn't need."""
        if self.real is not None:
            try:
                self.real.ping()
            except SendError:
                pass
        return self.stand_in.ping()

    def send(self, msg, done):
        with self._lock:
            use_real = self.left > 0 and msg.event_id and msg.camera not in self.private
            if use_real:
                self.left -= 1
        if not use_real:
            return self.stand_in.send(msg, done)
        msg.title = "\U0001F9EA Evaluation (test clip) · " + msg.title
        try:
            out = self.real.send(msg, [])
        except SendError:
            with self._lock:
                self.left += 1
            raise
        self.real_ids.add(msg.event_id)
        return out


def clip_seconds(when: datetime | None, first: datetime) -> float | None:
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (when - first).total_seconds()


def fmt(v, digits=2, unit=""):
    return "-" if v is None or v != v else f"{v:.{digits}f}{unit}"


def pct(v):
    return "-" if v is None or v != v else f"{100 * v:.0f} %"


def ci(pair):
    return "" if not pair or pair[0] != pair[0] else f" (95 % CI {100 * pair[0]:.0f}–{100 * pair[1]:.0f} %)"


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", default="still,moving,zone,site", help="comma-separated: " + ", ".join(SETS))
    parser.add_argument("--cameras", type=int, default=8, help="clips playing at once")
    parser.add_argument("--limit", type=int, help="at most this many clips per set (a quick run)")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--weights", help="the detector to evaluate (default: the model in use, the newest .pt in models/)")
    parser.add_argument("--telegram", type=int, default=0, metavar="N",
                        help="send the first N alerts (public test clips only) to your real Telegram, to time delivery")
    parser.add_argument("--text-only", action="store_true",
                        help="send alerts without the picture (as photo: false in configs/server.yaml), to time that")
    parser.add_argument("--sqlite", action="store_true", help="use a SQLite file even if the project's PostgreSQL is set up")
    parser.add_argument("--database-url", help=argparse.SUPPRESS)
    parser.add_argument("--out", help="output folder (default runs/evaluation/<time>)")
    parser.add_argument("--rescore", metavar="DIR", help="score an earlier run (its folder) again, without playing the clips")
    args = parser.parse_args()

    wanted = [x.strip() for x in args.sets.split(",") if x.strip()]
    bad = [x for x in wanted if x not in SETS]
    if bad:
        print(f"Unknown set(s) {bad}: choose from {', '.join(SETS)}")
        return 2
    out = Path(args.rescore or args.out or PROJECT_ROOT / "runs" / "evaluation" / f"{datetime.now():%Y%m%d_%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)

    # -- settings and rules ------------------------------------------------------------------------
    settings, streams = Settings.load(PPE_CONFIG), StreamSettings.load()
    site = RuleSet.load()
    ppe_rules = [r for r in site.rules if r.type in ("no_helmet", "no_vest") and r.cameras is None]
    ppe_kinds = {r.type for r in ppe_rules}
    s = ServerSettings.load()
    s.storage_dir, s.live_dir = out / "events", out / "live"
    s.rate_max = 10 ** 6                                 # no rate limit: every alert must be timed
    for name, ch in s.channels.items():
        ch.enabled = name == "telegram"
        if args.text_only:
            ch.photo = False

    # -- the clips and their truth ------------------------------------------------------------------
    clips = []                                           # (set, camera id, truth, kinds evaluated)
    for name in wanted:
        prefix, where = SETS[name]
        if name == "site":
            truths = load_dir(where) if where.is_dir() else []
            truths = [t for t in truths if t.path.is_file()]
        elif (where / "clips.json").is_file():
            truths = from_photo_clips(where, zone_dwell=ZONE_DWELL)
        else:
            truths = []
        if not truths:
            print(f"  {name}: no clips ({where}); skipped")
            continue
        for t in truths[:args.limit] if args.limit else truths:
            kinds = {"zone_intrusion"} if t.zone else ppe_kinds
            t.episodes = [e for e in t.episodes if e.kind in kinds]
            t.checked = tuple(k for k in t.checked if k in kinds)
            clips.append((name, (prefix + Path(t.clip).stem)[:64], t, kinds))
    if not clips:
        print("No test clips found. Make them with: python scripts/make_event_clips.py (and --motion sway / pan)")
        return 1
    ppe_cams = tuple(c for n, c, t, k in clips if not t.zone)
    zone_cams = [(c, t) for n, c, t, k in clips if t.zone]
    rules = RuleSet([Rule(r.id, r.type, ppe_cams, r.zone, r.severity, r.dwell, r.cooldown, None, r.description)
                     for r in ppe_rules] if ppe_cams else [], {}, site.timezone, "evaluation")
    if zone_cams:
        rules.rules.append(Rule("zone", "zone_intrusion", tuple(c for c, _ in zone_cams), "zone", "high", dwell=ZONE_DWELL))
        for c, t in zone_cams:
            rules.zones[c] = CameraZones(c, [Zone("zone", t.zone.get("name", "Test zone"),
                                                  tuple(map(tuple, t.zone["polygon"])))])
    private = {c for n, c, t, k in clips if t.private}

    total_s = sum(t.duration for _, _, t, _ in clips)
    if args.rescore:
        # score an earlier run again, without playing the clips (after a change to the scoring)
        pb = json.loads((out / "playback.json").read_text(encoding="utf-8"))
        url = args.database_url or f"sqlite:///{out / 'evaluation.sqlite'}"
        store = EventStore(url, s.storage_dir, create=False)
        first_wall = {k: datetime.fromisoformat(v) for k, v in pb["first_wall"].items()}
        cam_fps, drift, batch_fps, step_ms = pb["cam_fps"], pb["drift"], pb["batch_fps"], pb["step_ms"]
        real_ids, stand_in_count, real_used = set(pb["real_ids"]), pb["stand_in_messages"], pb["real_telegram"]
        weights, backend_name, args.cameras = Path(pb["weights"]), pb["backend"], pb["cameras_at_once"]
        print(f"Scoring the run in {out} again ({store.engine.dialect.name}).")
    else:
        # -- database, model, alert worker ------------------------------------------------------------
        if args.database_url:
            url = args.database_url
        elif args.sqlite or not localpg.in_use(s):
            url = f"sqlite:///{out / 'evaluation.sqlite'}"
        else:
            localpg.start()
            url = localpg.ensure_database(s, "ppe_eval", fresh=True)
        store = EventStore(url, s.storage_dir)
        weights = Path(args.weights) if args.weights else max(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, default=None)
        if weights is None or not weights.is_file():
            print("No model in models/.")
            return 1
        pose = PRETRAINED_DIR / Path(settings.pose["model"]).name if settings.pose["enabled"] else None
        backend_name = choose_backend(args.backend if args.backend != "auto" else streams.backend, weights, pose)
        backend = load(backend_name, weights, pose)

        stand_in = StandInTelegram()
        real = None
        if args.telegram > 0:
            real = TelegramChannel.from_secrets(s.secrets, ip=getattr(s.channels.get("telegram"), "ip", "ipv4"))
            if real.missing():
                print(f"--telegram: {real.missing()}. Continuing with the stand-in only.")
                real = None
        channel = RoutedTelegram(TelegramChannel("000:STAND-IN", ["1"], api_base=stand_in.url), real, args.telegram, private)
        worker = AlertWorker(store, s, {"telegram": channel},
                             log=lambda m: print("  " + m) if ("gave up" in m or "can't" in m or "Error" in m) else None)
        run_id = store.start_run(host=platform.node(), backend=backend_name, model=weights.stem,
                                 camera_ids=[c for _, c, _, _ in clips])
        store.sync_config({c: f"{n}: {t.clip}" for n, c, t, _ in clips}, rules)
        sink = DbSink(store, s, run_id, log=lambda m: print("  " + m))
        wt = threading.Thread(target=worker.run, name="alert-worker", daemon=True)
        wt.start()

        print(f"{len(clips)} clips ({total_s / 60:.1f} min of video) from {', '.join(dict.fromkeys(n for n, *_ in clips))}; "
              f"{args.cameras} at a time; model {weights.stem}, backend {backend_name}; database {store.engine.dialect.name}; "
              f"alerts to a stand-in for Telegram" + (f" (and the first {args.telegram} to your phone)" if real else ""))
        print(f"Rules: {', '.join(sorted(ppe_kinds)) or 'none'} on the PPE clips" + (f", a zone rule ({ZONE_DWELL:g} s) on the zone clips" if zone_cams else ""))

        # -- play the clips, a batch at a time -------------------------------------------------------------
        first_wall: dict[str, datetime] = {}
        cam_fps: dict[str, float] = {}
        drift: dict[str, float] = {}
        batch_fps: list[float] = []                            # per camera, in full batches
        step_ms: list[float] = []
        t_run = time.monotonic()
        for b in range(0, len(clips), args.cameras):
            batch = clips[b:b + args.cameras]
            sources = [CameraSource(c, str(t.path)) for _, c, t, _ in batch]
            mon = MultiCameraMonitor(sources, backend, settings, rules, process_fps=streams.process_fps, imgsz=streams.imgsz,
                                     out_dir=None, sinks=[sink],
                                     reader_options={**streams.reader_options(), "loop_files": False})
            mon.start(warmup=b == 0)
            longest = max(t.duration for _, _, t, _ in batch)
            t0 = time.monotonic()
            while True:
                mon.step()
                readers = mon.readers.values()
                if all(r.stats.ended or r.stats.state == STOPPED for r in readers):
                    end = time.monotonic() + 0.5                 # analyse the last frames still waiting
                    while time.monotonic() < end:
                        mon.step()
                    break
                if time.monotonic() - t0 > longest + 60:
                    print("  (a clip didn't finish within a minute of its length; stopping this batch)")
                    break
            for cid, r in mon.readers.items():
                if r.stats.first_frame is not None:
                    first_wall[cid] = r.wall(r.stats.first_frame)
                    fps_clip = next(t.fps for _, c, t, _ in batch if c == cid)
                    drift[cid] = (r.stats.last_frame - r.stats.first_frame) - (r.stats.frames - 1) / fps_clip
                st = mon.stats[cid]
                if st.processed > 1 and len(st.done_times) > 1:
                    cam_fps[cid] = (len(st.done_times) - 1) / (st.done_times[-1] - st.done_times[0])
                    if len(batch) == args.cameras:
                        batch_fps.append(cam_fps[cid])
            step_ms += [1000 * x for x in mon.step_times]
            mon.stop()
            done = b + len(batch)
            print(f"  {done:3d}/{len(clips)} clips played; {sink.stored} events stored, {worker.sent} alerts delivered "
                  f"({time.monotonic() - t_run:.0f} s)")

        sink.close(60)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            c = store.alert_counts()
            if not c.get("pending") and not c.get("sending"):
                break
            time.sleep(0.5)
        worker.stop()
        wt.join(10)
        worker.close()
        stand_in.close()
        real_ids, stand_in_count, real_used = set(channel.real_ids), len(stand_in.received), real is not None
        (out / "playback.json").write_text(json.dumps({
            "first_wall": {k: v.isoformat() for k, v in first_wall.items()}, "cam_fps": cam_fps, "drift": drift,
            "batch_fps": batch_fps, "step_ms": step_ms, "real_ids": sorted(real_ids), "stand_in_messages": stand_in_count,
            "real_telegram": real_used, "weights": str(weights), "backend": backend_name, "cameras_at_once": args.cameras,
            "database": re.sub(r"://[^@]*@", "://", url)}, indent=1), encoding="utf-8")

    # -- what arrived, against the truth --------------------------------------------------------------
    with store.engine.connect() as c:
        evs = c.execute(events_t.select()).all()
        als = c.execute(alerts_t.select()).all()
    by_id = {a.id: a for a in als}
    delivered: dict[str, datetime] = {}
    status_of: dict[str, str] = {}
    for a in als:
        if a.is_summary or a.channel != "telegram":
            continue
        status_of[a.event_id] = a.status
        if a.status == "sent" and a.sent_at:
            delivered[a.event_id] = a.sent_at
        elif a.status == "summarised" and a.summary_id in by_id and by_id[a.summary_id].sent_at:
            delivered[a.event_id] = by_id[a.summary_id].sent_at
    per_cam: dict[str, list] = {}
    for e in evs:
        per_cam.setdefault(e.camera, []).append(e)

    results, rows = [], []
    summaries = {"all": Summary()}
    for name, cid, t, kinds in clips:
        first = first_wall.get(cid)
        recs = []
        cam_events = per_cam.get(cid, [])
        for e in cam_events:
            if e.id not in delivered or first is None:
                continue
            recs.append(AlertRecord(e.id, e.kind, clip_seconds(e.started_at, first), clip_seconds(e.confirmed_at, first),
                                    clip_seconds(delivered[e.id], first), tuple(e.box) if e.box else None,
                                    e.id in real_ids))
        r = match_clip(t, cid, recs, events_stored=len(cam_events))
        if first is None:
            r.note = "the clip never played"
        results.append((name, r))
        for key in ("all", name, *(f"kind:{k}" for k in sorted(kinds))):
            summaries.setdefault(key, Summary())
        summaries["all"].add(r)
        summaries[name].add(r)
        for k in kinds:                                  # per kind: this clip's episodes and alerts of that kind only
            sub = match_clip(replace(t, episodes=[e for e in t.episodes if e.kind == k],
                                     checked=tuple(x for x in t.checked if x == k)), cid,
                             [a for a in recs if a.kind == k])
            summaries[f"kind:{k}"].add(sub)
        for o in r.alerts:
            a = o.alert
            ep = t.episodes[o.episode] if o.episode is not None else None
            rows.append({"set": name, "clip": t.clip, "camera": cid, "event_id": a.event_id, "kind": a.kind,
                         "started_s": fmt(a.started), "confirmed_s": fmt(a.confirmed), "delivered_s": fmt(a.delivered),
                         "verdict": o.verdict, "episode_start_s": fmt(ep.start) if ep else "",
                         "latency_s": fmt(a.delivered - ep.start) if ep and o.verdict == "true" and a.delivered is not None else "",
                         "real_telegram": a.real_channel,
                         "note": ep.note if ep else ("a person the labels don't cover" if o.verdict == "unscored"
                                                     and a.kind in t.checked else "")})
            if o.verdict in ("true", "duplicate", "false"):
                store.set_status(a.event_id, "confirmed" if o.verdict != "false" else "false_alarm",
                                 f"evaluation: {o.verdict}" + (f" ({ep.note})" if ep and ep.note else ""))
        for i in r.missed:
            ep = t.episodes[i]
            rows.append({"set": name, "clip": t.clip, "camera": cid, "event_id": "", "kind": ep.kind,
                         "started_s": "", "confirmed_s": "", "delivered_s": "", "verdict": "missed",
                         "episode_start_s": fmt(ep.start), "latency_s": "", "real_telegram": "", "note": ep.note})

    with (out / "alerts.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["set"])
        w.writeheader()
        w.writerows(rows)

    # -- the five numbers ------------------------------------------------------------------------------
    card_path = weights.with_suffix(".json")
    card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.is_file() else {}
    by_src = card.get("test_by_source", {})
    map_ppe3 = (by_src.get("test_ppe3") or {}).get("map50")
    map_all = (card.get("test") or {}).get("map50")
    n_ppe3 = (by_src.get("test_ppe3") or {}).get("images")
    n_all = (card.get("test") or {}).get("images")
    fps_med = statistics.median(batch_fps) if batch_fps else (statistics.median(cam_fps.values()) if cam_fps else None)
    fps_min = min(batch_fps) if batch_fps else None
    A = summaries["all"].as_dict()
    lat = A["latency"] or {}
    undelivered = sum(1 for e in evs if status_of.get(e.id) not in (None, "sent", "summarised"))
    no_alert = sum(1 for e in evs if e.id not in status_of)

    metrics = {"when": datetime.now().isoformat(timespec="seconds"), "machine": f"{platform.node()} ({platform.platform()})",
               "model": weights.stem, "backend": backend_name, "database": store.engine.dialect.name,
               "cameras_at_once": args.cameras, "process_fps": streams.process_fps, "clips": len(clips),
               "text_only": bool(args.text_only),
               "video_minutes": round(total_s / 60, 2), "sets": wanted, "rules": sorted(ppe_kinds) + (["zone_intrusion"] if zone_cams else []),
               "map50": {"ppe3_test": map_ppe3, "ppe3_images": n_ppe3, "all_test": map_all, "all_images": n_all},
               "throughput": {"median_fps": fps_med, "min_fps": fps_min, "per_camera_in_full_batches": len(batch_fps),
                              "loop_step_ms_median": statistics.median(step_ms) if step_ms else None},
               "summaries": {k: v.as_dict() for k, v in summaries.items()},
               "events_stored": len(evs), "events_without_alert": no_alert, "alerts_not_delivered": undelivered,
               "stand_in_messages": stand_in_count, "real_telegram_messages": len(real_ids),
               "reader_drift_s_max": max(drift.values()) if drift else None}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str), encoding="utf-8")

    def met(ok):
        return "**yes**" if ok else "**no**"

    lat_ok = bool(lat) and lat["p90"] <= TARGETS["latency_s"]
    rows5 = [
        ("mAP@50 (person, helmet, vest)", f"≥ {TARGETS['map50']}",
         f"{fmt(map_ppe3, 3)} on the {n_ppe3} Phase 1 test photos; {fmt(map_all, 3)} on all {n_all} test photos",
         met(map_ppe3 is not None and map_ppe3 >= TARGETS["map50"]), "model card (held out, never trained on)"),
        ("Throughput", f"{TARGETS['fps']:g} frames/s per camera, 8 cameras",
         f"{fmt(fps_med, 1)} frames/s per camera (slowest {fmt(fps_min, 1)}), {args.cameras} cameras at once",
         met(fps_med is not None and fps_med >= 0.95 * TARGETS["fps"] and args.cameras >= 8),
         f"this run, {backend_name}, with the database and alerts on"),
        ("Alert latency", f"≤ {TARGETS['latency_s']:g} s, incl. the dwell",
         f"median {fmt(lat.get('median'), 1, ' s')}, 90 % within {fmt(lat.get('p90'), 1, ' s')}, slowest {fmt(lat.get('max'), 1, ' s')}"
         if lat else "-", met(lat_ok), f"{lat.get('n', 0)} caught violations; violation start on camera to alert delivered"),
        ("Missed violations (event recall)", f"≥ {100 * TARGETS['recall']:.0f} % caught",
         f"{pct(A['recall'])} caught ({A['caught']}/{A['sustained']}){ci(A['recall_ci'])}",
         met(A["recall"] == A["recall"] and A["recall"] >= TARGETS["recall"]), "real, sustained violations"),
        ("False alarms (event precision)", f"≥ {100 * TARGETS['precision']:.0f} % of alerts real",
         f"{pct(A['precision'])} real ({A['true'] + A['duplicate']}/{A['true'] + A['duplicate'] + A['false']}){ci(A['precision_ci'])}; "
         f"{fmt(A['false_per_hour'], 1)} false alerts per camera-hour",
         met(A["precision"] == A["precision"] and A["precision"] >= TARGETS["precision"]),
         f"{A['false']} false alarm(s) in {60 * A['hours']:.0f} camera-minutes"),
    ]
    L = [f"# End-to-end evaluation: {datetime.now():%Y-%m-%d %H:%M}", "",
         f"{len(clips)} test clips ({total_s / 60:.1f} min of video) played once each as cameras, {args.cameras} at a time, "
         f"through the whole system: model `{weights.stem}` on {backend_name} at {streams.process_fps:g} frames/s per camera, "
         f"the {store.engine.dialect.name} database, the alert worker, and a stand-in for Telegram on this machine"
         + (f" (plus {len(real_ids)} alerts through the real Telegram)" if real_used else "")
         + f". Machine: {platform.node()}. Rules: {', '.join(metrics['rules'])} (configs/rules.yaml as it is; "
           "the rate limit off, so every alert is timed)."
         + (" Alerts sent as text only, without the picture." if args.text_only else ""), "",
         "## The five headline numbers (book, Chapter 4)", "",
         "| Metric | Target | Measured | Met? | Based on |", "|---|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in rows5]
    L += ["", "## By clip set", "",
          "| Set | Clips | Minutes | Real violations | Caught | Alerts | Real | Duplicate | False | Not scored | Recall | Precision | False / cam-h | Latency median / 90 % |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in [k for k in summaries if k != "all" and not k.startswith("kind:")] + ["all"]:
        d = summaries[key].as_dict()
        la = d["latency"] or {}
        L.append(f"| {'**all**' if key == 'all' else key} | {d['clips']} | {60 * d['hours']:.1f} | {d['sustained']} | {d['caught']} | "
                 f"{d['alerts']} | {d['true']} | {d['duplicate']} | {d['false']} | {d['unscored']} | {pct(d['recall'])} | "
                 f"{pct(d['precision'])} | {fmt(d['false_per_hour'], 1)} | {fmt(la.get('median'), 1)} / {fmt(la.get('p90'), 1)} s |")
    L += ["", "| Kind | Real violations | Caught | Recall | Alerts | False | Precision | Latency median / 90 % |",
          "|---|---|---|---|---|---|---|---|"]
    for key in [k for k in summaries if k.startswith("kind:")]:
        d = summaries[key].as_dict()
        la = d["latency"] or {}
        L.append(f"| {key[5:]} | {d['sustained']} | {d['caught']} | {pct(d['recall'])}{ci(d['recall_ci'])} | {d['alerts']} | "
                 f"{d['false']} | {pct(d['precision'])} | {fmt(la.get('median'), 1)} / {fmt(la.get('p90'), 1)} s |")
    te, ts_ = A["to_event"] or {}, A["to_send"] or {}
    L += ["", "## Where the time goes", "",
          f"- **Violation start to event** (detection, tracking, the vote and the dwell): median {fmt(te.get('median'), 1, ' s')}, "
          f"90 % within {fmt(te.get('p90'), 1, ' s')}. The dwell alone is "
          f"{', '.join(f'{r.dwell or settings.events.dwell:g} s ({r.type})' for r in ppe_rules)}"
          + (f", {ZONE_DWELL:g} s (zone)" if zone_cams else "") + ".",
          f"- **Event to alert delivered** (database, queue, sending): median {fmt(ts_.get('median'), 2, ' s')}, "
          f"90 % within {fmt(ts_.get('p90'), 2, ' s')}, to the stand-in on this machine."]
    rl, rs = A["real_latency"] or {}, A["real_to_send"] or {}
    if rl:
        L.append(f"- **Through the real Telegram** ({rl['n']} alerts, {'text only' if args.text_only else 'with the picture'}): "
                 f"violation start to delivered median "
                 f"{fmt(rl['median'], 1, ' s')} (slowest {fmt(rl['max'], 1, ' s')}); event to delivered median "
                 f"{fmt(rs.get('median'), 1, ' s')}.")
    L += [f"- Events stored: {len(evs)}; without an alert (severity below the channel's minimum): {no_alert}; "
          f"alerts not delivered: {undelivered}. Frame pacing of the clips: at most {fmt(metrics['reader_drift_s_max'], 2, ' s')} "
          f"late by the end of a clip (the times above assume none).", ""]
    L += ["## Every false alarm and every miss", ""]
    wrong = [r for r in rows if r["verdict"] in ("false", "missed")]
    if wrong:
        L += ["| Set | Clip | Kind | Verdict | At (s) | Note |", "|---|---|---|---|---|---|"]
        for r in wrong:
            L.append(f"| {r['set']} | {r['clip']} | {r['kind']} | {r['verdict']} | "
                     f"{r['confirmed_s'] if r['verdict'] == 'false' else r['episode_start_s']} | {r['note']} |")
        L += ["", "Evidence images of the false alarms: the `events/` folder next to this report (they are also marked "
              "\"false alarm\" in the evaluation database)."]
    else:
        L.append("None.")
    L += ["", "## How to read this", "",
          "- **Real violation** = a sustained ground-truth episode: the person is in view without the item (or in the zone) long "
          "enough to expect an alert. Shorter or mostly hidden ones, and people whose status wasn't labelled, are "
          "\"not scored\": an alert on them counts neither way.",
          "- **Precision** counts duplicates as real (each is about a real violation) but they are listed apart.",
          "- **95 % CI** = a Wilson interval: the range the true rate is likely in, given how few clips there are.",
          "- **False alerts per camera-hour** here come from clips where people are in view the whole time; a real "
          "camera also watches empty scenes, so its hourly rate is usually lower. The photo clips are made from public "
          "test photos, not real video: only the site clips are real footage."]
    report = "\n".join(L) + "\n"
    (out / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Report: {out / 'report.md'}")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
