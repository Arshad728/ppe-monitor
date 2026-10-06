"""A fake CCTV network on your own machine: MediaMTX as the RTSP server, plus one ffmpeg
process per camera that loops a video file into its RTSP path forever.

    video file --ffmpeg (loops, re-encodes to H.264)--> MediaMTX :8554/cam1 <-- OpenCV reads it

From the reading side this is indistinguishable from a real IP camera, which is the point:
every later phase can be developed and tested without access to a real site.
"""

from __future__ import annotations

import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO
from urllib.parse import urlparse

import yaml

from ..config import PROJECT_ROOT, Camera
from ..executables import encoder_args, find_ffmpeg, find_mediamtx, pick_encoder
from ..ingestion.rtsp import rtsp_describe

DEFAULT_SERVER_CONFIG = PROJECT_ROOT / "configs" / "mediamtx.yml"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "fake_cameras"
# A publisher that ran this long before exiting counts as healthy; its restart count resets.
HEALTHY_UPTIME_S = 30.0


class FakeCameraError(RuntimeError):
    pass


def build_publish_command(
    ffmpeg: str, source: Path, url: str, *, encoder: str = "libx264", copy: bool = False
) -> list[str]:
    """The ffmpeg command that plays `source` in real time, forever, into `url`.

    -re            read the file at its natural frame rate, like a live camera
    -stream_loop -1  start again from the beginning when the file ends
    -threads 1     decode the file on one thread. A multi-threaded decoder keeps about one
                   frame per CPU core in flight and is reset at every loop, which froze the
                   stream for ~cores/fps seconds each time (1.1 s on a 10-core Mac at 10 fps)
    copy=True      send the file's own encoded video untouched (cheapest, but the file must
                   already be H.264/H.265 with regular keyframes, or readers join slowly)
    otherwise      re-encode to H.264 with no B-frames and a keyframe every second, which
                   is what most IP cameras send and lets a reader start within a second
    """
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-re", "-stream_loop", "-1",
           "-threads", "1", "-i", str(source), "-an"]
    if copy:
        cmd += ["-c:v", "copy"]
    else:
        cmd += ["-c:v", encoder, *encoder_args(encoder, live=True), "-pix_fmt", "yuv420p", "-bf", "0",
                "-force_key_frames", "expr:gte(t,n_forced*1)"]
    # "localhost" as 127.0.0.1: a statically built ffmpeg (the one in the Docker image) crashes when
    # it has to look a name up; the stream is the same either way
    cmd += ["-f", "rtsp", "-rtsp_transport", "tcp", url.replace("://localhost:", "://127.0.0.1:", 1)]
    return cmd


def host_port(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    return parsed.hostname or "localhost", parsed.port or 554


def port_is_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def write_server_config(out_path: Path, *, port: int, tcp_only: bool = False,
                        base_config: Path = DEFAULT_SERVER_CONFIG) -> Path:
    """Copy the project's MediaMTX config with a different RTSP port (used by the tests,
    so they never collide with a fake network you already have running on 8554)."""
    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    config["rtspAddress"] = f"127.0.0.1:{port}"
    if tcp_only:
        config["rtspTransports"] = ["tcp"]
    out_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return out_path


@dataclass
class _Publisher:
    camera: Camera
    command: list[str]
    log_path: Path
    process: subprocess.Popen | None = None
    log_file: IO | None = None
    restarts: int = 0
    started_at: float = 0.0
    next_restart_at: float = 0.0
    failed: bool = False


@dataclass
class FakeCameraNetwork:
    cameras: list[Camera]
    start_server: bool = True
    server_config: Path = DEFAULT_SERVER_CONFIG
    encoder: str = "auto"
    copy: bool = False
    log_dir: Path = DEFAULT_LOG_DIR
    max_restarts: int = 5
    _server: subprocess.Popen | None = field(default=None, init=False)
    _server_log: IO | None = field(default=None, init=False)
    _publishers: list[_Publisher] = field(default_factory=list, init=False)

    # ------------------------------------------------------------------ setup
    def _check_inputs(self) -> tuple[str, str, int]:
        if not self.cameras:
            raise FakeCameraError("No simulated cameras to start (no camera in the config has sim_source)")
        missing = [c for c in self.cameras if c.sim_source is None or not c.sim_source.is_file()]
        if missing:
            listing = "\n".join(f"  {c.id}: {c.sim_source}" for c in missing)
            raise FakeCameraError(
                "These cameras' sim_source files don't exist:\n" + listing +
                "\nRun `python scripts/make_test_clips.py`, or point sim_source at your own clips."
            )
        endpoints = {host_port(c.url) for c in self.cameras}
        if len(endpoints) != 1:
            raise FakeCameraError(f"All simulated cameras must use the same RTSP server, found {endpoints}")
        host, port = endpoints.pop()
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise FakeCameraError("ffmpeg not found. On macOS: `brew install ffmpeg` (see README).")
        return ffmpeg, host, port

    def start(self, ready_timeout: float = 20.0) -> None:
        ffmpeg, host, port = self._check_inputs()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            if self.start_server:
                self._start_server(host, port)
            elif not port_is_open(host, port):
                raise FakeCameraError(
                    f"No RTSP server is listening on {host}:{port}. Start MediaMTX first "
                    "(or pass --with-server to let this script start it)."
                )

            encoder = "copy" if self.copy else pick_encoder(ffmpeg, self.encoder)
            for cam in self.cameras:
                cmd = build_publish_command(ffmpeg, cam.sim_source, cam.url,
                                            encoder=encoder, copy=self.copy)
                pub = _Publisher(camera=cam, command=cmd, log_path=self.log_dir / f"{cam.id}.log")
                self._publishers.append(pub)
                self._launch(pub)

            not_ready = self.wait_until_ready(ready_timeout)
            if not_ready:
                details = "\n".join(f"  {cam_id}: see {self.log_dir / (cam_id + '.log')}" for cam_id in not_ready)
                raise FakeCameraError(
                    f"These cameras never became readable within {ready_timeout:.0f} s:\n{details}")
        except BaseException:
            self.stop()  # never leave a half-started network (orphaned ffmpeg/MediaMTX) behind
            raise

    def _start_server(self, host: str, port: int) -> None:
        mediamtx = find_mediamtx()
        if not mediamtx:
            raise FakeCameraError(
                "MediaMTX not found. Run `python scripts/get_mediamtx.py` (or put a mediamtx binary on your PATH), "
                "or start it yourself and drop --with-server."
            )
        if port_is_open(host, port):
            raise FakeCameraError(
                f"Something is already listening on {host}:{port} - probably a MediaMTX you started "
                "earlier. Stop it, or run without --with-server to reuse it."
            )
        self._server_log = open(self.log_dir / "mediamtx.log", "a", encoding="utf-8")
        self._server = subprocess.Popen([mediamtx, str(self.server_config)],
                                        stdout=self._server_log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self._server.poll() is not None:
                raise FakeCameraError(
                    f"MediaMTX exited immediately (code {self._server.returncode}); "
                    f"see {self.log_dir / 'mediamtx.log'}"
                )
            if port_is_open(host, port):
                return
            time.sleep(0.2)
        raise FakeCameraError(f"MediaMTX did not start listening on port {port} within 10 s")

    def _launch(self, pub: _Publisher) -> None:
        if pub.log_file is None:
            pub.log_file = open(pub.log_path, "a", encoding="utf-8")
        pub.log_file.write(f"\n--- start {time.strftime('%H:%M:%S')}: {' '.join(pub.command)}\n")
        pub.log_file.flush()
        pub.process = subprocess.Popen(pub.command, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=pub.log_file)
        pub.started_at = time.monotonic()

    # ---------------------------------------------------------------- running
    def wait_until_ready(self, timeout: float) -> list[str]:
        """Wait until the server reports every camera's stream as live. Returns the ids of
        cameras that never became ready."""
        pending = {pub.camera.id: pub.camera for pub in self._publishers}
        deadline = time.monotonic() + timeout
        while pending and time.monotonic() < deadline:
            self.poll()
            for cam_id, cam in list(pending.items()):
                if rtsp_describe(cam.url) == 200:  # the server has a live stream on this path
                    del pending[cam_id]
            if pending:
                time.sleep(0.5)
        return list(pending)

    def poll(self) -> list[str]:
        """Restart publishers that have died (with back-off). Returns messages about what happened."""
        messages = []
        if self._server is not None and self._server.poll() is not None:
            messages.append(f"MediaMTX stopped (code {self._server.returncode}); see {self.log_dir / 'mediamtx.log'}")
        now = time.monotonic()
        for pub in self._publishers:
            if pub.failed or pub.process is None or pub.process.poll() is None:
                continue
            if pub.next_restart_at == 0.0:
                if now - pub.started_at > HEALTHY_UPTIME_S:
                    pub.restarts = 0  # it had been running fine; don't hold old crashes against it
                if pub.restarts >= self.max_restarts:
                    pub.failed = True
                    messages.append(f"{pub.camera.id}: publisher keeps crashing, giving up (see {pub.log_path})")
                    continue
                delay = min(10.0, 2 ** pub.restarts)
                pub.next_restart_at = now + delay
                messages.append(f"{pub.camera.id}: publisher exited (code {pub.process.returncode}), "
                                f"restarting in {delay:.0f} s")
            elif now >= pub.next_restart_at:
                pub.restarts += 1
                pub.next_restart_at = 0.0
                self._launch(pub)
        return messages

    def pause(self, camera_id: str) -> None:
        """Stop one camera's publisher and keep it stopped, as if the camera lost power or its
        cable was pulled (for testing reconnection). resume() brings it back."""
        for pub in self._publishers:
            if pub.camera.id == camera_id:
                pub.failed = True          # poll() leaves it alone
                _terminate(pub.process)
                return
        raise KeyError(camera_id)

    def resume(self, camera_id: str) -> None:
        for pub in self._publishers:
            if pub.camera.id == camera_id:
                pub.failed, pub.restarts, pub.next_restart_at = False, 0, 0.0
                if pub.process is None or pub.process.poll() is not None:
                    self._launch(pub)
                return
        raise KeyError(camera_id)

    @property
    def server_running(self) -> bool:
        return self._server is None or self._server.poll() is None

    def status(self) -> list[tuple[str, str]]:
        rows = []
        for pub in self._publishers:
            if pub.failed:
                state = "FAILED"
            elif pub.process is not None and pub.process.poll() is None:
                state = "streaming" + (f" (restarted {pub.restarts}x)" if pub.restarts else "")
            else:
                state = "restarting"
            rows.append((pub.camera.id, state))
        return rows

    # --------------------------------------------------------------- teardown
    def stop(self) -> None:
        for pub in self._publishers:
            _terminate(pub.process)
            if pub.log_file:
                pub.log_file.close()
                pub.log_file = None
        self._publishers.clear()
        _terminate(self._server)
        self._server = None
        if self._server_log:
            self._server_log.close()
            self._server_log = None

    def __enter__(self) -> "FakeCameraNetwork":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def _terminate(proc: subprocess.Popen | None, timeout: float = 5.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)


__all__ = ["FakeCameraNetwork", "FakeCameraError", "build_publish_command", "write_server_config",
           "port_is_open", "host_port", "DEFAULT_SERVER_CONFIG"]
