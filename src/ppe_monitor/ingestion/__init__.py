"""Reading live camera streams.

Phase 0: `rtsp.open_capture` and `rtsp.probe_stream` (open one stream and measure it).
Phase 4: one reader thread + latest-frame queue per camera, with reconnects.
"""

from .rtsp import StreamReport, open_capture, open_capture_with_mode, probe_stream, rtsp_describe

__all__ = ["StreamReport", "open_capture", "open_capture_with_mode", "probe_stream", "rtsp_describe"]
