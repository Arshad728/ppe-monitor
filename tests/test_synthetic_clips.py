import cv2
import numpy as np

from ppe_monitor.executables import find_ffmpeg
from ppe_monitor.sim.synthetic import clip_fps, make_synthetic_clip


def test_synthetic_clip_has_requested_shape_and_length(tmp_path):
    info = make_synthetic_clip(tmp_path / "c.mp4", "TEST", seconds=2, fps=10, size=(320, 180), seed=1)
    assert info.path.is_file()
    assert info.h264 == bool(find_ffmpeg())

    cap = cv2.VideoCapture(str(info.path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    assert len(frames) == 20
    assert frames[0].shape == (180, 320, 3)
    assert clip_fps(info.path) == 10
    # frames change over time (people walk, the counter ticks), and are never flat
    assert not np.array_equal(frames[0], frames[-1])
    assert all(f.std() > 10 for f in frames)


def test_clip_fps_of_non_video_is_none(tmp_path):
    bogus = tmp_path / "not_a_video.mp4"
    bogus.write_text("hello", encoding="utf-8")
    assert clip_fps(bogus) is None
