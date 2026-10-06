import platform

import pytest

from ppe_monitor.executables import encoder_args, encoder_works, ffmpeg_encoders, find_ffmpeg, pick_encoder


def test_best_device_and_gpu_report_agree():
    pytest.importorskip("torch")
    from ppe_monitor.device import best_device, gpu_report, why_no_gpu

    device = best_device()
    assert device in {"cuda", "mps", "cpu"}
    report = gpu_report(matmul_size=256, image_size=64, batch=1)  # small: this is a test, not a benchmark
    assert report.device == device
    if device == "cpu":
        assert why_no_gpu()
        assert not report.gpu_ok
    else:
        # On a GPU machine (e.g. the Mac): results must match the CPU, and every benchmark must run.
        assert report.gpu_ok
        assert set(report.speedups) == {"matmul", "cnn_fp32", "cnn_fp16"}


def test_apple_silicon_mac_uses_its_gpu():
    if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
        pytest.skip("only meaningful on an Apple Silicon Mac")
    pytest.importorskip("torch")
    from ppe_monitor.device import best_device

    assert best_device() == "mps"


def test_live_and_file_encoder_settings_differ():
    assert "zerolatency" in encoder_args("libx264", live=True)
    assert "-crf" in encoder_args("libx264", live=False)
    assert "-realtime" in encoder_args("h264_videotoolbox", live=True)
    assert "-realtime" not in encoder_args("h264_videotoolbox", live=False)


@pytest.mark.skipif(not find_ffmpeg(), reason="needs ffmpeg")
def test_auto_encoder_really_works():
    ffmpeg = find_ffmpeg()
    chosen = pick_encoder(ffmpeg)
    assert encoder_works(ffmpeg, chosen)
    if platform.system() == "Darwin" and platform.machine() == "arm64" and "h264_videotoolbox" in ffmpeg_encoders(ffmpeg):
        assert chosen == "h264_videotoolbox"  # the Mac's hardware encoder, not the CPU


@pytest.mark.skipif(not find_ffmpeg(), reason="needs ffmpeg")
def test_listed_but_broken_encoder_is_rejected():
    ffmpeg = find_ffmpeg()
    with pytest.raises(RuntimeError):
        pick_encoder(ffmpeg, "definitely_not_an_encoder")
