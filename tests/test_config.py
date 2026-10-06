from pathlib import Path

import pytest
import yaml

from ppe_monitor.config import PROJECT_ROOT, ConfigError, get_camera, load_cameras


def write_config(tmp_path: Path, cameras: list[dict]) -> Path:
    path = tmp_path / "cameras.yaml"
    path.write_text(yaml.safe_dump({"cameras": cameras}), encoding="utf-8")
    return path


def test_project_camera_config_is_valid():
    cameras = load_cameras()
    assert [c.id for c in cameras] == ["cam1", "cam2", "cam3"]
    for cam in cameras:
        assert cam.is_simulated
        assert cam.url.startswith("rtsp://localhost:8554/")
        assert cam.sim_source.is_relative_to(PROJECT_ROOT / "data" / "clips")


def test_minimal_real_camera(tmp_path):
    path = write_config(tmp_path, [{"id": "gate", "url": "rtsp://10.0.0.5:554/stream1"}])
    (cam,) = load_cameras(path)
    assert cam.name == "gate"
    assert cam.transport == "tcp"
    assert cam.enabled and not cam.is_simulated


def test_password_comes_from_environment_and_is_masked(tmp_path, monkeypatch):
    monkeypatch.setenv("CAM_PW", "s3cret")
    path = write_config(tmp_path, [{"id": "gate", "url": "rtsp://admin:${CAM_PW}@10.0.0.5/s1"}])
    (cam,) = load_cameras(path)
    assert cam.url == "rtsp://admin:s3cret@10.0.0.5/s1"
    assert cam.safe_url == "rtsp://admin:****@10.0.0.5/s1"
    assert "s3cret" not in cam.safe_url


def test_unset_environment_variable_is_an_error(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_PW", raising=False)
    path = write_config(tmp_path, [{"id": "gate", "url": "rtsp://admin:${MISSING_PW}@10.0.0.5/s1"}])
    with pytest.raises(ConfigError, match="neither in the environment nor in configs/secrets.env"):
        load_cameras(path)


@pytest.mark.parametrize(
    "entry, message",
    [
        ({"id": "Cam 1", "url": "rtsp://h/x"}, "id"),
        ({"id": "cam1", "url": "http://h/x"}, "rtsp://"),
        ({"id": "cam1", "url": "rtsp://h/x", "transport": "quic"}, "transport"),
        ({"id": "cam1", "url": "rtsp://10.0.0.5/x", "sim_source": "a.mp4"}, "localhost"),
    ],
)
def test_invalid_entries_are_rejected(tmp_path, entry, message):
    with pytest.raises(ConfigError, match=message):
        load_cameras(write_config(tmp_path, [entry]))


def test_duplicate_ids_are_rejected(tmp_path):
    entries = [{"id": "cam1", "url": "rtsp://h/a"}, {"id": "cam1", "url": "rtsp://h/b"}]
    with pytest.raises(ConfigError, match="duplicate"):
        load_cameras(write_config(tmp_path, entries))


def test_disabled_cameras_are_skipped_unless_asked(tmp_path):
    entries = [{"id": "a", "url": "rtsp://h/a"}, {"id": "b", "url": "rtsp://h/b", "enabled": False}]
    path = write_config(tmp_path, entries)
    assert [c.id for c in load_cameras(path)] == ["a"]
    assert [c.id for c in load_cameras(path, include_disabled=True)] == ["a", "b"]
    assert get_camera("b", path).enabled is False


def test_missing_and_empty_files(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_cameras(tmp_path / "nope.yaml")
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="non-empty"):
        load_cameras(empty)


def test_a_camera_password_can_come_from_the_secrets_file(tmp_path, monkeypatch):
    cfg = tmp_path / "cameras.yaml"
    cfg.write_text("cameras:\n  - id: gate\n    name: Gate\n    url: rtsp://admin:${CAM_GATE_PASSWORD}@10.0.0.5:554/Streaming/Channels/102\n")
    secrets = tmp_path / "secrets.env"
    secrets.write_text("# comment\nCAM_GATE_PASSWORD=s3cret\n")
    monkeypatch.delenv("CAM_GATE_PASSWORD", raising=False)
    (cam,) = load_cameras(cfg, secrets=secrets)
    assert cam.url == "rtsp://admin:s3cret@10.0.0.5:554/Streaming/Channels/102" and "s3cret" not in cam.safe_url
    monkeypatch.setenv("CAM_GATE_PASSWORD", "from-env")                  # the environment wins
    assert load_cameras(cfg, secrets=secrets)[0].url.startswith("rtsp://admin:from-env@")
    monkeypatch.delenv("CAM_GATE_PASSWORD")
    with pytest.raises(ConfigError, match="secrets.env"):
        load_cameras(cfg, secrets=tmp_path / "none.env")
