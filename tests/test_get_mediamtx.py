"""The download itself needs the internet, but checksum matching and unpacking don't."""

import io
import os
import tarfile
import zipfile

import get_mediamtx as gm


def make_tar_gz(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def make_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_asset_names_match_release_naming():
    assert gm.asset_name("v1.21.0", gm.PLATFORMS[("Darwin", "arm64")]) == "mediamtx_v1.21.0_darwin_arm64.tar.gz"
    assert gm.asset_name("v1.21.0", gm.PLATFORMS[("Windows", "AMD64")]) == "mediamtx_v1.21.0_windows_amd64.zip"


def test_expected_checksum_handles_binary_and_text_marks():
    listing = ("aaa111 *mediamtx_v1.21.0_linux_amd64.tar.gz\n"
               "BBB222  mediamtx_v1.21.0_darwin_arm64.tar.gz\n")
    assert gm.expected_checksum(listing, "mediamtx_v1.21.0_linux_amd64.tar.gz") == "aaa111"
    assert gm.expected_checksum(listing, "mediamtx_v1.21.0_darwin_arm64.tar.gz") == "bbb222"
    assert gm.expected_checksum(listing, "mediamtx_v1.21.0_windows_amd64.zip") is None
    # a name that merely ends with the asset name must not match
    assert gm.expected_checksum("ccc *evil_mediamtx_v1.21.0_linux_amd64.tar.gz",
                                "mediamtx_v1.21.0_linux_amd64.tar.gz") is None


def test_extract_from_tar_gz_makes_executable(tmp_path):
    archive = make_tar_gz({"mediamtx": b"#!binary", "mediamtx.yml": b"x: 1", "LICENSE": b"MIT"})
    target = tmp_path / "tools" / "mediamtx"
    gm.extract_binary(archive, "darwin_arm64.tar.gz", "mediamtx", target)
    assert target.read_bytes() == b"#!binary"
    assert os.access(target, os.X_OK)
    assert not (tmp_path / "tools" / "mediamtx.yml").exists()  # only the executable is taken


def test_extract_from_zip(tmp_path):
    archive = make_zip({"mediamtx.exe": b"MZbinary", "LICENSE": b"MIT"})
    target = tmp_path / "mediamtx.exe"
    gm.extract_binary(archive, "windows_amd64.zip", "mediamtx.exe", target)
    assert target.read_bytes() == b"MZbinary"
