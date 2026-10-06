#!/usr/bin/env python3
"""Download the MediaMTX RTSP server for this computer into tools/mediamtx/.

    python scripts/get_mediamtx.py

Fetches the official release from GitHub, checks it against the release's published
SHA-256 checksums, and unpacks it. The version is pinned (see MEDIAMTX_VERSION) because
configs/mediamtx.yml is written for it.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import platform
import shutil
import stat
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401

from ppe_monitor.executables import MEDIAMTX_DIR

MEDIAMTX_VERSION = "v1.21.0"
BASE_URL = "https://github.com/bluenviron/mediamtx/releases/download"
RELEASES_PAGE = "https://github.com/bluenviron/mediamtx/releases"

# (platform.system(), platform.machine()) -> suffix used in release file names
PLATFORMS = {
    ("Darwin", "arm64"): "darwin_arm64.tar.gz",
    ("Darwin", "x86_64"): "darwin_amd64.tar.gz",
    ("Linux", "x86_64"): "linux_amd64.tar.gz",
    ("Linux", "aarch64"): "linux_arm64.tar.gz",
    ("Linux", "arm64"): "linux_arm64.tar.gz",
    ("Windows", "AMD64"): "windows_amd64.zip",
}


def asset_name(version: str, suffix: str) -> str:
    return f"mediamtx_{version}_{suffix}"


def expected_checksum(checksums_text: str, asset: str) -> str | None:
    """Find `asset` in a sha256sum-style listing ("<hash> *<file>" or "<hash>  <file>")."""
    for line in checksums_text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == asset:
            return parts[0].lower()
    return None


def extract_binary(archive: bytes, suffix: str, exe_name: str, target: Path) -> None:
    """Pull just the executable out of a release archive and make it runnable."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if suffix.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(archive)) as zf, zf.open(exe_name) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tf:
            src = tf.extractfile(tf.getmember(exe_name))
            if src is None:
                raise ValueError(f"{exe_name} in the archive is not a regular file")
            with src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def download(url: str) -> bytes:
    print(f"  downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "ppe-monitor-setup"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default=MEDIAMTX_VERSION, help=f"release tag (default {MEDIAMTX_VERSION})")
    parser.add_argument("--force", action="store_true", help="re-download even if already present")
    args = parser.parse_args()

    key = (platform.system(), platform.machine())
    suffix = PLATFORMS.get(key)
    if not suffix:
        print(f"No MediaMTX download known for {key}. Get it from {RELEASES_PAGE} "
              f"and put the executable in {MEDIAMTX_DIR}/")
        return 1

    exe_name = "mediamtx.exe" if key[0] == "Windows" else "mediamtx"
    target = MEDIAMTX_DIR / exe_name
    if target.exists() and not args.force:
        print(f"MediaMTX already present at {target} (use --force to replace it)")
        return 0

    asset = asset_name(args.version, suffix)
    try:
        archive = download(f"{BASE_URL}/{args.version}/{asset}")
        checksums = download(f"{BASE_URL}/{args.version}/checksums.sha256").decode()
    except OSError as exc:
        print(f"Download failed: {exc}\nCheck your internet connection, or download {asset} manually from "
              f"{RELEASES_PAGE}/tag/{args.version} and put the '{exe_name}' file from it into {MEDIAMTX_DIR}/")
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            print("On macOS with Python from python.org, run 'Install Certificates.command' "
                  "(in Applications > Python 3.x) once, then try again.")
        return 1

    expected = expected_checksum(checksums, asset)
    actual = hashlib.sha256(archive).hexdigest()
    if expected is None:
        print(f"{asset} is not listed in the release's checksums.sha256 - refusing to install an unverified file.")
        return 1
    if expected != actual:
        print(f"Checksum mismatch for {asset} - refusing to install it.\n  expected {expected}\n  got      {actual}")
        return 1
    print("  checksum verified")

    extract_binary(archive, suffix, exe_name, target)
    print(f"Installed MediaMTX {args.version} to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
