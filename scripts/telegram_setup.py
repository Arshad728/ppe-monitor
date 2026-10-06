#!/usr/bin/env python3
"""Connect the alerts to your Telegram (Phase 5), and send a test alert to your phone.

Before running it:
  1. In Telegram, open @BotFather, send /newbot, and choose a name and a username for the bot.
     BotFather replies with a token (like 123456789:AA...). Keep it private.
  2. Open your new bot in Telegram and press Start (or send it any message), so it may write to you.

Then:  bash scripts/mac_phase5.sh telegram      (or: python scripts/telegram_setup.py)

It asks for the token (typing is hidden), checks it with Telegram, finds the chat you just
started, saves both in configs/secrets.env (never in git), and sends a test alert with a picture.

    python scripts/telegram_setup.py --test      # only send the test alert again
    python scripts/telegram_setup.py --speed     # how fast this network reaches Telegram (IPv4 and IPv6)
"""

import argparse
import getpass
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import _bootstrap  # noqa: F401

import cv2
import numpy as np

from ppe_monitor.backend.channels import Message, SendError, TelegramChannel, http_client, shrink
from ppe_monitor.backend.settings import SECRETS_FILE, ServerSettings, write_env_value
from ppe_monitor.backend.localpg import EXAMPLE
from ppe_monitor.config import PROJECT_ROOT


def test_picture(big: bool = False) -> bytes:
    img = np.full((360, 640, 3), (40, 36, 32), np.uint8)
    if big:                                           # about the size and detail of a real snapshot
        small = np.random.default_rng(0).integers(0, 255, (54, 96, 3), dtype=np.uint8)
        img = cv2.GaussianBlur(cv2.resize(small, (1920, 1080), interpolation=cv2.INTER_CUBIC), (0, 0), 3)
    cv2.ellipse(img, (320, 190), (110, 95), 0, 180, 360, (0, 161, 237), -1)      # a helmet
    cv2.rectangle(img, (190, 186), (450, 206), (0, 133, 201), -1)
    cv2.putText(img, "PPE monitor - test alert", (150, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(img, "If you can read this on your phone, alerts will reach you.", (70, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return cv2.imencode(".jpg", img)[1].tobytes()


def connect_time(family: int, host: str, port: int = 443, timeout: float = 5.0):
    """(address, seconds) of a plain TCP connection over IPv4 or IPv6, or (address or None, error)."""
    try:
        infos = socket.getaddrinfo(host, port, family, socket.SOCK_STREAM)
    except socket.gaierror:
        return None, "no address"
    addr = infos[0][4]
    s = socket.socket(family, socket.SOCK_STREAM)
    s.settimeout(timeout)
    t0 = time.perf_counter()
    try:
        s.connect(addr)
        return addr[0], time.perf_counter() - t0
    except OSError as exc:
        return addr[0], f"{type(exc).__name__} after {time.perf_counter() - t0:.1f} s"
    finally:
        s.close()


def speed(token: str, chats: list[str], api_base: str, storage: Path) -> int:
    url = urlparse(api_base)
    host, port = url.hostname or "api.telegram.org", url.port or (443 if url.scheme == "https" else 80)
    print(f"How fast this Mac reaches {host}:")
    t0 = time.perf_counter()
    try:
        names = socket.getaddrinfo(host, 443, 0, socket.SOCK_STREAM)
    except socket.gaierror as exc:
        print(f"  can't look up {host}: {exc}")
        return 1
    fams = {socket.AF_INET: "IPv4", socket.AF_INET6: "IPv6"}
    print(f"  name lookup: {time.perf_counter() - t0:.2f} s, addresses: "
          + ", ".join(sorted({fams.get(i[0], '?') for i in names})) + f" (the system tries {fams.get(names[0][0], '?')} first)")
    results = {}
    for fam, label in fams.items():
        addr, r = connect_time(fam, host, port)
        results[label] = r
        print(f"  plain connection over {label}: " + (f"{r * 1000:.0f} ms ({addr})" if isinstance(r, float) else f"fails: {r}"))
    for ip in ("any", "ipv4"):
        bot = TelegramChannel(token, chats, api_base=api_base, client=http_client(ip))
        try:
            cold = bot.ping()
            warm = bot.ping()
            print(f"  Telegram, {'system choice' if ip == 'any' else 'IPv4 only (the setting in use)'}: "
                  f"first call {cold:.2f} s, next call {warm:.2f} s")
        except SendError as exc:
            print(f"  Telegram, {ip}: fails: {exc}")
    if chats:
        bot = TelegramChannel(token, chats[:1], api_base=api_base)
        bot.ping()
        recent = sorted((p for p in storage.glob("*/*.jpg") if not p.stem.endswith("_frame")),
                        key=lambda p: p.stat().st_mtime)
        photo = shrink(recent[-1].read_bytes() if recent else test_picture(big=True))
        t0 = time.perf_counter()
        try:
            bot.send(Message("\u23F1 Speed test", ["A picture the size of an alert's."], photo), [])
            print(f"  sending a {len(photo) // 1024} KB picture: {time.perf_counter() - t0:.2f} s")
        except SendError as exc:
            print(f"  sending a picture fails: {exc}")
    ipv6 = results.get("IPv6")
    if names and names[0][0] == socket.AF_INET6 and not isinstance(ipv6, float):
        print("\nIPv6 is announced but doesn't work here: every new connection would first wait for IPv6 to "
              "time out. Alerts use IPv4 only (ip: ipv4 in configs/server.yaml), which avoids that.")
    return 0


def save(key: str, value: str) -> None:
    if not SECRETS_FILE.is_file():
        SECRETS_FILE.write_text(EXAMPLE.read_text(encoding="utf-8") if EXAMPLE.is_file() else "", encoding="utf-8")
    write_env_value(SECRETS_FILE, key, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="store_true", help="only send the test alert with what is saved")
    parser.add_argument("--new-token", action="store_true", help="ask for the token again even if one is saved")
    parser.add_argument("--speed", action="store_true", help="measure how fast this network reaches Telegram")
    args = parser.parse_args()
    s = ServerSettings.load()
    token = s.secrets.get("TELEGRAM_BOT_TOKEN")
    api_base = s.secrets.get("TELEGRAM_API_BASE", "https://api.telegram.org")

    if not args.test and (args.new_token or not token):
        print("Paste the token @BotFather gave you, then press Enter (it won't show while you paste):")
        token = getpass.getpass("  token: ").strip()
        if ":" not in token:
            print("That doesn't look like a bot token (it has the form 123456789:AA...).")
            return 1
    if not token:
        print("No TELEGRAM_BOT_TOKEN saved yet. Run without --test.")
        return 1
    if args.speed:
        return speed(token, s.secrets.list("TELEGRAM_CHAT_ID"), api_base, s.storage_dir)
    bot = TelegramChannel(token, [], api_base=api_base)
    try:
        me = bot.me()
    except SendError as exc:
        print(f"Telegram didn't accept the token: {exc}")
        return 1
    print(f"Token works: the bot is @{me.get('username')} ({me.get('first_name')}).")
    if not args.test:
        save("TELEGRAM_BOT_TOKEN", token)

    chats = s.secrets.list("TELEGRAM_CHAT_ID") if args.test else []
    tries = 0
    while not chats:
        found = bot.updates()
        if len(found) == 1:
            chats = [found[0]["chat_id"]]
            print(f"Found your chat: {found[0]['name'] or found[0]['chat_id']} ({found[0]['type']}).")
        elif found:
            print("Several chats have written to the bot:")
            for i, c in enumerate(found, 1):
                print(f"  {i}. {c['name'] or '?'} ({c['type']}, id {c['chat_id']})  \"{c['text']}\"")
            pick = input("Which should get the alerts? Numbers separated by commas (e.g. 1 or 1,3): ")
            try:
                chats = [found[int(x) - 1]["chat_id"] for x in pick.replace(" ", "").split(",") if x]
            except (ValueError, IndexError):
                print("Didn't understand that; try again.")
        else:
            tries += 1
            if tries > 3:
                print("Still no message to the bot. Run this again after pressing Start in the bot's chat.")
                return 1
            input(f"No one has written to @{me.get('username')} yet. In Telegram, open it, press Start "
                  "(or send 'hi'), then press Enter here... ")
    if not args.test:
        save("TELEGRAM_CHAT_ID", ",".join(chats))
        print(f"Saved the token and chat id in {SECRETS_FILE.relative_to(PROJECT_ROOT)} (not in git).")

    bot.chat_ids = chats
    t0 = time.perf_counter()
    msg = Message("✅ PPE monitor is connected",
                  ["This is a test alert. Real ones look like this, with the evidence picture:",
                   "\U0001F7E0 HIGH · No helmet · Main gate", "14:05:12: person #3 without it for 3 s"],
                  test_picture())
    try:
        bot.send(msg, [])
    except SendError as exc:
        print(f"The test alert didn't go through: {exc}")
        return 1
    print(f"Test alert sent in {time.perf_counter() - t0:.1f} s: check your phone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
