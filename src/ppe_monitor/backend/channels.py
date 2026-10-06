"""Where alerts go: Telegram (first, as the book suggests: easiest to set up) and email.

Each channel sends one message, with the evidence image, to all its recipients. A failure says
whether trying again could help (network trouble, the service busy) or not (a wrong token or
chat id), so the worker retries only what is worth retrying.
WhatsApp is left out: its business API needs an approved account, which the book puts last.
"""

from __future__ import annotations

import html
import smtplib
import ssl
import time
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import cv2
import httpx
import numpy as np

from .settings import Secrets

KIND_TEXT = {"no_helmet": "No helmet", "no_vest": "No hi-vis vest", "zone_intrusion": "Restricted zone"}
SEVERITY_MARK = {"critical": "\U0001F534 CRITICAL", "high": "\U0001F7E0 HIGH", "medium": "\U0001F7E1 MEDIUM",
                 "low": "⚪ LOW"}


class SendError(Exception):
    def __init__(self, message: str, *, permanent: bool = False, retry_after: float | None = None):
        super().__init__(message)
        self.permanent, self.retry_after = permanent, retry_after


@dataclass
class Message:
    title: str             # one line
    lines: list[str]       # details, one per line
    photo: bytes | None = None
    link: str = ""
    camera: str = ""       # which camera it is about (not shown; for routing and logs)
    event_id: str = ""

    def text(self) -> str:
        return "\n".join([self.title, *self.lines] + ([self.link] if self.link else []))

    def telegram_html(self) -> str:
        body = [f"<b>{html.escape(self.title)}</b>", *(html.escape(x) for x in self.lines)]
        if self.link:
            body.append(f'<a href="{html.escape(self.link, quote=True)}">Open on the dashboard</a>')
        return "\n".join(body)[:1000]        # a photo caption holds at most 1024 characters


def shrink(jpeg: bytes | None, longest: int = 1280, quality: int = 85) -> bytes | None:
    """A smaller copy of an evidence image for sending: Telegram shows at most 1280 px anyway, and a
    smaller upload arrives sooner on a slow connection."""
    if not jpeg:
        return jpeg
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jpeg
    h, w = img.shape[:2]
    if max(h, w) > longest:
        s = longest / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok and len(buf) < len(jpeg) else jpeg


def local(dt: datetime | None) -> str:
    return dt.astimezone().strftime("%H:%M:%S, %d %b") if dt else "?"


def event_message(a: dict, dashboard_url: str = "") -> Message:
    """One alert (a row from EventStore.claim_alerts) -> the message people receive."""
    kind = KIND_TEXT.get(a["kind"], a["kind"])
    zone = f" ({a['zone']})" if a.get("zone") and a["kind"] == "zone_intrusion" else ""
    title = f"{SEVERITY_MARK.get(a['severity'], a['severity'])} · {kind}{zone} · {a['camera_name']}"
    lasted = ""
    if a.get("started_at") and a.get("confirmed_at"):
        lasted = f" for {(a['confirmed_at'] - a['started_at']).total_seconds():.0f} s"
    lines = [f"{local(a.get('confirmed_at'))}: person #{a.get('track_id')}"
             + (" in the zone" if a["kind"] == "zone_intrusion" else " without it") + lasted,
             f"Camera {a['camera']}, rule {a.get('rule') or a['kind']}"]
    photo = shrink(a["snapshot"].read_bytes()) if a.get("snapshot") and Path(a["snapshot"]).is_file() else None
    link = f"{dashboard_url}/#event={a['event_id']}" if dashboard_url else ""
    return Message(title, lines, photo, link, camera=a["camera"], event_id=a["event_id"])


def summary_message(g: dict, minutes: float, dashboard_url: str = "") -> Message:
    n = len(g["ids"])
    kinds = ", ".join(f"{KIND_TEXT.get(k, k).lower()} ×{c}" for k, c in g["kinds"].most_common())
    title = f"\U0001F4CB {n} more alert{'s' if n > 1 else ''} on {g['camera_name']}"
    lines = [f"{local(g.get('first'))} to {local(g.get('last'))}: {kinds}",
             f"Held back to at most a few alerts per camera every {minutes:g} min. The newest is shown; "
             "all of them are on the dashboard."]
    photo = shrink(g["snapshot"].read_bytes()) if g.get("snapshot") and Path(g["snapshot"]).is_file() else None
    link = f"{dashboard_url}/#event={g['newest_event']}" if dashboard_url else ""
    return Message(title, lines, photo, link, camera=g["camera"])


def http_client(ip: str = "ipv4", connect: float = 5.0, read: float = 30.0) -> httpx.Client:
    """One client, kept for the worker's lifetime, so its connection to the service stays open
    between alerts (no new TLS handshake each time).

    ip="ipv4" connects over IPv4 only. Where a network announces IPv6 but doesn't carry it, trying
    IPv6 first costs a whole connect timeout before IPv4 is tried, on every new connection: that
    is the likely reason the first alerts on the Mac arrived after ~19 s instead of ~1 s. "any" lets
    the system choose."""
    transport = httpx.HTTPTransport(local_address="0.0.0.0", retries=1) if ip == "ipv4" else httpx.HTTPTransport(retries=1)
    limits = httpx.Limits(max_connections=8, max_keepalive_connections=4, keepalive_expiry=300)
    return httpx.Client(transport=transport, timeout=httpx.Timeout(read, connect=connect), limits=limits)


class TelegramChannel:
    """The Telegram Bot API: sendPhoto (or sendMessage without a picture) to each chat id."""
    name = "telegram"

    def __init__(self, token: str, chat_ids: list[str], *, api_base: str = "https://api.telegram.org",
                 client: httpx.Client | None = None, ip: str = "ipv4"):
        self.token, self.chat_ids = token, chat_ids
        self.api_base = api_base.rstrip("/")
        self.client = client or http_client(ip)

    @classmethod
    def from_secrets(cls, secrets: Secrets, ip: str = "ipv4") -> "TelegramChannel":
        return cls(secrets.get("TELEGRAM_BOT_TOKEN"), secrets.list("TELEGRAM_CHAT_ID"),
                   api_base=secrets.get("TELEGRAM_API_BASE", "https://api.telegram.org"), ip=ip)

    def ping(self) -> float:
        """Ask Telegram who the bot is; returns the round trip in seconds. Keeps the connection warm."""
        t0 = time.perf_counter()
        self.call("getMe", {})
        return time.perf_counter() - t0

    def missing(self) -> str:
        if not self.token:
            return "TELEGRAM_BOT_TOKEN is not set (configs/secrets.env): run bash scripts/mac_phase5.sh telegram"
        if not self.chat_ids:
            return "TELEGRAM_CHAT_ID is not set (configs/secrets.env): run bash scripts/mac_phase5.sh telegram"
        return ""

    def call(self, method: str, data: dict, files: dict | None = None) -> dict:
        url = f"{self.api_base}/bot{self.token}/{method}"
        try:
            r = self.client.post(url, data=data, files=files)
        except httpx.HTTPError as exc:
            raise SendError(f"telegram: {type(exc).__name__}: {exc}") from exc
        try:
            body = r.json()
        except ValueError:
            body = {"ok": False, "description": r.text[:200]}
        if r.status_code == 200 and body.get("ok"):
            return body
        desc = body.get("description") or f"HTTP {r.status_code}"
        if r.status_code == 429:
            raise SendError(f"telegram: {desc}", retry_after=float((body.get("parameters") or {}).get("retry_after", 5)))
        # 400 (e.g. chat not found), 401 (bad token), 403 (bot blocked), 404: retrying won't help
        raise SendError(f"telegram: {desc}", permanent=r.status_code in (400, 401, 403, 404))

    def send(self, msg: Message, done: list[str]) -> list[str]:
        """Send to every chat id not in `done`; returns the updated `done`. Raises SendError on the
        first failure (the chats reached before it stay in `done`, so a retry skips them)."""
        done = list(done)
        for chat in self.chat_ids:
            if chat in done:
                continue
            try:
                if msg.photo:
                    self.call("sendPhoto", {"chat_id": chat, "caption": msg.telegram_html(), "parse_mode": "HTML"},
                              {"photo": ("snapshot.jpg", msg.photo, "image/jpeg")})
                else:
                    self.call("sendMessage", {"chat_id": chat, "text": msg.telegram_html(), "parse_mode": "HTML",
                                              "disable_web_page_preview": "true"})
            except SendError as exc:
                exc.done = done               # the chats already reached: a retry skips them
                raise
            done.append(chat)
        return done

    def updates(self) -> list[dict]:
        """Recent messages to the bot (to find a chat id): [{chat_id, name, type, text}]."""
        body = self.call("getUpdates", {"timeout": 0})
        out, seen = [], set()
        for u in body.get("result", []):
            m = u.get("message") or u.get("channel_post") or u.get("my_chat_member") or {}
            chat = m.get("chat") or {}
            if chat.get("id") is None or chat["id"] in seen:
                continue
            seen.add(chat["id"])
            name = chat.get("title") or " ".join(x for x in (chat.get("first_name"), chat.get("last_name")) if x)
            out.append({"chat_id": str(chat["id"]), "name": name or chat.get("username", ""), "type": chat.get("type"),
                        "text": (m.get("text") or "")[:40]})
        return out

    def me(self) -> dict:
        return self.call("getMe", {}).get("result", {})


class EmailChannel:
    """SMTP with STARTTLS (port 587) or SSL (port 465); one message to all recipients."""
    name = "email"

    def __init__(self, host: str, port: int, user: str, password: str, sender: str, to: list[str], timeout: float = 20):
        self.host, self.port, self.user, self.password = host, port, user, password
        self.sender, self.to, self.timeout = sender or user, to, timeout

    @classmethod
    def from_secrets(cls, secrets: Secrets) -> "EmailChannel":
        return cls(secrets.get("SMTP_HOST"), int(secrets.get("SMTP_PORT", "587") or 587), secrets.get("SMTP_USER"),
                   secrets.get("SMTP_PASSWORD"), secrets.get("EMAIL_FROM"), secrets.list("EMAIL_TO"))

    def missing(self) -> str:
        need = [k for k, v in (("SMTP_HOST", self.host), ("EMAIL_TO", self.to), ("EMAIL_FROM or SMTP_USER", self.sender)) if not v]
        return f"{', '.join(need)} not set (configs/secrets.env)" if need else ""

    def send(self, msg: Message, done: list[str]) -> list[str]:
        if "all" in done:
            return done
        m = EmailMessage()
        m["Subject"], m["From"], m["To"] = f"[PPE] {msg.title}", self.sender, ", ".join(self.to)
        m.set_content(msg.text())
        if msg.photo:
            m.add_attachment(msg.photo, maintype="image", subtype="jpeg", filename="snapshot.jpg")
        try:
            if self.port == 465:
                with smtplib.SMTP_SSL(self.host, self.port, timeout=self.timeout, context=ssl.create_default_context()) as s:
                    self._login_send(s, m)
            else:
                with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as s:
                    s.ehlo()
                    if s.has_extn("starttls"):
                        s.starttls(context=ssl.create_default_context())
                        s.ehlo()
                    self._login_send(s, m)
        except smtplib.SMTPAuthenticationError as exc:
            raise SendError(f"email: login refused ({exc.smtp_code})", permanent=True) from exc
        except smtplib.SMTPRecipientsRefused as exc:
            raise SendError(f"email: recipients refused: {', '.join(exc.recipients)}", permanent=True) from exc
        except (smtplib.SMTPException, OSError) as exc:
            raise SendError(f"email: {type(exc).__name__}: {exc}") from exc
        return ["all"]

    def _login_send(self, s, m) -> None:
        if self.user and self.password:
            s.login(self.user, self.password)
        s.send_message(m)


def make_channels(settings) -> dict:
    tg = settings.channels.get("telegram")
    return {"telegram": TelegramChannel.from_secrets(settings.secrets, ip=getattr(tg, "ip", "ipv4")),
            "email": EmailChannel.from_secrets(settings.secrets)}
