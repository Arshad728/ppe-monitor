"""A stand-in for api.telegram.org on this machine, for checks and evaluations.

It answers the Bot API calls the alert worker makes (getMe, sendPhoto, sendMessage) the way
Telegram does, and records when each message arrived. Nothing leaves the machine and nobody's
phone buzzes, but the worker, its HTTP client, retries and timings are the real ones.
`outage` refuses the first N messages with HTTP 502, to exercise the retries.
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class StandInTelegram:
    def __init__(self, outage: int = 0, host: str = "127.0.0.1", port: int = 0):
        self.received: list[dict] = []
        self.refused, self.outage = 0, outage
        self._lock = threading.Lock()
        me = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self, code: int, body: bytes, kind: str = "application/json"):
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                method = self.path.rsplit("/", 1)[-1]
                if method == "getMe":           # the worker keeping its connection warm
                    self._reply(200, b'{"ok": true, "result": {"username": "stand_in_bot"}}')
                    return
                with me._lock:
                    refuse = me.outage > 0
                    if refuse:
                        me.outage -= 1
                        me.refused += 1
                if refuse:
                    self._reply(502, b"Bad Gateway", "text/plain")
                    return
                chat = re.search(rb'name="chat_id"\r\n\r\n([^\r]+)', body)
                with me._lock:
                    me.received.append({"t": time.time(), "method": method, "bytes": len(body),
                                        "photo": b'name="photo"' in body,
                                        "chat": chat.group(1).decode() if chat else None,
                                        "summary": b"more alert" in body})
                    n = len(me.received)
                self._reply(200, json.dumps({"ok": True, "result": {"message_id": n}}).encode())

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, name="stand-in-telegram", daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
