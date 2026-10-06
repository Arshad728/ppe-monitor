"""The safety officer's dashboard (Phase 5): live cameras, the event feed, counts, and the
confirm / false-alarm verdict on each event. A single page (static/), served with the API by
backend/api.py; plain HTML, CSS and JavaScript, nothing loaded from the internet."""

from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / "static"
