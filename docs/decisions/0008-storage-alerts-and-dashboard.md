# 0008 — Storing events, sending alerts, and the dashboard

**Status:** accepted, 26 Sep 2026 (every check passed on the Mac; alerts reach the phone 3–4 s after the event) · **Book:** Chapters 13, 14 and 22

## Context

- **The book's Phase 5** (Chapter 22): a PostgreSQL schema for cameras, zones, rules and
  violations; an annotated snapshot with each event and a retention policy; alert workers on a
  queue (Telegram first, then email, WhatsApp last) with retries and a per-camera rate limit;
  FastAPI and a minimal dashboard with a live feed, filters, snapshots, counts by hour and
  camera, and a "mark as false alarm" button.
- **Done when:** a violation on a test stream reaches your phone, with its snapshot, within a few
  seconds, and shows up correctly on the dashboard.
- **The pitfalls it names:** calling Telegram or email from the loop that processes video, and
  skipping the false-alarm button.
- **This machine:** a MacBook Air with no Docker and no Homebrew; the project's Python lives in a
  conda environment. Docker Compose comes in Phase 6.

## Decision

1. **PostgreSQL, the project's own, on this Mac only** (`backend/localpg.py`, `scripts/db.py`).
   - It comes from conda-forge into the project's environment: nothing is installed system-wide.
   - Its files are in `data/postgres`. It listens on 127.0.0.1:5433, so it can't clash with
     another PostgreSQL on 5432.
   - The password is made up at `init` and kept in `configs/secrets.env`, which is not in git and
     only its owner can read (mode 600). Tokens and passwords never go in `configs/server.yaml`.
   - **SQLite works too**, from the same code (SQLAlchemy Core): `DATABASE_URL=sqlite:///...` in
     `configs/secrets.env`. It is fine for one machine, and it is what the tests use when no
     PostgreSQL is given.
   - Times are stored in UTC; the dashboard shows local time.
2. **The tables** (`backend/db.py`):

   | Table | Holds |
   |---|---|
   | `events` | every confirmed violation: kind, rule, severity, camera, zone, person, when it began and was confirmed, the person's box, the image paths, and the verdict (new / confirmed / false alarm, with a note) |
   | `alerts` | the alert queue: one row per event and channel, with its attempts, errors and delivery time |
   | `cameras`, `camera_log` | each camera's name and live state, and every drop-out and return |
   | `rules`, `zones` | the rules and zones in force, copied from the config files when the cameras start |
   | `runs`, `services` | each start of the camera service; heartbeats of the camera service and the alert worker |

   There is no migration tool yet: the schema is created as a whole (`schema_info` records
   version 1). Phase 6 can add Alembic when the schema first has to change under real data.
3. **Evidence images are files; the database holds their paths.**
   - Each event keeps the annotated snapshot and the clean frame, in `data/events/<day>/`. The
     clean frame is what a person labels when a false alarm is used for retraining.
   - Images older than 30 days are deleted; the event rows stay (`storage.keep_days`).
   - **Images of events marked "false alarm" are kept**: they are the hard examples for Phase 6.
4. **The queue is the `alerts` table** (the "transactional outbox" pattern).
   - An event and its alerts are written in one transaction, so an event can't be stored without
     its alerts, or the reverse.
   - A separate process, `scripts/alert_worker.py`, takes due rows (`FOR UPDATE SKIP LOCKED` on
     PostgreSQL, so two workers never take the same row), sends them, and records how it went.
   - **Why not Redis or Celery:** that is one more service to install, start and watch, and its
     queue lives in memory. A table survives restarts, can be queried, and the dashboard can show
     each alert's fate beside its event.
5. **The video loop never waits for the database or the network.**
   - The camera service hands events to a queue in memory (`backend/sink.py`). A writer thread
     draws the evidence image, saves it, and stores the event.
   - If the database is down, the writer retries. Its queue holds 1,000 items; past that, events
     are dropped from the database but are still in the run's `events.jsonl`.
   - A retry reuses the event's id, so an event can't be stored twice.
6. **Telegram first, then email** (`backend/channels.py`).
   - **Telegram:** the Bot API's `sendPhoto`, with the evidence image and a short caption. Each
     chat reached is recorded, so a retry after a partial failure doesn't message anyone twice.
   - **Email:** SMTP with STARTTLS or SSL, and the image attached.
   - **Retries:** wait 2, 4, 8 ... s, up to 2 min, 6 attempts in all. A wrong token or chat id
     fails at once (retrying can't help); Telegram's "too many requests" is waited out as asked.
   - **WhatsApp is not built:** its business API needs an approved account, which is why the book
     puts it last.
   - **IPv4 for Telegram** (`ip: ipv4`, a 5 s connect limit, one connection kept open). The Mac's
     network announces IPv6 but doesn't carry it, so every new connection first waited for IPv6
     to time out. The first real alerts took 19–23 s; with IPv4 they take 3–4 s
     (`mac_phase5.sh telegram --speed` measures it).
   - **Privacy:** with `photo: true` (the default) the evidence image, which shows workers, is also
     stored on Telegram's or the mail provider's servers. `photo: false` sends text only, and the
     image stays on the dashboard.
7. **Rate limit and severity, per channel** (`backend/worker.py`, `configs/server.yaml`).
   - **At most 5 alerts per camera and channel in any 10 minutes.** Past that, alerts are held,
     and go out together as one summary once the camera is back under its limit. A busy camera
     can't flood a phone, and nothing is lost.
   - **Each channel has a minimum severity:** Telegram medium and above; email high and above
     (and off by default).
8. **The dashboard: FastAPI and one static page** (`backend/api.py`, `dashboard/static/`).
   - Plain HTML, CSS and JavaScript. There are no build tools, and nothing loads from the
     internet, so it works on a site without internet access.
   - **Refreshing:** the page asks for new data every 3 s instead of keeping a live connection,
     which is simpler and plenty at this scale. The camera tiles are JPEG files that the camera
     service writes about once a second.
   - **Access:** it listens on 127.0.0.1 by default. It refuses any other address unless
     `DASHBOARD_PASSWORD` is set; with it, every page and call needs the password (HTTP basic
     authentication).
9. **The false-alarm button** records the verdict and a note, and keeps the images. The event
   list, the counts and the "false alarms" share all follow it.

## Consequences

- **Four programs run together:** PostgreSQL, the camera service (`run_cameras.py --db --live`),
  the alert worker and the dashboard. `bash scripts/mac_phase5.sh run` starts and stops all of
  them; Phase 6 puts them in Docker Compose.
- **The worker checks the queue every 0.5 s,** so it adds at most half a second before an alert
  is sent. PostgreSQL's LISTEN/NOTIFY could remove that; it isn't needed for "within a few
  seconds".
- **Event ids from the camera service** (`<camera>-<rule>-<track>-<ms>`, unique within a run
  only, a Phase 4 finding) are kept as `run_event_id`. The database gives each event a UUID.
- **Looping test clips** raise a new event every loop. That is how the check exercises the rate
  limit. With real cameras, the events engine's cooldown and the rate limit keep alerts sparse.
