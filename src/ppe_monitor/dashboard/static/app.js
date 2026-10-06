"use strict";
/* The PPE monitor dashboard: plain JavaScript, no libraries, nothing from the internet.
   Everything shown comes from /api (backend/api.py). Names and notes are inserted with
   textContent, never as HTML. */

const KIND = { no_helmet: "No helmet", no_vest: "No hi-vis vest", zone_intrusion: "Restricted zone" };
const SEVERITY = { critical: ["critical", "Critical"], high: ["serious", "High"], medium: ["warning", "Medium"], low: ["", "Low"] };
const CAM_STATE = { live: ["good", "Live"], connecting: ["warning", "Connecting"], reconnecting: ["serious", "Reconnecting"],
                    stopped: ["critical", "Stopped"] };
const VERDICT = { new: "Not reviewed", confirmed: "✓ Confirmed", false_alarm: "✗ False alarm" };

const state = { hours: 24, camera: "", kind: "", status: "", items: [], total: 0, pages: 1, seen: new Set(),
                cams: {}, open: null, img: "snapshot", firstLoad: true };
const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined) e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return e;
}
const svg = (tag, attrs = {}) => {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
};

async function api(path, options) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

function query() {
  const p = new URLSearchParams();
  const since = new Date(Date.now() - state.hours * 3600e3).toISOString();
  p.set("since", since);
  if (state.camera) p.set("camera", state.camera);
  if (state.kind) p.set("kind", state.kind);
  if (state.status) p.set("status", state.status);
  return p;
}

const fmtTime = (iso) => new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
const fmtDay = (iso) => new Date(iso).toLocaleDateString([], { day: "numeric", month: "short" });
const ago = (s) => s === null || s === undefined ? "never" : s < 60 ? `${Math.round(s)} s ago` : s < 3600 ? `${Math.round(s / 60)} min ago` : `${Math.round(s / 3600)} h ago`;
const camName = (id) => (state.cams[id] && state.cams[id].name) || id;
const compact = (n) => n >= 10000 ? `${(n / 1000).toFixed(0)}K` : n >= 1000 ? `${(n / 1000).toFixed(1)}K` : String(n);

/* ---- header: are the camera service and the alert worker running? ---- */
function renderServices(o) {
  const ul = $("services");
  ul.replaceChildren();
  const svc = (name, label, extra) => {
    const s = o.services[name];
    const age = s ? s.age_s : null;
    const tone = age === null ? "critical" : age < 20 ? "good" : age < 120 ? "warning" : "critical";
    const text = age === null ? `${label}: not started` : age < 20 ? `${label}: running${extra ? " · " + extra : ""}` : `${label}: last seen ${ago(age)}`;
    ul.append(el("li", { class: "pill", title: s ? `last report ${s.last_seen}` : "" }, el("span", { class: `dot ${tone}`, "aria-hidden": "true" }), text));
  };
  svc("camera_service", "Cameras");
  const w = o.services.alert_worker;
  const ch = w && w.detail && w.detail.channels ? Object.entries(w.detail.channels).filter(([k]) => o.channels[k]) : [];
  const chText = ch.map(([k, ok]) => `${k}${ok ? "" : " (not set up)"}`).join(", ");
  svc("alert_worker", "Alerts", chText);
  ul.append(el("li", { class: "pill" }, el("span", { class: "dot good", "aria-hidden": "true" }), `Database: ${o.database}`));
}

/* ---- stat tiles ---- */
function renderStats(o) {
  const s = o.stats;
  const range = { 1: "last hour", 24: "last 24 hours", 168: "last 7 days", 720: "last 30 days" }[state.hours];
  const reviewed = s.by_status.confirmed + s.by_status.false_alarm;
  const sent = o.alerts.sent || 0;
  const tiles = [
    ["hero", "Events", compact(s.total), range],
    ["", "Not reviewed", compact(s.by_status.new), s.total ? `of ${compact(s.total)}` : ""],
    ["", "No helmet", compact(s.by_kind.no_helmet || 0), ""],
    ["", "Restricted zone", compact(s.by_kind.zone_intrusion || 0), s.by_kind.no_vest ? `no vest: ${s.by_kind.no_vest}` : ""],
    ["", "False alarms", s.false_alarm_share === null ? "–" : `${Math.round(100 * s.false_alarm_share)}%`,
     reviewed ? `of ${reviewed} reviewed` : "none reviewed yet"],
    ["", "Alerts sent", compact(sent), [o.alerts.held || o.alerts.summarised ? `${(o.alerts.held || 0) + (o.alerts.summarised || 0)} grouped` : "",
                                         o.alerts.failed ? `${o.alerts.failed} failed` : ""].filter(Boolean).join(", ")],
  ];
  $("stats").replaceChildren(...tiles.map(([cls, label, value, hint]) =>
    el("div", { class: `tile ${cls}` }, el("div", { class: "label", text: label }), el("div", { class: "value", text: value }),
       el("div", { class: "hint", text: hint || " " }))));
}

/* ---- cameras: live pictures ---- */
function renderCams(cams) {
  const box = $("cams");
  if (!cams.length) {
    box.replaceChildren(el("div", { class: "empty", text: "No cameras yet. Start the camera service: bash scripts/mac_phase5.sh run" }));
    return;
  }
  const have = new Map([...box.querySelectorAll(".cam")].map((c) => [c.dataset.id, c]));
  const out = [];
  for (const c of cams) {
    let card = have.get(c.id);
    if (!card) {
      card = el("div", { class: "cam", "data-id": c.id },
        el("div", { class: "pic" }, el("span", { class: "ph", text: "No live picture" })),
        el("div", { class: "meta" }, el("span", { class: "name" }), el("span", { class: "badge st" }), el("span", { class: "rate" })));
    }
    const [tone, label] = CAM_STATE[c.state] || ["", c.state || "unknown"];
    const stale = c.last_seen === null || (Date.now() - new Date(c.last_seen)) > 30e3;
    card.querySelector(".name").textContent = c.name || c.id;
    card.querySelector(".name").title = `${c.id}${c.detail ? " – " + c.detail : ""}`;
    card.querySelector(".st").replaceChildren(el("span", { class: `dot ${stale ? "" : tone}`, "aria-hidden": "true" }), stale ? "No report" : label);
    card.querySelector(".rate").textContent = !stale && c.fps ? `${c.fps.toFixed(1)} fps${c.lag_ms ? ` · ${Math.round(c.lag_ms)} ms` : ""}` : "";
    card.dataset.live = c.tile_age_s !== null && c.tile_age_s < 15 ? "1" : "";
    out.push(card);
  }
  box.replaceChildren(...out);
  $("cams-note").textContent = `${cams.filter((c) => c.state === "live").length} of ${cams.length} live`;
}

function refreshTiles() {
  if (document.visibilityState !== "visible") return;
  for (const card of document.querySelectorAll(".cam")) {
    const pic = card.querySelector(".pic");
    if (!card.dataset.live) {
      if (!pic.querySelector(".ph")) pic.replaceChildren(el("span", { class: "ph", text: "No live picture" }));
      continue;
    }
    const img = new Image();
    img.alt = `Live view of ${card.querySelector(".name").textContent}`;
    img.onload = () => pic.replaceChildren(img);
    img.src = `/api/cameras/${encodeURIComponent(card.dataset.id)}/live.jpg?t=${Date.now()}`;
  }
}

/* ---- charts ---- */
const tip = $("tip");
function showTip(evt, value, label) {
  tip.replaceChildren(el("strong", { text: value }), el("span", { text: label }));
  tip.hidden = false;
  const x = Math.min(evt.clientX + 12, window.innerWidth - tip.offsetWidth - 8);
  tip.style.left = `${x}px`;
  tip.style.top = `${evt.clientY - tip.offsetHeight - 10}px`;
}
const hideTip = () => { tip.hidden = true; };

function niceMax(v) {                      // a round top for the axis, with a whole-number middle tick
  if (v <= 10) return Math.max(2, Math.ceil(v / 2) * 2);
  const p = 10 ** Math.floor(Math.log10(v));
  for (const m of [1, 2, 4, 5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}

function chartTime(stats) {
  const box = $("chart-time");
  const data = stats.series;
  const W = Math.max(280, box.clientWidth), H = 190, L = 34, R = 6, T = 10, B = 26;
  const max = niceMax(Math.max(1, ...data.map((d) => d.n)));
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img",
    "aria-label": `Events per ${stats.step}: ${stats.total} in all` });
  const y = (v) => T + (H - T - B) * (1 - v / max);
  for (const t of [0, max / 2, max]) {
    s.append(svg("line", { class: t ? "gridline" : "baseline", x1: L, x2: W - R, y1: Math.round(y(t)) + 0.5, y2: Math.round(y(t)) + 0.5 }));
    const lab = svg("text", { class: "axis-text", x: L - 6, y: y(t) + 4, "text-anchor": "end" });
    lab.textContent = Number.isInteger(t) ? t.toLocaleString() : t.toFixed(1);
    s.append(lab);
  }
  const n = data.length, band = (W - L - R) / Math.max(1, n);
  const bw = Math.max(2, Math.min(24, band - 2));
  const every = Math.ceil(n / Math.max(2, Math.floor((W - L - R) / 64)));
  data.forEach((d, i) => {
    const x = L + i * band + (band - bw) / 2;
    const label = stats.step === "hour"
      ? `${fmtDay(d.t)} ${new Date(d.t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`
      : fmtDay(d.t);
    const hit = svg("rect", { class: "bar-hit", x: L + i * band, y: T, width: band, height: H - T - B, tabindex: 0,
      "aria-label": `${label}: ${d.n} event${d.n === 1 ? "" : "s"}` });
    if (d.n > 0) {
      const top = y(d.n), h = H - B - top, r = Math.min(4, h, bw / 2);
      const bar = svg("path", { class: "bar", d: `M${x},${H - B}V${top + r}q0,-${r} ${r},-${r}h${bw - 2 * r}q${r},0 ${r},${r}V${H - B}Z` });
      hit.addEventListener("pointermove", (e) => { bar.classList.add("hot"); showTip(e, `${d.n}`, label); });
      hit.addEventListener("pointerleave", () => { bar.classList.remove("hot"); hideTip(); });
      s.append(hit, bar);
    } else {
      hit.addEventListener("pointermove", (e) => showTip(e, "0", label));
      hit.addEventListener("pointerleave", hideTip);
      s.append(hit);
    }
    if (i % every === 0) {
      const t = svg("text", { class: "axis-text", x: L + i * band + band / 2, y: H - 8, "text-anchor": "middle" });
      t.textContent = stats.step === "hour" ? new Date(d.t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : fmtDay(d.t);
      s.append(t);
    }
  });
  box.replaceChildren(s);
  $("chart-note").textContent = `per ${stats.step}, ${stats.total} in all`;
}

function chartCameras(stats) {
  const box = $("chart-cam");
  const rows = Object.entries(stats.by_camera);
  if (!rows.length) { box.replaceChildren(el("div", { class: "empty", text: "No events in this range." })); return; }
  const W = Math.max(280, box.clientWidth), rowH = 26, L = Math.min(170, W * 0.38), R = 44;
  const max = Math.max(...rows.map(([, n]) => n));
  const s = svg("svg", { viewBox: `0 0 ${W} ${rows.length * rowH}`, height: rows.length * rowH, role: "img", "aria-label": "Events by camera" });
  rows.forEach(([cam, n], i) => {
    const yMid = i * rowH + rowH / 2, len = Math.max(2, (W - L - R) * n / max), h = 14, r = Math.min(4, len / 2);
    const name = svg("text", { class: "hbar-label", x: L - 8, y: yMid + 4, "text-anchor": "end" });
    const full = camName(cam);
    name.textContent = full.length > 24 ? full.slice(0, 23) + "…" : full;
    const bar = svg("path", { class: "bar", d: `M${L},${yMid - h / 2}H${L + len - r}q${r},0 ${r},${r}V${yMid + h / 2 - r}q0,${r} -${r},${r}H${L}Z` });
    const val = svg("text", { class: "hbar-value", x: L + len + 6, y: yMid + 4 });
    val.textContent = n.toLocaleString();
    const hit = svg("rect", { class: "bar-hit", x: 0, y: i * rowH, width: W, height: rowH, tabindex: 0, "aria-label": `${full}: ${n}` });
    hit.addEventListener("pointermove", (e) => { bar.classList.add("hot"); showTip(e, `${n}`, full); });
    hit.addEventListener("pointerleave", () => { bar.classList.remove("hot"); hideTip(); });
    hit.addEventListener("click", () => { $("f-camera").value = cam; state.camera = cam; refreshAll(); });
    s.append(name, hit, bar, val);
  });
  box.replaceChildren(s);
}

/* ---- the event feed ---- */
function alertText(alerts) {
  if (!alerts || !alerts.length) return ["", "–"];
  const sent = alerts.filter((a) => a.status === "sent");
  if (sent.length) {
    const lat = Math.min(...sent.map((a) => a.latency_ms || 0));
    return ["good", `${sent.map((a) => a.channel).join(", ")}${lat ? ` · ${(lat / 1000).toFixed(1)} s` : ""}`];
  }
  if (alerts.some((a) => a.status === "held" || a.status === "summarised")) return ["", "In a summary"];
  if (alerts.some((a) => a.status === "failed")) return ["critical", "Failed"];
  return ["warning", "Sending"];
}

function eventRow(e) {
  const [sevTone, sevLabel] = SEVERITY[e.severity] || ["", e.severity];
  const [alTone, alLabel] = alertText(e.alerts);
  const what = `${KIND[e.kind] || e.kind}${e.kind === "zone_intrusion" && e.zone ? ` (${e.zone})` : ""}`;
  const tr = el("tr", { tabindex: 0, "data-id": e.id, onclick: () => openEvent(e.id),
    onkeydown: (k) => { if (k.key === "Enter") openEvent(e.id); } },
    el("td", { class: "time" }, fmtTime(e.confirmed_at), el("small", { text: fmtDay(e.confirmed_at) })),
    el("td", { text: camName(e.camera) }),
    el("td", {}, el("div", { class: "what" },
      e.has_snapshot ? el("img", { class: "thumb", loading: "lazy", alt: "", src: `/api/events/${e.id}/snapshot.jpg` }) : el("span", { class: "thumb" }),
      el("div", {}, el("div", { text: what }), el("div", { class: "muted small", text: `person #${e.track_id}` })))),
    el("td", {}, el("span", { class: "badge" }, el("span", { class: `dot ${sevTone}`, "aria-hidden": "true" }), sevLabel)),
    el("td", {}, el("span", { class: "badge" }, el("span", { class: `dot ${alTone}`, "aria-hidden": "true" }), alLabel)),
    el("td", { class: `status-${e.status}`, text: VERDICT[e.status] || e.status }));
  if (!state.firstLoad && !state.seen.has(e.id)) tr.classList.add("fresh");
  return tr;
}

async function refreshEvents() {
  const p = query();
  p.set("limit", String(50 * state.pages));
  const r = await api(`/api/events?${p}`);
  state.items = r.items;
  state.total = r.total;
  const body = $("events");
  if (!r.items.length) {
    body.replaceChildren(el("tr", {}, el("td", { colspan: 6, class: "empty", text: "No events in this range." })));
  } else {
    body.replaceChildren(...r.items.map(eventRow));
  }
  r.items.forEach((e) => state.seen.add(e.id));
  state.firstLoad = false;
  $("ev-note").textContent = r.total ? `${r.items.length} of ${r.total}` : "";
  $("more").hidden = r.items.length >= r.total;
}

/* ---- one event, in the drawer ---- */
function fact(dl, k, v) { dl.append(el("dt", { text: k }), el("dd", {}, v)); }

async function openEvent(id) {
  let e;
  try { e = await api(`/api/events/${id}`); } catch (err) { return; }
  state.open = e;
  history.replaceState(null, "", `#event=${id}`);
  $("d-title").textContent = `${KIND[e.kind] || e.kind} · ${e.camera_name}`;
  showImage();
  const dl = $("d-facts");
  dl.replaceChildren();
  const [sevTone, sevLabel] = SEVERITY[e.severity] || ["", e.severity];
  fact(dl, "When", `${fmtDay(e.confirmed_at)}, ${fmtTime(e.confirmed_at)}`);
  const lasted = e.started_at ? Math.round((new Date(e.confirmed_at) - new Date(e.started_at)) / 1000) : null;
  fact(dl, "Person", `#${e.track_id}${lasted !== null ? `, for ${lasted} s before the alert` : ""}`);
  fact(dl, "Camera", `${e.camera_name} (${e.camera})`);
  fact(dl, "Rule", `${e.rule}${e.zone ? `, zone ${e.zone}` : ""}`);
  fact(dl, "Severity", el("span", { class: "badge" }, el("span", { class: `dot ${sevTone}`, "aria-hidden": "true" }), sevLabel));
  fact(dl, "Verdict", `${VERDICT[e.status]}${e.status_at ? ` (${fmtDay(e.status_at)} ${fmtTime(e.status_at)})` : ""}`);
  $("d-note").value = e.note || "";
  for (const b of document.querySelectorAll(".d-buttons .btn")) {
    b.classList.toggle("current", b.dataset.status === e.status);
    if (b.dataset.status === "new") b.hidden = e.status === "new";     // "Undo" only once there is a verdict
  }
  const ul = $("d-alerts");
  ul.replaceChildren();
  if (!e.alerts.length) ul.append(el("li", { class: "muted", text: "No alert for this event (its severity is below every channel's minimum)." }));
  for (const a of e.alerts) {
    const [tone, text] = a.status === "sent"
      ? ["good", `${a.is_summary ? `Summary of ${a.summary_count} held alerts, sent` : "Sent"}${a.latency_ms && !a.is_summary ? `, ${(a.latency_ms / 1000).toFixed(1)} s after the event` : ""}`]
      : a.status === "failed" ? ["critical", `Failed after ${a.attempts} attempt(s): ${a.last_error || ""}`]
      : a.status === "held" ? ["", "Held by the rate limit; will go out in a summary"]
      : a.status === "summarised" ? ["", "Sent as part of a summary"]
      : ["warning", `Waiting${a.last_error ? ` (last error: ${a.last_error})` : ""}`];
    ul.append(el("li", {}, el("span", { class: `dot ${tone}`, "aria-hidden": "true" }), el("strong", { text: a.channel }), el("span", { text })));
  }
  $("drawer").classList.add("open");
  $("drawer").setAttribute("aria-hidden", "false");
}

function showImage() {
  const e = state.open;
  if (!e) return;
  const which = state.img === "frame" && e.has_frame ? "frame" : "snapshot";
  const img = $("d-img");
  if ((which === "snapshot" && !e.has_snapshot) || (which === "frame" && !e.has_frame)) {
    img.removeAttribute("src");
    img.alt = "Image no longer kept";
  } else {
    img.alt = which === "frame" ? "The clean camera frame" : "Evidence image of the event";
    img.src = `/api/events/${e.id}/${which}.jpg`;
  }
  for (const b of document.querySelectorAll(".d-imgtabs button")) b.classList.toggle("on", b.dataset.img === which);
}

function closeDrawer() {
  state.open = null;
  $("drawer").classList.remove("open");
  $("drawer").setAttribute("aria-hidden", "true");
  history.replaceState(null, "", location.pathname);
}

async function setVerdict(status) {
  if (!state.open) return;
  await api(`/api/events/${state.open.id}`, { method: "PATCH", body: JSON.stringify({ status, note: $("d-note").value }) });
  await openEvent(state.open.id);
  refreshAll();
}

/* ---- refresh ---- */
async function refreshOverview() {
  const p = query();
  p.delete("status"); p.delete("kind");
  p.set("hours", String(state.hours));
  const o = await api(`/api/overview?${p}`);
  state.cams = Object.fromEntries(o.cameras.map((c) => [c.id, c]));
  const sel = $("f-camera");
  const want = [""].concat(o.cameras.map((c) => c.id));
  if ([...sel.options].map((x) => x.value).join() !== want.join()) {
    sel.replaceChildren(el("option", { value: "", text: "All" }), ...o.cameras.map((c) => el("option", { value: c.id, text: c.name || c.id })));
    sel.value = state.camera;
  }
  renderServices(o);
  renderStats(o);
  renderCams(o.cameras);
  chartTime(o.stats);
  chartCameras(o.stats);
}

let busy = false;
async function refreshAll() {
  if (busy) return;
  busy = true;
  try { await Promise.all([refreshOverview(), refreshEvents()]); }
  catch (err) {
    $("services").replaceChildren(el("li", { class: "pill" }, el("span", { class: "dot critical", "aria-hidden": "true" }), "Dashboard server not reachable"));
  } finally { busy = false; }
}

/* ---- wiring ---- */
for (const b of document.querySelectorAll("#range button")) {
  b.addEventListener("click", () => {
    state.hours = Number(b.dataset.hours);
    document.querySelectorAll("#range button").forEach((x) => { x.classList.toggle("on", x === b); x.setAttribute("aria-checked", x === b); });
    state.pages = 1;
    refreshAll();
  });
}
for (const [id, key] of [["f-camera", "camera"], ["f-kind", "kind"], ["f-status", "status"]]) {
  $(id).addEventListener("change", () => { state[key] = $(id).value; state.pages = 1; refreshAll(); });
}
$("more").addEventListener("click", () => { state.pages += 1; refreshEvents(); });
$("d-close").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (k) => { if (k.key === "Escape") closeDrawer(); });
for (const b of document.querySelectorAll(".d-buttons .btn")) b.addEventListener("click", () => setVerdict(b.dataset.status));
for (const b of document.querySelectorAll(".d-imgtabs button")) b.addEventListener("click", () => { state.img = b.dataset.img; showImage(); });
$("d-note").addEventListener("change", async () => {
  if (state.open) await api(`/api/events/${state.open.id}`, { method: "PATCH", body: JSON.stringify({ status: state.open.status, note: $("d-note").value }) });
});
window.addEventListener("resize", () => { clearTimeout(window._rs); window._rs = setTimeout(refreshOverview, 200); });

refreshAll().then(() => {
  refreshTiles();
  const m = location.hash.match(/event=([0-9a-f-]{36})/);
  if (m) openEvent(m[1]);
});
setInterval(refreshAll, 3000);
setInterval(refreshTiles, 1000);
