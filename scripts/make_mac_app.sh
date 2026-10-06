#!/usr/bin/env bash
# Make "PPE Monitor", an app you double-click instead of typing commands (macOS only).
#
#   bash ~/Documents/ppe-monitor/scripts/make_mac_app.sh            # the app, in ~/Applications and on the Desktop
#   bash ~/Documents/ppe-monitor/scripts/make_mac_app.sh --login    # ... and start it when you log in
#   bash ~/Documents/ppe-monitor/scripts/make_mac_app.sh --no-login # stop starting it at log-in
#   bash ~/Documents/ppe-monitor/scripts/make_mac_app.sh --remove   # remove the app (the project is untouched)
#   (the same: bash scripts/ppe.sh app [--login | --no-login | --remove])
#
# Double-clicking the app:
#   - if the monitor isn't running, opens a Terminal window running `ppe.sh start --keep-running`:
#     every camera, the alerts and the dashboard, started again if any part stops, and the Mac kept
#     awake while it runs. The dashboard opens in your browser. Close the window (or Ctrl+C in it)
#     to stop the monitor.
#   - if it is running, just opens the dashboard.
# The first time, macOS asks whether "PPE Monitor" may control Terminal: answer OK. It needs that to
# open the window. With --login it also asks about "System Events", which keeps the login list.
# Make the app again after moving the project folder or changing the dashboard's port.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
NAME="PPE Monitor"
APPDIR="$HOME/Applications"
APP="$APPDIR/$NAME.app"
DESK="$HOME/Desktop/$NAME.app"

[ "$(uname)" = "Darwin" ] || { echo "This makes a macOS app; run it on the Mac."; exit 1; }
for tool in osacompile osascript; do
  command -v "$tool" >/dev/null 2>&1 || { echo "$tool is missing (it comes with macOS)."; exit 1; }
done

login_item() {   # add | remove
  if [ "$1" = "add" ]; then
    osascript -e "tell application \"System Events\" to if not (exists login item \"$NAME\") then make login item at end with properties {path:\"$APP\", hidden:false}" \
      && echo "  It will start when you log in (System Settings > General > Login Items lists it)." \
      || echo "  Could not add it to the login items. Add it by hand: System Settings > General > Login Items > +, then $APP"
  else
    osascript -e "tell application \"System Events\" to if (exists login item \"$NAME\") then delete login item \"$NAME\"" >/dev/null 2>&1
    echo "  It no longer starts when you log in."
  fi
}

case "${1:-}" in
  --remove)
    login_item remove
    rm -rf "$APP"
    [ -L "$DESK" ] && rm -f "$DESK"
    echo "Removed $APP and its Desktop shortcut. The project and its data are untouched."
    exit 0 ;;
  --no-login)
    login_item remove
    exit 0 ;;
  ""|--login) ;;
  *) echo "usage: bash scripts/make_mac_app.sh [--login | --no-login | --remove]"; exit 2 ;;
esac

# The dashboard's port, from configs/server.yaml (8080 unless changed)
PORT="$(awk '/^dashboard:/ {d=1; next} d && /^[^ #]/ {d=0} d && $1=="port:" {print $2; exit}' configs/server.yaml)"
PORT="${PORT:-8080}"
URL="http://127.0.0.1:$PORT"

TMP="$(mktemp -d)"
SRC="$TMP/ppe_monitor.applescript"
cat > "$SRC" <<APPLESCRIPT
-- PPE Monitor: made by scripts/make_mac_app.sh. Make it again rather than editing it.
property projectDir : "$ROOT"
property dashboardURL : "$URL"

on run
	set answer to "000"
	try
		set answer to do shell script "curl -s -o /dev/null -w '%{http_code}' --max-time 2 " & quoted form of (dashboardURL & "/api/health")
	end try
	if answer is not "000" then
		-- already running: just show the dashboard
		open location dashboardURL
	else
		tell application "Terminal"
			activate
			do script "clear; bash " & quoted form of (projectDir & "/scripts/ppe.sh") & " start --keep-running"
		end tell
	end if
end run
APPLESCRIPT

mkdir -p "$APPDIR"
rm -rf "$APP"
osacompile -o "$APP" "$SRC" || { echo "osacompile failed (see above)."; rm -rf "$TMP"; exit 1; }
rm -rf "$TMP"

# The icon: assets/app_icon.png -> the app's applet.icns
if [ -f assets/app_icon.png ] && command -v sips >/dev/null 2>&1 && command -v iconutil >/dev/null 2>&1; then
  SET="$(mktemp -d)/app.iconset"
  mkdir -p "$SET"
  for s in 16 32 128 256 512; do
    sips -z "$s" "$s" assets/app_icon.png --out "$SET/icon_${s}x${s}.png" >/dev/null
    sips -z $((s * 2)) $((s * 2)) assets/app_icon.png --out "$SET/icon_${s}x${s}@2x.png" >/dev/null
  done
  if iconutil -c icns "$SET" -o "$APP/Contents/Resources/applet.icns" 2>/dev/null; then
    touch "$APP"
  else
    echo "  (could not make the icon; the app works with the standard one)"
  fi
  rm -rf "$(dirname "$SET")"
fi

# A shortcut on the Desktop
[ -e "$DESK" ] && [ ! -L "$DESK" ] && rm -rf "$DESK"
ln -sfn "$APP" "$DESK"

echo "Made: $APP"
echo "  and a shortcut on your Desktop. Double-click it to start the monitor, or to open the dashboard"
echo "  if it's already running. The first time, macOS asks if it may control Terminal: answer OK."
[ "${1:-}" = "--login" ] && login_item add
echo "  Keep the Mac plugged in and its lid open while it watches: closing a laptop's lid puts it to sleep."
exit 0
