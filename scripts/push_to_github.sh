#!/usr/bin/env bash
# Put the project on GitHub as a private repository, and later send what changed.
#
#   bash ~/Documents/ppe-monitor/scripts/push_to_github.sh               # first time: sign in, make the private repository, push
#   bash ~/Documents/ppe-monitor/scripts/push_to_github.sh               # afterwards: commit what changed and push it
#   bash ~/Documents/ppe-monitor/scripts/push_to_github.sh --name NAME   # another repository name (default: ppe-monitor)
#
# What goes in is decided by .gitignore: the code, configs, docs, the book and the model in use.
# Video clips, datasets, training runs, logs, events, older models and configs/secrets.env never do,
# and the script stops before pushing if any of them is about to.
# Signing in uses the GitHub CLI (gh) in your browser: your password never goes into this window,
# and gh keeps its token in the macOS keychain. If git or gh is missing, both are installed from
# conda-forge into a small conda environment called "github", not into the project's environment.
# Licence: the code is AGPL-3.0 (LICENSE); the model is for non-commercial use (README, "Licence").

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
NAME="ppe-monitor"
DESCRIPTION="Worker Safety (PPE) Monitoring System: missing helmets and vests, and restricted zones, on live CCTV, with alerts"

while [ $# -gt 0 ]; do
  case "$1" in
    --name) [ $# -ge 2 ] || { echo "usage: bash scripts/push_to_github.sh [--name NAME]"; exit 2; }
            NAME="$2"; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "usage: bash scripts/push_to_github.sh [--name NAME]"; exit 2 ;;
  esac
done
[ -n "$NAME" ] || { echo "usage: bash scripts/push_to_github.sh [--name NAME]"; exit 2; }

step() { echo ""; echo "== $*"; }
fail() { echo ""; echo "Stopped: $*"; exit 1; }

# ---------------------------------------------------------------------------------------------
step "1/5  git and the GitHub CLI"
# Apple's git only if the Command Line Tools are installed (otherwise /usr/bin/git opens an installer).
HAVE_GIT=""
if command -v git >/dev/null 2>&1; then
  if [ "$(uname)" != "Darwin" ] || [ "$(command -v git)" != "/usr/bin/git" ] || xcode-select -p >/dev/null 2>&1; then
    HAVE_GIT=1
  fi
fi
HAVE_GH=""
command -v gh >/dev/null 2>&1 && HAVE_GH=1

if [ -z "$HAVE_GIT" ] || [ -z "$HAVE_GH" ]; then
  CONDA=""
  for c in "$(command -v conda 2>/dev/null)" /opt/anaconda3/bin/conda "$HOME/anaconda3/bin/conda" /opt/miniconda3/bin/conda \
           "$HOME/miniconda3/bin/conda" "$HOME/miniforge3/bin/conda"; do
    if [ -n "$c" ] && [ -x "$c" ]; then CONDA="$c"; break; fi
  done
  [ -n "$CONDA" ] || fail "git or gh is missing, and there is no conda to install them with. Install GitHub's CLI from https://cli.github.com, then run this again."
  ENV_DIR="$("$CONDA" env list 2>/dev/null | awk '$1=="github" {print $NF}')"
  if [ -z "$ENV_DIR" ] || [ ! -x "$ENV_DIR/bin/gh" ] || [ ! -x "$ENV_DIR/bin/git" ]; then
    echo "Installing git and the GitHub CLI from conda-forge into a conda environment called \"github\" (~1 min) ..."
    if [ -z "$ENV_DIR" ]; then
      "$CONDA" create -y -q -n github -c conda-forge --override-channels gh git >/dev/null || fail "conda could not install gh and git"
    else
      "$CONDA" install -y -q -n github -c conda-forge --override-channels gh git >/dev/null || fail "conda could not install gh and git"
    fi
    ENV_DIR="$("$CONDA" env list 2>/dev/null | awk '$1=="github" {print $NF}')"
  fi
  [ -x "$ENV_DIR/bin/gh" ] && [ -x "$ENV_DIR/bin/git" ] || fail "the \"github\" conda environment has no gh or git"
  export PATH="$ENV_DIR/bin:$PATH"
fi
echo "$(git --version), $(gh --version | head -1)"

# ---------------------------------------------------------------------------------------------
step "2/5  Signing in to GitHub"
if gh auth status --hostname github.com >/dev/null 2>&1; then
  echo "Already signed in."
else
  echo "It shows a one-time code: copy it, press Enter, and paste the code into the GitHub page that opens"
  echo "in your browser, then approve. If it asks whether git should use your GitHub sign-in, answer Yes."
  gh auth login --hostname github.com --git-protocol https --web || fail "not signed in to GitHub"
fi
gh auth setup-git --hostname github.com >/dev/null 2>&1 || true      # git pushes with gh's sign-in
OWNER="$(gh api user --jq .login)" || fail "could not read your GitHub account"
echo "Signed in as $OWNER."

# ---------------------------------------------------------------------------------------------
step "3/5  The local repository"
if [ ! -d .git ]; then
  git init -q -b main . || fail "git init failed"
  echo "Made a new repository here."
fi
if [ -z "$(git config user.email || true)" ]; then          # commits by your GitHub account, email kept private
  ID="$(gh api user --jq .id)"
  git config user.name "$OWNER"
  git config user.email "${ID}+${OWNER}@users.noreply.github.com"
fi
git add -A . || fail "git add failed"

# What must never go to GitHub, even by mistake
BAD="$(git ls-files | grep -E '^configs/secrets\.env$|(^|/)\.env$|^data/clips/.*\.(mp4|mov|avi|mkv|m4v)$|^data/(events|postgres)/|^datasets/|^runs/|^logs/.+\.(log|txt)$|^models/(candidates|exported|pretrained)/' || true)"
if [ -n "$BAD" ]; then
  git reset -q
  echo "$BAD" | head -20
  fail "these files must not go to GitHub. Check .gitignore, then run this again."
fi
BIG="$(git ls-files -z | while IFS= read -r -d '' f; do
         [ -f "$f" ] && [ "$(wc -c < "$f" | tr -d ' ')" -gt 52428800 ] && echo "$f"; done)"
[ -z "$BIG" ] || { git reset -q; echo "$BIG"; fail "files over 50 MB (GitHub refuses files over 100 MB)."; }

if git rev-parse --verify -q HEAD >/dev/null; then
  if git diff --cached --quiet; then
    echo "Nothing changed since the last commit."
  else
    git diff --cached --stat | tail -1
    git commit -q -m "Update from the Mac, $(date '+%d %b %Y %H:%M')" || fail "git commit failed"
    echo "Committed the changes."
  fi
else
  git commit -q -m "Worker Safety (PPE) Monitoring System" || fail "git commit failed"
  echo "First commit made."
fi
echo "$(git ls-files | wc -l | tr -d ' ') files in the repository; latest commit: $(git log -1 --format='%h %s')"

# ---------------------------------------------------------------------------------------------
step "4/5  The repository on GitHub"
if git remote get-url origin >/dev/null 2>&1; then
  echo "Sends to $(git remote get-url origin)"
elif gh repo view "$OWNER/$NAME" >/dev/null 2>&1; then
  git remote add origin "https://github.com/$OWNER/$NAME.git"
  echo "$OWNER/$NAME already exists on GitHub: sending to it."
else
  gh repo create "$NAME" --private --description "$DESCRIPTION" --source . --remote origin \
    || fail "could not make the repository $OWNER/$NAME"
  echo "Made the private repository $OWNER/$NAME."
fi
VIS="$(gh repo view --json visibility --jq .visibility 2>/dev/null || echo unknown)"
[ "$VIS" = "PRIVATE" ] || echo "Note: the repository on GitHub is $VIS, not private. Change it on GitHub: Settings > General > Danger Zone."

# ---------------------------------------------------------------------------------------------
step "5/5  Pushing"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
git push -u origin "$BRANCH" || fail "the push failed (see above). If GitHub already has other commits, ask before forcing anything."

echo ""
echo "Done: $(gh repo view --json url --jq .url 2>/dev/null || echo "https://github.com/$OWNER/$NAME")"
echo "Later changes: run this again. To copy the project to another Mac: git clone that address into"
echo "~/Documents/ppe-monitor, then bash ~/Documents/ppe-monitor/scripts/mac_setup.sh."
exit 0
