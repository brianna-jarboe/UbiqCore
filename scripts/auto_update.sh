#!/usr/bin/env bash
set -euo pipefail

# Auto-update UbiqCore from GitHub and restart app service only when HEAD changed.

REPO_DIR="${REPO_DIR:-/home/brianna/apps/ubiqcore}"
APP_SERVICE_NAME="${APP_SERVICE_NAME:-ubiqcore-streamlit.service}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCK_FILE="${LOCK_FILE:-/tmp/ubiqcore-auto-update.lock}"
LOG_TAG="ubiqcore-auto-update"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

restart_and_check_service() {
  # 1) Try system service directly (works if unit user has permissions).
  if systemctl restart "$APP_SERVICE_NAME" 2>/dev/null && systemctl is-active --quiet "$APP_SERVICE_NAME" 2>/dev/null; then
    log "Service restart succeeded via systemctl"
    return 0
  fi

  # 2) Try sudo non-interactive for system service.
  if sudo -n systemctl restart "$APP_SERVICE_NAME" 2>/dev/null && sudo -n systemctl is-active --quiet "$APP_SERVICE_NAME" 2>/dev/null; then
    log "Service restart succeeded via sudo -n systemctl"
    return 0
  fi

  # 3) Try user service as fallback.
  if systemctl --user restart "$APP_SERVICE_NAME" 2>/dev/null && systemctl --user is-active --quiet "$APP_SERVICE_NAME" 2>/dev/null; then
    log "Service restart succeeded via systemctl --user"
    return 0
  fi

  log "Failed to restart/check $APP_SERVICE_NAME via all methods"
  return 1
}

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "Another update process is already running. Exiting."
  exit 0
fi

if [[ ! -d "$REPO_DIR/.git" ]]; then
  log "Repo directory is invalid: $REPO_DIR"
  exit 1
fi

cd "$REPO_DIR"

if ! git diff --quiet || ! git diff --cached --quiet; then
  log "Working tree has local changes. Skipping auto-update to avoid conflicts."
  exit 0
fi

CURRENT_HEAD="$(git rev-parse HEAD)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

log "Fetching latest origin/$BRANCH"
git fetch --prune origin "$BRANCH"
REMOTE_HEAD="$(git rev-parse "origin/$BRANCH")"

if [[ "$CURRENT_HEAD" == "$REMOTE_HEAD" ]]; then
  log "Already up to date at $CURRENT_HEAD"
  exit 0
fi

log "New commit detected: $CURRENT_HEAD -> $REMOTE_HEAD"
git pull --ff-only origin "$BRANCH"

# Reinstall dependencies only when dependency manifests changed in the pull.
if git diff --name-only "$CURRENT_HEAD" HEAD | grep -Eq '^(requirements\.txt|pyproject\.toml|setup\.py)$'; then
  log "Dependency manifest changed. Installing Python requirements."
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install -r requirements.txt
fi

log "Restarting app service: $APP_SERVICE_NAME"
restart_and_check_service

NEW_HEAD="$(git rev-parse HEAD)"
log "Update complete. Running commit: $NEW_HEAD"
