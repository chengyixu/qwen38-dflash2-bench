#!/bin/bash
# Controller for the oMLX z-lab fork serving Qwen3.8-27B 4-bit + DFlash2.
# Drop-in replacement for the retired llama.cpp deployment on port 7870.
set -euo pipefail

PORT=7870
HOST="127.0.0.1"
BASE_URL="http://${HOST}:${PORT}"
LABEL="local.omlx.qwen38-dflash2"
PLIST="${HOME}/Library/LaunchAgents/${OLD_LABEL:-local.llamacpp.qwen38-dflash2}.plist"
LOG_DIR="${HOME}/Library/Logs/LocalLLM"
LOG_FILE="${LOG_DIR}/qwen38-omlx-dflash2.log"
DEPLOYMENT_ROOT="${HOME}/Models/qwen38-omlx-dflash2"
MODEL_DIR="${HOME}/Models/mlx"
MODEL_ID="Qwen3.8-27B-4bit"
API_KEY="REDACTED"
APP_RES="/Applications/oMLX.app/Contents/Resources"

server_pids() {
    pgrep -f "omlx.cli serve.*--port ${PORT}" 2>/dev/null || true
}

is_running() {
    [ -n "$(server_pids)" ]
}

active_model_id() {
    curl -fsS --max-time 3 -H "Authorization: Bearer ${API_KEY}" "${BASE_URL}/v1/models" 2>/dev/null \
        | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["data"][0]["id"] if d.get("data") else "")' 2>/dev/null || true
}

status() {
    local pid
    pid=$(lsof -ti ":${PORT}" 2>/dev/null | head -n1 || true)
    if [ -z "$pid" ] && ! is_running; then
        printf 'STOPPED: port %s is free\n' "$PORT"
        return
    fi
    local model_id health
    model_id=$(active_model_id)
    health=$(curl -fsS --max-time 3 "${BASE_URL}/health" 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("status",""))' 2>/dev/null || true)
    if [ "$health" = "healthy" ]; then
        printf 'RUNNING (ready): %s on %s (PID %s)\n' "${model_id:-starting}" "$BASE_URL" "$pid"
    else
        printf 'STARTING: server up at %s but model still loading (PID %s)\n' "$BASE_URL" "$pid"
    fi
}

sync_provider() {
    local provider_json="${HOME}/OWLSpace/provider.json"
    local provider_env="${HOME}/OWLSpace/provider.env"

    if [ -f "$provider_json" ]; then
        MODEL_ID="$MODEL_ID" python3 - "$provider_json" <<'PY'
import json, os, sys
path = sys.argv[1]
with open(path, encoding="utf-8") as file:
    provider = json.load(file)
provider["active_id"] = "builtin:local:http://127.0.0.1:7870"
for entry in provider.get("providers", []):
    if entry.get("label") == "local":
        entry["base_url"] = "http://127.0.0.1:7870"
        entry["group"] = os.environ["MODEL_ID"]
        entry["api_key"] = os.environ.get("OMLX_API_KEY", "none")
local_provider = provider.get("local_provider")
if isinstance(local_provider, dict):
    local_provider["baseURL"] = "http://127.0.0.1:7870"
    local_provider["model"] = os.environ["MODEL_ID"]
with open(path, "w", encoding="utf-8") as file:
    json.dump(provider, file, indent=2)
    file.write("\n")
PY
    fi

    cat > "$provider_env" <<EOF
# OWLSpace provider: local Qwen3.8-27B 4-bit + DFlash2 via oMLX (z-lab fork)
unset ANTHROPIC_AUTH_TOKEN
export CLAUDE_CODE_DISABLE_1M_CONTEXT=1
export CLAUDE_CODE_AUTO_COMPACT_WINDOW=45000
export OMLX_API_KEY='${API_KEY}'
export ANTHROPIC_BASE_URL='http://127.0.0.1:7870'
export ANTHROPIC_API_KEY='${API_KEY}'
export ANTHROPIC_MODEL='${MODEL_ID}'
EOF
}

stop() {
    launchctl bootout "gui/$(id -u)/${OLD_LABEL:-local.llamacpp.qwen38-dflash2}" 2>/dev/null || true
    launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
    local pids
    pids=$(server_pids)
    if [ -n "$pids" ]; then
        kill -TERM $pids 2>/dev/null || true
        for _ in {1..30}; do
            [ -z "$(server_pids)" ] && break
            sleep 1
        done
    fi
    # last-resort: free the port
    local pid
    pid=$(lsof -ti ":${PORT}" 2>/dev/null | head -n1 || true)
    if [ -n "$pid" ] && ps -p "$pid" -o command= | grep -q "omlx"; then
        kill -TERM "$pid" 2>/dev/null || true
        sleep 3
    fi
}

write_plist() {
    mkdir -p "$LOG_DIR"
    cat > "${HOME}/Library/LaunchAgents/${LABEL}.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>${APP_RES}/Python/cpython-3.11/bin/python3.11</string>
    <string>-m</string>
    <string>omlx.cli</string>
    <string>serve</string>
    <string>--model-dir</string>
    <string>${MODEL_DIR}</string>
    <string>--host</string>
    <string>${HOST}</string>
    <string>--port</string>
    <string>${PORT}</string>
    <string>--log-level</string>
    <string>info</string>
  </array>
  <key>WorkingDirectory</key>
  <string>${APP_RES}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONPATH</key>
    <string>${APP_RES}:${APP_RES}/Python/framework-mlx-base/lib/python3.11/site-packages:${APP_RES}/Python/cpython-3.11/lib/python3.11/site-packages</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${LOG_FILE}</string>
  <key>StandardErrorPath</key>
  <string>${LOG_FILE}</string>
</dict>
</plist>
EOF
    plutil -lint "${HOME}/Library/LaunchAgents/${LABEL}.plist" >/dev/null
}

warm_model() {
    curl -fsS --max-time 900 -H "Authorization: Bearer ${API_KEY}" \
        -H "Content-Type: application/json" \
        "${BASE_URL}/v1/chat/completions" \
        -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8,\"temperature\":0}" \
        >/dev/null
}

start() {
    stop
    write_plist
    launchctl bootstrap "gui/$(id -u)" "${HOME}/Library/LaunchAgents/${LABEL}.plist" || true

    printf 'Loading Qwen3.8-27B 4-bit + DFlash2'
    for _ in {1..120}; do
        if [ "$(active_model_id)" = "$MODEL_ID" ] || lsof -ti ":${PORT}" >/dev/null 2>&1; then
            printf '\n'
            break
        fi
        printf '.'
        sleep 1
    done
    warm_model
    sync_provider
    status
}

test_api() {
    curl -fsS --max-time 600 -H "Authorization: Bearer ${API_KEY}" \
        -H "Content-Type: application/json" \
        "${BASE_URL}/v1/chat/completions" \
        -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: DFlash2 is ready\"}],\"temperature\":0,\"max_tokens\":300}" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"][-200:])'
}

usage() {
    cat <<EOF
Usage: $(basename "$0") <start|stop|restart|status|test|logs>

Serves Qwen3.8-27B 4-bit with DFlash2 speculative decoding via the
z-lab oMLX fork. Endpoint: ${BASE_URL}/v1 (OpenAI + Anthropic compatible)
EOF
}

case "${1:-status}" in
    start) start ;;
    stop) stop; status ;;
    restart) stop; start ;;
    status) status ;;
    test) test_api ;;
    logs) tail -n 100 -f "$LOG_FILE" ;;
    *) usage; exit 2 ;;
esac
