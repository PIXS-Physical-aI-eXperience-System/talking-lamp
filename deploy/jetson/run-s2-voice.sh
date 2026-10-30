#!/usr/bin/env bash
set -eo pipefail

repo=${TALKING_LAMP_REPO:-/home/asdf/talking-lamp-integration}
voice_root=${TALKING_LAMP_VOICE_ROOT:-/home/asdf/talking-lamp/voice-bench}
voice_python=${TALKING_LAMP_VOICE_PYTHON:-$voice_root/venvs/melo-onnx/bin/python}
llm_url=${TALKING_LAMP_VOICE_LLM_URL:-http://127.0.0.1:11434/v1/chat/completions}
voice_port=${TALKING_LAMP_VOICE_PORT:-5150}
ros_setup=${ROS_SETUP_FILE:-/opt/ros/jazzy/setup.bash}

source "$ros_setup"
source "$repo/jetson_ws/install/setup.bash"
set -u

if [[ ! -x "$voice_python" || ! -f "$voice_root/bench/voice_agent.py" ||
      ! -f "$voice_root/ros/lamp_voice_node.py" ]]; then
    echo "S2 voice files are missing under $voice_root" >&2
    exit 1
fi
if [[ ! "$voice_port" =~ ^[0-9]+$ ]] || ((voice_port < 1 || voice_port > 65535)); then
    echo "S2 voice port is invalid" >&2
    exit 1
fi
if ss -ltn "( sport = :$voice_port )" | grep -q "127.0.0.1:$voice_port"; then
    echo "S2 voice port $voice_port is already in use" >&2
    exit 1
fi

cd "$voice_root"
"$voice_python" bench/voice_agent.py --port "$voice_port" --llm "$llm_url" &
agent_pid=$!
node_pid=

cleanup() {
    if [[ -n "$node_pid" ]]; then
        kill -TERM "$node_pid" 2>/dev/null || true
    fi
    kill -TERM "$agent_pid" 2>/dev/null || true
    if [[ -n "$node_pid" ]]; then
        wait "$node_pid" 2>/dev/null || true
    fi
    wait "$agent_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ready=false
for ((attempt=0; attempt<180; attempt++)); do
    if ! kill -0 "$agent_pid" 2>/dev/null; then
        echo "S2 voice agent exited during startup" >&2
        exit 1
    fi
    if ss -ltn "( sport = :$voice_port )" | grep -q "127.0.0.1:$voice_port"; then
        ready=true
        break
    fi
    sleep 0.5
done
if [[ "$ready" != true ]]; then
    echo "S2 voice agent did not open port $voice_port within 90 seconds" >&2
    exit 1
fi

python3 ros/lamp_voice_node.py --agent "127.0.0.1:$voice_port" &
node_pid=$!
wait -n "$agent_pid" "$node_pid"
exit 1
