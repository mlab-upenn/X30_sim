#!/usr/bin/env bash
# Launch the simulated X30 motion host (left pane) and the MotionSDK policy runner (right pane)
# side by side in a tmux session.
#
# Usage: ./run_sim.sh                       # silent_policy, MuJoCo viewer
#        POLICY=slope_policy ./run_sim.sh
#        SIM_DISPLAY=:3 ./run_sim.sh        # force an X display
#        HEADLESS=1 ./run_sim.sh            # no viewer (bridge prints pose every 2 s)
#
# Keys (right pane): w/s vx, a/d vy, q/e yaw (+-0.1 per press), space stop, x release + exit.
# Ctrl-b + arrow switches panes, Ctrl-b d detaches, `tmux kill-session -t x30_sim` stops.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION=x30_sim
PY="$ROOT/.venv/bin/python"
POLICY="${POLICY:-silent_policy}"
BRIDGE_ARGS=""

if ! compgen -G "$ROOT/lib/deeprobotics_x30_motion_sdk_py*.so" >/dev/null; then
    echo "SDK python module not built; see README.md (Setup)." >&2
    exit 1
fi

# Find an X display we're authorized on: the current one may carry a cookie for a different display.
display_ok() {
    DISPLAY="$1" timeout 5 "$PY" -c 'import glfw, sys; sys.exit(0 if glfw.init() else 1)' >/dev/null 2>&1
}
if [[ -n "${HEADLESS:-}" ]]; then
    BRIDGE_ARGS="--headless"
elif [[ -n "${SIM_DISPLAY:-}" ]]; then
    export DISPLAY="$SIM_DISPLAY"
elif ! display_ok "${DISPLAY:-}"; then
    found=""
    for sock in /tmp/.X11-unix/X*; do
        d=":${sock##*/X}"
        if display_ok "$d"; then found="$d"; break; fi
    done
    if [[ -z "$found" ]]; then
        echo "No usable X display found (DISPLAY=${DISPLAY:-unset}); running headless." >&2
        BRIDGE_ARGS="--headless"
    else
        echo "DISPLAY=${DISPLAY:-unset} not usable, using $found"
        export DISPLAY="$found"
    fi
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -c "$ROOT" -e DISPLAY="${DISPLAY:-}" -e XAUTHORITY="${XAUTHORITY:-}"
tmux send-keys -t "$SESSION:0.0" "'$PY' sim_bridge.py $BRIDGE_ARGS" C-m

tmux split-window -h -t "$SESSION:0" -c "$ROOT"
# Give the bridge a moment to bind before the runner connects.
tmux send-keys -t "$SESSION:0.1" "sleep 2 && '$PY' -W ignore run_policy_sdk.py --policy $POLICY" C-m
tmux select-pane -t "$SESSION:0.1"

if [[ -n "${TMUX:-}" ]]; then
    tmux switch-client -t "$SESSION"
else
    tmux attach -t "$SESSION"
fi
