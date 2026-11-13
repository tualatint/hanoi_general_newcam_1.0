#!/usr/bin/env bash
# Run the exact audited 8786 server configuration.
#
#   ./run_server.sh                 # port 8788, seed 896250, loops
#   PORT=8790 ./run_server.sh
#   NO_LOOP=1 ./run_server.sh       # one episode, then exit
#   SLOW=1 ./run_server.sh          # fast vision cache/skip OFF (clean A/B)
#   POLICY_ONLY=1 ./run_server.sh   # disable transport servos + sample averaging
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/env.sh

export INSERT_CKPT="${INSERT_CKPT:-$ROOT/runs/bc_vservo_insert_v2b/best.ckpt}"
export P0_CKPT="${P0_CKPT:-$ROOT/runs/bc_vservo_transport_bal_v2_p0/best.ckpt}"
export P1_CKPT="${P1_CKPT:-$ROOT/runs/bc_vservo_transport_bal_v2_p1/best.ckpt}"
export P2_CKPT="${P2_CKPT:-$ROOT/runs/bc_vservo_transport_bal_v2_p2/best.ckpt}"

# Exact 8786 runtime configuration
export N_SAMPLES="${N_SAMPLES:-5}"            # transport action averaging
export INSERT_SAMPLES="${INSERT_SAMPLES:-1}"  # insert is single-sample
export EXEC_STEPS="${EXEC_STEPS:-4}"
export INSERT_EXEC_STEPS="${INSERT_EXEC_STEPS:-2}"
export TRANSPORT_FORCE_CLOSE="${TRANSPORT_FORCE_CLOSE:-1}"
export TRANSPORT_GRASP_CENTER="${TRANSPORT_GRASP_CENTER:-0}"
export TRANSPORT_GRASP_CENTER_K="${TRANSPORT_GRASP_CENTER_K:-0.5}"
export TRANSPORT_GRASP_GATE="${TRANSPORT_GRASP_GATE:-0}"
export TRANSPORT_GRASP_SETTLE="${TRANSPORT_GRASP_SETTLE:-4}"
export TRANSPORT_LIFT_GATE="${TRANSPORT_LIFT_GATE:-1}"
export TRANSPORT_LIFT_GATE_MM="${TRANSPORT_LIFT_GATE_MM:-0.008}"
export TRANSPORT_LIFT_GATE_OFF_MM="${TRANSPORT_LIFT_GATE_OFF_MM:--0.010}"
export TRANSPORT_LIFT_MIN_AZ="${TRANSPORT_LIFT_MIN_AZ:-0.06}"
export TRANSPORT_TERMINAL_GATE="${TRANSPORT_TERMINAL_GATE:-0.080}"
export TRANSPORT_TERMINAL_SERVO="${TRANSPORT_TERMINAL_SERVO:-0.5}"
export TRANSPORT_TERMINAL_Z_GATE="${TRANSPORT_TERMINAL_Z_GATE:-0.020}"
export VSERVO_ICP_WORKERS="${VSERVO_ICP_WORKERS:-1}"
export HANOI_CAMERA_POSE="${HANOI_CAMERA_POSE:-$ROOT/results/camera_pose_latest.json}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

if [ "${SLOW:-0}" = "1" ]; then
    export VSERVO_STATIC_CACHE=0
    export VSERVO_VISION_SKIP=0
else
    export VSERVO_STATIC_CACHE="${VSERVO_STATIC_CACHE:-1}"
    export VSERVO_VISION_SKIP="${VSERVO_VISION_SKIP:-1}"
fi

if [ "${POLICY_ONLY:-0}" = "1" ]; then
    export N_SAMPLES=1
    export TRANSPORT_TERMINAL_SERVO=0
    export TRANSPORT_SERVO=0
    export TRANSPORT_FORCE_CLOSE=0
    export TRANSPORT_LIFT_GATE=0
    export TRANSPORT_GRASP_CENTER=0
    export VSERVO_STATIC_CACHE=0
    export VSERVO_VISION_SKIP=0
fi

args=(--port "${PORT:-8788}" --host 0.0.0.0
      --seed "${SEED:-896250}" --max-steps "${MAX_STEPS:-260}"
      --video-size "${VIDEO_SIZE:-512}" --autostart)
if [ "${NO_LOOP:-0}" = "1" ]; then args+=(--no-loop); fi

echo "[run_server] N_SAMPLES=$N_SAMPLES INSERT_SAMPLES=$INSERT_SAMPLES"
echo "[run_server] cache=$VSERVO_STATIC_CACHE skip=$VSERVO_VISION_SKIP terminal_servo=$TRANSPORT_TERMINAL_SERVO force_close=$TRANSPORT_FORCE_CLOSE lift_gate=$TRANSPORT_LIFT_GATE"
exec "$PY" -u scripts/realtime_general_newcam.py "${args[@]}"
