#!/usr/bin/env bash
# Distilled 1-step student stack on port 8789.
#
# Runs the SAME v1.2 server/controller as ../run_server.sh, but points
# TRANSPORT_CKPT / INSERT_CKPT at the distilled students in this folder.
# The original v1.2 checkpoints and ../run_server.sh are not modified.
#
# Peg layouts: per-episode 2-D Gaussian offsets (sigma 12 mm), collision and
# open-jaw clearance rejected -- the same scene as the v1.2 live session.
#
#   ./run_distill_server.sh
#   PORT=8790 SEED=974000 ./run_distill_server.sh
#   NO_LOOP=1 ./run_distill_server.sh
set -euo pipefail
DISTILL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$DISTILL/.." && pwd)"
PY="${PY:-python}"

cd "$DISTILL"
export TRANSPORT_CKPT="${TRANSPORT_CKPT:-$DISTILL/runs/transport_distill/best.ckpt}"
export INSERT_CKPT="${INSERT_CKPT:-$DISTILL/runs/insert_distill/best.ckpt}"
export P0_CKPT= P1_CKPT= P2_CKPT=

# v1.2 runtime configuration
export N_SAMPLES="${N_SAMPLES:-5}"
export INSERT_SAMPLES="${INSERT_SAMPLES:-1}"
export EXEC_STEPS="${EXEC_STEPS:-4}"
export INSERT_EXEC_STEPS="${INSERT_EXEC_STEPS:-2}"
export TRANSPORT_FORCE_CLOSE=1
export TRANSPORT_TERMINAL_SERVO=0.5
export TRANSPORT_TERMINAL_GATE=0.080
export TRANSPORT_TERMINAL_Z_GATE=0.020
export INSERT_SERVO=0.5
export INSERT_SERVO_CENTER_XY=0.0008
export INSERT_TERMINAL_SERVO=0.5
export INSERT_TERMINAL_GATE=0.010
export INSERT_TERMINAL_Z_GATE=0.060
export INSERT_TERMINAL_CENTER_XY=0.0008
export INSERT_CENTER_XY=0.0008
export INSERT_CENTER_MAX_WAIT=14
export INSERT_RELEASE=1
export INSERT_RELEASE_EARLY_MM=0.005
export TRANSPORT_GRASP_GATE=1
export TRANSPORT_GRASP_GATE_MM=0.025
export TRANSPORT_GRASP_GATE_LO_MM=-0.015
export TRANSPORT_GRASP_SETTLE=4
export TRANSPORT_LIFT_GATE=1
export TRANSPORT_LIFT_GATE_MM=0.008
export TRANSPORT_LIFT_GATE_OFF_MM=-0.010
export TRANSPORT_LIFT_MIN_AZ=0.06
export TRANSPORT_GRASP_ASSIST=1
export TRANSPORT_SLIP_RESET=1
export TRANSPORT_GRASP_COMMIT=1
export TRANSPORT_GRASP_ALIGN=1
export TRANSPORT_LIFT_STALL=1
export HANOI_CAMERA_POSE="${HANOI_CAMERA_POSE:-$REPO/results/camera_pose_latest.json}"
export VSERVO_STATIC_CACHE=1
export VSERVO_ICP_WORKERS=1
export VSERVO_VISION_SKIP=0
export VSERVO_SEMANTIC_CHANNEL=1
# Gaussian peg placement (deployed 8786 scene).
export HANOI_PEG_JITTER_MM="${HANOI_PEG_JITTER_MM:-12}"
export HANOI_PEG_MIN_DIST_MM="${HANOI_PEG_MIN_DIST_MM:-0}"
export HANOI_PEG_SEARCH_MM="${HANOI_PEG_SEARCH_MM:-70}"
export HANOI_GRIPPER_HALF_MM="${HANOI_GRIPPER_HALF_MM:-48}"
export HANOI_PEG_COLLISION_MARGIN_MM="${HANOI_PEG_COLLISION_MARGIN_MM:-5}"
export HANOI_PEG_TRIES="${HANOI_PEG_TRIES:-12}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

SEED="${SEED:-973000}"
PORT="${PORT:-8789}"
# shellcheck disable=SC1091
source "$REPO/scripts/env.sh"

cd "$REPO"
args=(--port "$PORT" --host 0.0.0.0 --seed "$SEED"
      --max-steps "${MAX_STEPS:-260}" --video-size "${VIDEO_SIZE:-512}"
      --autostart)
if [ "${NO_LOOP:-0}" = "1" ]; then args+=(--no-loop); fi

echo "[distill_server] transport=$TRANSPORT_CKPT"
echo "[distill_server] insert=$INSERT_CKPT"
echo "[distill_server] seed=$SEED port=$PORT peg_jitter=${HANOI_PEG_JITTER_MM}mm"
exec "$PY" -u scripts/realtime_general_newcam.py "${args[@]}"
