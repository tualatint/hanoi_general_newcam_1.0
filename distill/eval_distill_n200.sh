#!/usr/bin/env bash
# Sharded headless eval of the distilled stack on the v1.2 jitter scene.
# Same controller/interlocks as run_distill_server.sh, no Flask/rendering.
#
#   ./eval_distill_n200.sh
#   N=100 W=4 SEED_BASE=1020000 TAG=myrun ./eval_distill_n200.sh
#   T=../runs/bc_vservo_transport_sem7_sgd180/best.ckpt \
#   I=../runs/bc_vservo_insert_v2b/best.ckpt ...   # teacher comparison
set -u
DISTILL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$DISTILL/.." && pwd)"
PY="${PY:-python}"
N="${N:-200}"
W="${W:-4}"
SEED_BASE="${SEED_BASE:-1011000}"
TAG="${TAG:-distill_j12_n${N}}"
cd "$DISTILL"
mkdir -p results logs data

export HANOI_INC="$REPO"
export HANOI_CAMERA_POSE="$REPO/results/camera_pose_latest.json"
export PYTHONPATH="$REPO/src:$REPO"
# shellcheck disable=SC1091
source "$REPO/scripts/env.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# v1.2 runtime configuration
export N_SAMPLES=5
export INSERT_SAMPLES=1
export EXEC_STEPS=4
export INSERT_EXEC_STEPS=2
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
export VSERVO_STATIC_CACHE=1
export VSERVO_ICP_WORKERS=1
export VSERVO_VISION_SKIP=0
export VSERVO_SEMANTIC_CHANNEL=1
export HANOI_PEG_JITTER_MM=12
export HANOI_PEG_MIN_DIST_MM=0
export HANOI_PEG_SEARCH_MM=70
export HANOI_GRIPPER_HALF_MM=48
export HANOI_PEG_COLLISION_MARGIN_MM=5
export HANOI_PEG_TRIES=12

T="${T:-$DISTILL/runs/transport_distill/best.ckpt}"
I="${I:-$DISTILL/runs/insert_distill/best.ckpt}"
per=$(( (N + W - 1) / W ))
pids=()
for w in $(seq 0 $((W - 1))); do
  base=$(( SEED_BASE + w * per ))
  rem=$(( N - w * per ))
  [ "$rem" -le 0 ] && break
  nw=$per
  [ "$rem" -lt "$per" ] && nw=$rem
  "$PY" eval_general_chain.py \
    --transport-ckpt "$T" --insert-ckpt "$I" \
    --n "$nw" --seed-base "$base" \
    --transport-steps 120 --insert-steps 120 \
    --camera-pose "$HANOI_CAMERA_POSE" \
    --vision-only-handoff \
    --n-samples 5 --insert-samples 1 \
    --exec-steps 4 --insert-exec-steps 2 \
    --save-failures "data/fail_${TAG}" \
    --out "results/chain_${TAG}_w${w}.json" \
    > "logs/eval_${TAG}_w${w}.log" 2>&1 &
  pids+=($!)
  echo "[eval] worker $w seeds ${base}..$((base + nw - 1)) pid $!"
done

rc=0
for p in "${pids[@]}"; do
  wait "$p" || rc=1
done
echo "[eval] workers done rc=$rc"
"$PY" merge_eval.py "results/chain_${TAG}_w*.json" \
  --out "results/chain_${TAG}_n${N}.json"
