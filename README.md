# hanoi_general_newcam_1.0 — v1.2 single-transport

Clean copy of the running general new-camera Hanoi stack, updated to
**one transport model** plus the new jitter/slip interlocks.

## v1.2 changes

- **One transport policy**
  `runs/bc_vservo_transport_sem7_sgd180/best.ckpt` (replaces the
  `P0/P1/P2` per-destination specialists).
- **Semantic point-cloud channel** (7th channel): `0 = other`,
  `1 = virtual goal`, `2 = object to manipulate`; computed from the
  vision-estimated disk pose/normal and the task disk colour — no simulator
  ground truth. `VSERVO_SEMANTIC_CHANNEL=1`.
- **New transport interlocks** (vision/proprio only):
  `TRANSPORT_GRASP_ALIGN=1`, `TRANSPORT_GRASP_ASSIST=1`,
  `TRANSPORT_GRASP_COMMIT=1`, `TRANSPORT_LIFT_STALL=1`,
  `TRANSPORT_SLIP_RESET=1`.
- Insert is the same `bc_vservo_insert_v2b` (non-semantic); the server
  detects the checkpoint type and binarises channel 6 for it.
- Store release: v1.0 (three transports), v1.1 (`gen_v2_uni_sem7`), v1.2
  (current `sem7_sgd180` + jitter/slip interlocks).

## Live result at capture

- **120/120 = 100%** live, all `switch=handoff_vision`;
  levels 0:48, 1:47, 2:25; disks s:70, m:29, l:21; mean 57.0 steps.
- Wilson 95% CI ~96.9–100%.
- Raw log: `results/webserver_8786_jitter_session.log`;
  summary: `results/jitter_summary.json`.

## Stack

- Transport: single `sem7_sgd180` policy.
- Insert: `bc_vservo_insert_v2b`.
- Vision-only residual servos/interlocks (`transport_residual_servo.py`):
  terminal transport servo, insert terminal servo, grasp-height gate,
  grasp align/assist/commit, lift gate, lift-stall reset, slip reset,
  carry-high rule, force-close, wedge release.
- 18-D vision observation + 1024x7 point cloud (semantic channel or binary
  ghost flag depending on the checkpoint).
- Camera pose: `results/camera_pose_latest.json`; OSMesa headless rendering.
- No fallback/rescue: `FALLBACK_CKPT=""`.

## Weights

| file | size |
|---|---:|
| `runs/bc_vservo_transport_sem7_sgd180/best.ckpt` | 621 MB |
| `runs/bc_vservo_insert_v2b/best.ckpt` | 621 MB |

Attached to the **v1.2 GitHub Release**; fetch with `./download_weights.sh`.
SHA-256s are in `CHECKSUMS.sha256` / `WEIGHTS.md`.
The v1.0 and v1.1 releases remain available.

## Setup

```bash
./download_weights.sh
bash setup_env.sh
conda activate hanoi-newcam
source scripts/env.sh
```

System packages for OSMesa:

```bash
sudo apt-get install -y libosmesa6-dev libglew-dev libglfw3 \
                        libgl1-mesa-glx libegl1-mesa patchelf
```

## Run

```bash
./run_server.sh                 # port 8788, seed 896250, loops
# http://127.0.0.1:8788/  or  http://<lan-ip>:8788/
```

Variants:

```bash
NO_LOOP=1 ./run_server.sh       # one episode
SLOW=1 ./run_server.sh          # static vision cache OFF
POLICY_ONLY=1 ./run_server.sh   # N=1, transport servos/interlocks OFF
PORT=8790 SEED=900000 ./run_server.sh
```

`POLICY_ONLY=1` is an ablation, not the 1.0 configuration.

## Scope and caveats

- Task: single-move random disk -> random peg at level 0/1/2, no full Hanoi
  size rule. Not the full multi-move chain.
- Hybrid controller: learned policies + hand-coded vision servos/interlocks
  + 5x transport action averaging. Not a pure-model result.
- Live session (120 episodes), not a frozen n=200 holdout; per-episode
  simulator resets.
- Task destination/level condition the ghost target and policy; legitimate
  for a task-conditioned controller, not target inference.
- The semantic channel is vision-derived but still an engineered observation;
  it must be reproducible on the real camera.

## Extra: distilled 1-step students (`distill/`)

An optional extra variant: one-step DDIM students of the two v1.2 policies
(9.3x / 8.8x faster policy calls). It lives in `distill/`, ships as the
`v1.3-distill` release assets, and does **not** modify the v1.2 checkpoints,
`run_server.sh`, or this release. Paired n=200 on fresh sigma=12 seeds:
teacher 200/200, distilled 198/200 = 99.0%. See `distill/README.md`.

## License

Apache-2.0, inherited from RL-100. See `LICENSE` and `NOTICE`.
