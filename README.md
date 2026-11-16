# hanoi_general_newcam_1.0 — v1.1 single-transport

Clean copy of the audited general new-camera Hanoi stack, updated to
**one transport model** instead of three destination specialists.

## v1.1 changes

- **One transport policy** `runs/bc_vservo_transport_gen_v2_uni_sem7/best.ckpt`
  replaces the `P0/P1/P2` per-destination specialists.
- **Semantic point-cloud channel** (7th channel): `0 = other`,
  `1 = virtual goal`, `2 = object to manipulate`. It is computed from the
  vision-estimated disk pose/normal and the task disk colour — no simulator
  ground truth. `VSERVO_SEMANTIC_CHANNEL=1`.
- The insert policy is the same `bc_vservo_insert_v2b` (non-semantic); the
  server detects the checkpoint type and binarises the 7th channel for it.
- New insert **wedge-release** interlock and a wider grasp-height gate
  (`TRANSPORT_GRASP_GATE=1`, `[-15 mm, +25 mm]`).
- The running 8786 server uses `N_SAMPLES=5` transport action averaging,
  `INSERT_SAMPLES=1`, `VSERVO_VISION_SKIP=0`, `VSERVO_STATIC_CACHE=1`.

## Live result at capture

- **16/16 = 100%** live, all `switch=handoff_vision`;
  levels 0:8, 1:6, 2:2; disks s:12, m:4; mean 57.1 steps.
- Wilson 95% CI ~80.6–100% (live sample; not a frozen n=200 holdout).
- Raw log: `results/webserver_8786_sem7_session.log`;
  summary: `results/sem7_summary.json`.

## Stack

- Transport: single `gen_v2_uni_sem7` policy.
- Insert: `bc_vservo_insert_v2b`.
- Vision-only residual servos/interlocks (`transport_residual_servo.py`):
  terminal transport servo, insert terminal servo, grasp-height gate,
  lift gate, carry-high rule, force-close, wedge release.
- 18-D vision observation + 1024x7 point cloud (semantic channel or binary
  ghost flag depending on the checkpoint).
- Camera pose: `results/camera_pose_latest.json`; OSMesa headless rendering.
- No fallback/rescue: `FALLBACK_CKPT=""`.

See `AUDIT.md`. The **v1.0 release** remains the three-transport version
(P0/P1/P2 + insert v2b).

## Weights

| file | size |
|---|---:|
| `runs/bc_vservo_transport_gen_v2_uni_sem7/best.ckpt` | 621 MB |
| `runs/bc_vservo_insert_v2b/best.ckpt` | 621 MB |

Attached to the **v1.1 GitHub Release**; fetch with `./download_weights.sh`.
SHA-256s are in `CHECKSUMS.sha256` / `WEIGHTS.md`.

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
- The result is a hybrid controller: learned policies + hand-coded vision
  servos/interlocks + 5x transport action averaging. It is not a pure-model
  result.
- Live session, not a frozen n=200 holdout; per-episode simulator resets.
- Task destination/level condition the ghost target and policy; legitimate
  for a task-conditioned controller, not target inference.
- The semantic channel is vision-derived, but it is still an engineered
  observation channel; it must be reproducible on the real camera.

## License

Apache-2.0, inherited from RL-100. See `LICENSE` and `NOTICE`.
