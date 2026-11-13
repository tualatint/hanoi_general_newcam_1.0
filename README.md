# hanoi_general_newcam_1.0

Clean copy of the audited **8786** experiment: a vision-only general
single-move Hanoi pick-and-place chain that achieved **42/42 = 100%** in a
live session.

## What this is

- **Task:** random top disk -> random target peg at target stack level 0/1/2,
  no full Tower-of-Hanoi size rule. This is a **single-move** task, not the
  full 3-disk multi-move chain.
- **Stack:**
  - transport specialists `P0/P1/P2`, selected by the task's destination peg;
  - insert policy `bc_vservo_insert_v2b`;
  - vision-only residual servos / safety interlocks
    (`transport_residual_servo.py`): terminal transport servo, insert
    terminal servo, grasp-height gate, lift gate, carry-high rule, force-close;
  - 18-D vision observation + 1024x7 point cloud with a magenta ghost target;
  - new camera pose from `results/camera_pose_latest.json`;
  - OSMesa headless rendering.
- **Live result:** 42/42 = 100%, all `switch=handoff_vision`, levels 0/1/2,
  disks s/m/l. Wilson 95% CI ~91.6–100%. This is a live session, not a frozen
  n=200 holdout.

## Important runtime details (disclosed)

- **Transport action averaging:** `N_SAMPLES=5` — the transport action is the
  mean of **five** policy samples per decision. Insert uses
  `INSERT_SAMPLES=1`.
- **Hand-coded vision servos** are active:
  `TRANSPORT_TERMINAL_SERVO=0.5`, `TRANSPORT_FORCE_CLOSE=1`,
  `TRANSPORT_LIFT_GATE=1`, grasp gate/settle, carry-high rule, etc.
  They use only vision estimates, FK, and gripper width — no simulator object
  ground truth — but the 1.0 is **policy + vision servo**, not the diffusion
  policy alone.
- **Fast eval flags active:** `VSERVO_STATIC_CACHE=1`,
  `VSERVO_VISION_SKIP=1`, `VSERVO_ICP_WORKERS=1`.
- **No fallback/rescue:** `FALLBACK_CKPT=""`, `FALLBACK_STEPS=0`. The log has
  no `correction` or `transport_fail` switch.
- **No retry/Q-filter/best-of-N:** every episode increments `ep_done`;
  success is `info["success"] and stage in (insert, fallback) and switched`.
- The original server also loaded an unused default transport checkpoint; this
  package removes that unused load (all destination specialists exist, so the
  behavior is unchanged).

See `AUDIT.md` for the full audit and `RESULTS.md` for the session summary.

## Weights

| file | size |
|---|---:|
| `runs/bc_vservo_insert_v2b/best.ckpt` | 621 MB |
| `runs/bc_vservo_transport_bal_v2_p0/best.ckpt` | 621 MB |
| `runs/bc_vservo_transport_bal_v2_p1/best.ckpt` | 621 MB |
| `runs/bc_vservo_transport_bal_v2_p2/best.ckpt` | 621 MB |

The four checkpoints are attached to the **v1.0 GitHub Release** (release
assets, not Git LFS). `CHECKSUMS.sha256` has the SHA-256s. Fetch them with:

```bash
./download_weights.sh
```

## Setup

```bash
git lfs install
git lfs pull             # fetch the four checkpoints
bash setup_env.sh        # conda env "hanoi-newcam"
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
# open http://127.0.0.1:8788/  (or http://<lan-ip>:8788/)
```

Useful variants:

```bash
NO_LOOP=1 ./run_server.sh       # one episode, then exit
SLOW=1 ./run_server.sh          # fast cache/skip OFF (clean A/B)
POLICY_ONLY=1 ./run_server.sh   # N=1, transport servos OFF, fast flags OFF
PORT=8790 SEED=900000 ./run_server.sh
```

`POLICY_ONLY=1` is an ablation, not the 1.0 configuration. It disables the
five-sample averaging, the transport servos/guards, and the fast vision
schedule.

## Environment

- Python 3.8.20, torch 2.4.0+cu121, MuJoCo 3.2.3, diffusers 0.33.1, gym 0.21.0.
- RTX 4090, Ubuntu 20.04.
- OSMesa is the original server's renderer; `scripts/env.sh` defaults to it.

## Provenance

- Original workstream: `RL-100/hanoi_vservo_2026/` (server:
  `scripts/realtime_general_newcam.py`).
- Source checkpoints: `runs/bc_vservo_insert_v2b/best.ckpt`,
  `runs/bc_vservo_transport_bal_v2_p{0,1,2}/best.ckpt`.
- Session log: `results/webserver_8786_session.log`
  (42 episodes, 42 successes, all `handoff_vision`).
- The v2b insert checkpoint is **not** the v4 expert-relative-loss checkpoint
  from `hanoi_vservo_errloss_1.0`; this is a different stack.

## License

Apache-2.0, inherited from RL-100. See `LICENSE` and `NOTICE`.
