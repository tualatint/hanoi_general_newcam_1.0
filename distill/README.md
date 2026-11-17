# One-step distilled students (extra, v1.3-distill)

Optional extra content for the v1.2 stack: the two DP3 policies distilled to
**one DDIM step** each.  This folder does **not** modify the original v1.2
checkpoints, `run_server.sh`, or the v1.2 release.

| policy | checkpoint | size | role |
|---|---|---:|---|
| transport student | `runs/transport_distill/best.ckpt` | 619,480,526 | replaces `sem7_sgd180` at inference |
| insert student | `runs/insert_distill/best.ckpt` | 619,478,478 | replaces insert `v2b` at inference |

Both students are released as assets of the **v1.3-distill** GitHub Release
(not Git LFS).  Download:

```bash
cd distill
./download_weights.sh          # fetches + verifies CHECKSUMS.sha256
```

## Result

- Paired teacher vs students on the same 200 fresh sigma=12 seeds
  (1011000-1011199, Gaussian peg layouts):
  teacher **200/200 = 100%**; distilled **198/200 = 99.0%**
  (Wilson 95% CI 96.4-99.7%).  Both student failures are insert timeouts
  (one after a transport timeout, one after a clean handoff).
- Live 8789 webserver on the same scene: 109/109 at the time of capture.
- Policy latency (batch 1, n_obs 2): transport 69.20 -> **7.42 ms/chunk**
  (9.3x), insert 103.65 -> **11.73 ms/chunk** (8.8x).  The full server step
  is still vision-bound, so end-to-end episode time improves much less.
- Insert DAgger fine-tune was tried and is a **negative result**
  (196/200; it did not fix the two failures and added two insert
  regressions), so it is not included or promoted.  See `RESULTS.md`.

## Run (original v1.2 files untouched)

```bash
cd distill
./run_distill_server.sh              # port 8789, seed 973000, Gaussian pegs
PORT=8790 SEED=974000 ./run_distill_server.sh
NO_LOOP=1 ./run_distill_server.sh    # one episode
```

The launcher sets `TRANSPORT_CKPT` / `INSERT_CKPT` to the students and runs
the same `scripts/realtime_general_newcam.py` server with the v1.2 jitter
scene (`HANOI_PEG_JITTER_MM=12`).  Everything else is the original stack.

## Evaluate

```bash
cd distill
N=200 W=4 SEED_BASE=1011000 ./eval_distill_n200.sh
# teacher comparison on the same seeds:
T=../runs/bc_vservo_transport_sem7_sgd180/best.ckpt \
I=../runs/bc_vservo_insert_v2b/best.ckpt \
N=200 W=4 SEED_BASE=1011000 TAG=teacher_n200 ./eval_distill_n200.sh
```

Requires the original v1.2 checkpoints for the teacher comparison
(`../download_weights.sh` at the repo root).  `eval_general_chain.py` is the
headless vision-only chain evaluator (no rescue, no GT switch).

## Train (reproduction notes)

`distill.py` is the exact recipe used (RL-100 teacher/student same-noise MSE):

1. Freeze teacher UNet + encoder + normalizer (EMA).
2. Teacher: 10-step DDIM from noise `eps` -> `x0_teacher`.
3. Student (deepcopy of the teacher UNet): same `eps` at `t=90` -> `x0_student`.
4. `MSE(x0_student, x0_teacher)`; only the UNet trains; EMA 0.999.
5. Save with `num_inference_steps=1`, `steps_offset=90`.

The teacher demos and teacher checkpoints are part of the original training
workspace (see `PLAN.md`); they are not duplicated here.
