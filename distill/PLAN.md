# PLAN — one-step distillation of the two 8786 policies

## Goal

Make the 8786 transport + insert inference faster by distilling the two DP3
teachers (10-step DDIM) into **one-step students**, following the RL-100
reference distillation pattern (`ref_official/RL-100/rl_100/policy/rl100_3d.py`),
adapted to the local DDIM sample-prediction policy.

## Teachers

- Transport: `hanoi_vservo_2026/runs/bc_vservo_transport_sem7_sgd180/best.ckpt`
  (semantic 7-channel point cloud, 18-D state).
- Insert: `hanoi_vservo_2026/runs/bc_vservo_insert_v2b/best.ckpt`
  (ghost-flag 7-channel point cloud, 18-D state).

Both: DP3, horizon 5, n_obs 2, n_action 4, 10 DDIM steps,
`prediction_type='sample'`, EMA weights.

## Method (RL-100 pattern, DDIM version)

For each batch of teacher-training observations:

1. Freeze the teacher UNet + encoder + normalizer (EMA weights).
2. Teacher rollout: 10-step DDIM from a noise tensor `eps` with the batch
   conditioning -> normalized action chunk `x0_teacher`.
3. Student (deepcopy of the teacher UNet) receives the **same `eps`** at the
   first teacher timestep `t=90` and predicts `x0_student`.
4. Loss: `MSE(x0_student, x0_teacher)`; encoder + normalizer frozen; only the
   UNet is trained; EMA 0.999 for the student.

Why `t=90` and `steps_offset=90`:
- `DDIMScheduler.set_timesteps(10)` uses `[90,80,...,0]`, so the teacher's
  first step is `t=90`.
- Saving the student with `num_inference_steps=1` and
  `noise_scheduler.steps_offset=90` makes `set_timesteps(1)` use `[90]`.
- The one-step DDIM update at `t=90` returns the predicted `x0` (verified:
  `prev_sample == pred_original_sample` when clipping is inactive), so the
  student's training and inference inputs match.

## Data

- Transport: `data/vservo_transport_2051_bal_w0..w3_sem7.zarr`
  (semantic 7-channel, balanced transport).
- Insert: `data/vservo_insert_general_v2b.zarr` +
  `data/vservo_insert_general_v2.zarr`.
- Same horizon/pad as training: horizon 5, pad_before 1, pad_after 0,
  n_obs 2; `obs_keys=['state','action','point_cloud']`; max episodes cap.
- Teacher normalizer reused; no refit.

## Outputs

- `runs/transport_distill/best.ckpt` — student checkpoint:
  `cfg.policy.num_inference_steps=1`, `steps_offset=90`; cfg dataset paths
  preserved for semantic/non-semantic detection.
- `runs/insert_distill/best.ckpt` — same.
- Loadable by the existing server (`build_policy_from_ckpt`).

## Evaluation

1. Microbenchmark: teacher 10-step vs student 1-step `predict_action`
   latency (batch 1, n_obs 2); report ms/chunk and the 5-sample transport
   cost.
2. Closed-loop: run the v1.2 server/chain with the two students on the same
   seed base as the teacher; n>=50, compare SR and steps.
3. Acceptance: student SR within ~1-2 points of the teacher on the paired
   seeds; otherwise increase data/epochs or try a 2-step student / CM.

## Isolation

New folder `hanoi_distill_2026/`; teacher data/checkpoints read-only; the
running 8786 server is not touched. Distilled weights and logs live here.
