# RESULTS — one-step distillation (extra)

Teacher: v1.2 `bc_vservo_transport_sem7_sgd180` + `bc_vservo_insert_v2b`
(10 DDIM steps, EMA).  Student: deepcopy of the teacher UNet, same noise at
`t=90`, MSE to the teacher `x0`, encoder/normalizer frozen, EMA 0.999.
Saved with `num_inference_steps=1`, `steps_offset=90`.

## Training

| model | data | epochs x steps | final loss |
|---|---|---|---:|
| transport | `vservo_transport_2051_bal_w0..w3_sem7.zarr` | 10 x 200 | 0.000657 |
| insert | `vservo_insert_general_v2b.zarr` + `v2.zarr` | 10 x 200 | 0.002044 |

## Latency and action match

Batch 1, n_obs 2, 20 calls after warmup:

| policy | teacher 10-step | student 1-step | speedup | action MSE |
|---|---:|---:|---:|---:|
| transport | 69.20 ms/chunk | 7.42 ms/chunk | 9.3x | 1.2e-05 |
| insert | 103.65 ms/chunk | 11.73 ms/chunk | 8.8x | 9.6e-05 |

## Closed-loop (fresh sigma=12, Gaussian peg layouts)

Same controller/interlocks as the v1.2 server, seeds 1011000-1011199:

| policy | episodes | success | SR | Wilson 95% CI |
|---|---:|---:|---:|---|
| teacher | 200 | 200 | 100% | 98.1-100% |
| distilled students | 200 | 198 | 99.0% | 96.4-99.7% |

Both failures are `disk_s` insert timeouts:

| seed | task (disk, src, dst, level) | switch | t / i steps | attribution |
|---|---|---|---|---|
| 1011065 | (s, 2, 1, 1) | correction | 120 / 120 | transport: never reached the handoff box (z -14.3 mm), teacher hands off at t=48 |
| 1011189 | (s, 1, 0, 2) | handoff_vision | 32 / 120 | insert: clean handoff (err 0.26 mm), insert stalled; teacher seats in 4 steps |

Live 8789 webserver on the same scene: 109/109 at capture.

## Insert DAgger fine-tune (negative result, not included)

100 student rollouts (815 insert states) + the two failure trajectories
(120 states each) were labelled with the DP3 insert teacher and used to
fine-tune the student.  Offline same-noise MSE on the 240 failure states
improved from 0.0547 to 0.0049, but closed-loop it failed both original seeds
and added two new insert timeouts after clean handoffs:
**196/200 = 98.0%**.  Not promoted; the two student failures are early
divergence / physically stuck states, not a missing offline action match.

## Caveats

- Hybrid controller: learned policies + vision servos/interlocks + 5x
  transport action averaging.  The students replace only the two policy calls.
- n=200 is a fresh seed base but not the project's four frozen held-out bases.
- The one-step students were distilled from offline teacher data; if the
  deployment distribution shifts, add on-policy/DAgger data (see above for
  the first negative attempt) or use a 2-step student.
