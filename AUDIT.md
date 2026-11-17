# Audit — 8786 general newcam chain

## v1.2 update — `sem7_sgd180` + jitter/slip interlocks (audited 2026-11-17)

The running 8786 server now uses one transport checkpoint
`runs/bc_vservo_transport_sem7_sgd180/best.ckpt` (semantic channel), the same
insert `bc_vservo_insert_v2b`, and five new transport/insert interlocks:

- `TRANSPORT_GRASP_ASSIST=1` (grasp assist)
- `TRANSPORT_GRASP_ALIGN=1` (close only when aligned)
- `TRANSPORT_GRASP_COMMIT=1` (commit the grasp closure)
- `TRANSPORT_LIFT_STALL=1` (lift-stall detection/reset)
- `TRANSPORT_SLIP_RESET=1` (slip reset)

These use the vision error/disk estimate, FK, and gripper width; the
`transport_residual_servo.py` module remains vision-only by construction.

Checks:

| check | result |
|---|---|
| GT object pose used to choose actions | **Pass.** Handoff uses `_servo_err_raw`; semantic channel and all interlocks are vision/FK-derived; simulator `done` is ignored in transport. |
| Retry/reset of failed episodes | **Pass.** Every episode increments `ep_done`; no episode retry. `TRANSPORT_SLIP_RESET` resets controller state, not the episode. |
| Q-filter / best-of-N | **Pass.** No Q-filter; transport uses the **mean** of 5 samples. Insert N=1. |
| Fallback/rescue inflating SR | **Pass.** `FALLBACK_CKPT=""`; all 120 logged episodes are `handoff_vision`, no correction/fallback. |
| Success predicate trivially true | **Pass.** `_is_seated` requires upright, at rest, target peg, correct support height, low velocity; GT is evaluation-only. |
| Seed cherry-picking | **Not proven.** Live seeds 896250+; 120-episode continuous loop. |
| Frozen n=200 protocol | **No.** Live 120/120; Wilson 95% CI ~96.9–100%. |

Live result at capture: **120/120 = 100%**, all `handoff_vision`, levels
0:48, 1:47, 2:25; disks s:70, m:29, l:21; mean 57.0 steps. Raw log:
`results/webserver_8786_jitter_session.log`.

Caveats: hybrid controller (learned policies + vision servos/interlocks +
5x transport averaging); single-move task, not full multi-move Hanoi; live
sample, not frozen n=200; per-episode simulator resets; task destination/level
condition the target; semantic channel is an engineered observation.

Recommended before trusting v1.2: `SLOW=1` A/B, `POLICY_ONLY=1` ablation,
n>=200 on a fresh seed base, and a seed-disjointness check against the
transport/insert training data.

**Verdict:** no evidence of GT leakage, retry, Q-filter, or fallback in the
v1.2 action path. The 120/120 is honest for the live session, but it is a
hybrid/live result, not a pure-model/frozen-holdout 1.0.

---

## v1.1 update — single transport (audited 2026-11-16)

The running 8786 server was updated from three destination specialists
(`P0/P1/P2`) to **one transport policy**:
`runs/bc_vservo_transport_gen_v2_uni_sem7/best.ckpt`. The insert policy is
still `bc_vservo_insert_v2b`; `P0/P1/P2` are empty.

What changed:

- **Semantic 7th point-cloud channel**: `0 = other`, `1 = virtual goal`,
  `2 = object to manipulate` (`vservo_env.py`). It is computed from the
  vision-estimated disk centre/normal and the task disk colour — not from
  simulator ground truth.
- The server inspects each checkpoint config and binarises channel 6 for the
  non-semantic insert policy; the semantic transport policy receives the
  0/1/2 channel unchanged.
- New insert **wedge-release** interlock (vision + gripper width) and a wider
  grasp-height gate (`TRANSPORT_GRASP_GATE=1`, `[-15 mm, +25 mm]`).
- `VSERVO_VISION_SKIP=0` (clean vision schedule), `VSERVO_STATIC_CACHE=1`,
  `N_SAMPLES=5` transport averaging, `INSERT_SAMPLES=1`.

Checks (same criteria as v1.0):

| check | result |
|---|---|
| GT object pose used to choose actions | **Pass.** Handoff uses `_servo_err_raw`; the semantic channel is vision-derived; the servo uses `_servo_err`/`_disk_centre`/FK/gripper width. Simulator `done` is ignored in transport. |
| Retry/reset of failed episodes | **Pass.** Every episode increments `ep_done`; no retry. |
| Q-filter / best-of-N | **Pass.** No Q-filter; transport uses the **mean** of 5 samples, not best-of-N. Insert N=1. |
| Fallback/rescue inflating SR | **Pass.** `FALLBACK_CKPT=""`; the sem7 log has only `handoff_vision`, no correction/fallback. |
| Success predicate trivially true | **Pass.** `_is_seated` requires upright, at rest, target peg, correct support height, low velocity. GT is evaluation-only. |
| Seed cherry-picking | **Not proven.** Live seeds 896250+; 16-episode sample. |
| Frozen n=200 protocol | **No.** Live 16/16; Wilson 95% CI ~80.6–100%. |

Live result at capture: **16/16 = 100%**, all `handoff_vision`, levels 0:8,
1:6, 2:2, mean 57.1 steps. Raw log:
`results/webserver_8786_sem7_session.log`.

Caveats: still a **hybrid controller** (learned policies + vision servos/
interlocks + 5x transport averaging); single-move task, not full multi-move
Hanoi; small live sample; per-episode simulator resets; the semantic channel
is an engineered observation and must be reproducible on the real camera.

Recommended before trusting v1.1: `SLOW=1` A/B (static cache off),
`POLICY_ONLY=1` ablation (N=1, servos/interlocks off), n>=200 on a fresh seed
base, and a seed-disjointness check against the transport/insert training
data.

**Verdict:** no evidence of GT leakage, retry, Q-filter, or fallback in the
v1.1 action path. The 16/16 is honest for the live session, but it is a
hybrid/live-sample result, not a pure-model/frozen-holdout 1.0.

---

## v1.0 audit — three-transport version (2026-11-13)

Audited live at `http://127.0.0.1:8786/`. The server was
`hanoi_vservo_2026/scripts/realtime_general_newcam.py`, cwd
`hanoi_vservo_2026`, seed loop 896250+, `max_steps=260`.

## What it actually runs

- Task: `(disk_idx, src, dst, target_level)`; single-move top-disk placement,
  no full Hanoi size rule (`general_env.py:46-63`).
- Policies: transport `P0/P1/P2` (selected by destination peg), insert
  `bc_vservo_insert_v2b`, no fallback (`FALLBACK_CKPT=""`).
- Observation: 18-D vision state + 1024x7 cloud with ghost points.
- Active runtime extras:
  - `N_SAMPLES=5` — transport action = mean of 5 policy samples;
  - `INSERT_SAMPLES=1`;
  - `TRANSPORT_TERMINAL_SERVO=0.5`, `TRANSPORT_FORCE_CLOSE=1`,
    `TRANSPORT_LIFT_GATE=1`, `TRANSPORT_GRASP_CENTER_K=0.5`,
    `TRANSPORT_GRASP_SETTLE=4`, carry-high rule, grasp-height gate;
  - `VSERVO_STATIC_CACHE=1`, `VSERVO_VISION_SKIP=1`,
    `VSERVO_ICP_WORKERS=1`;
  - OSMesa; camera pose `results/camera_pose_latest.json`.

## Cheating checks

| check | result |
|---|---|
| GT object pose used to choose actions | **Pass.** Handoff uses `env._servo_err_raw` (vision); transport `done` is explicitly ignored as privileged (`realtime_general_newcam.py:316`). `transport_residual_servo.py` uses `_servo_err_raw`/`_disk_centre` (vision), FK, and gripper width; no GT object pose. |
| Retry/reset of failed episodes | **Pass.** `ep_done += 1` for every episode; `ep_success += int(success)` (`:370-379`); no retry. |
| Q-filter / best-of-N | **Pass.** No Q-filter; transport uses **mean of N=5** samples, not best-of-N (`_policy_act`). Insert N=1. |
| Fallback/rescue inflating SR | **Pass.** `FALLBACK_CKPT=""`, `FALLBACK_STEPS=0`; all 42 log episodes `switch=handoff_vision`, no `correction`/`transport_fail`. |
| Success predicate trivially true | **Pass.** `_is_seated` requires upright, at rest, target peg, correct support height, low velocity (`hanoi_env.py:390-407`). GT is used only for the evaluation predicate. |
| Seed cherry-picking | **Not proven.** Live seeds 896250+; the session log shows a continuous loop; no disjointness check against v2b/P0/P1/P2 training seeds yet. |
| Frozen n=200 protocol | **No.** 42-episode live session; Wilson 95% CI ~91.6–100%. |

## Caveats (not cheating, but they change the claim)

1. **Hybrid controller, not pure policy.** The 1.0 is transport specialists +
   v2b insert + hand-coded vision servos/guards. The learned policy alone is
   not the 1.0.
2. **Test-time averaging.** `N_SAMPLES=5` on transport is a 5x inference
   ensemble; it reduces variance and is not a single-sample policy result.
3. **Scoped task.** Single-move random disk/peg/level, no full Hanoi size
   rule; not the full 3-disk multi-move chain.
4. **Fast path active.** Cache/vision-skip flags change the observation
   schedule; the 1.0 is not the clean pipeline.
5. **Task conditioning.** Destination peg selects the transport specialist and
   the ghost target. Legitimate if the target is given; not target inference.
6. **Per-episode sim resets.** Does not test the real-world no-reset /
   terminal-safety constraint.
7. **Checkpoint identity.** Insert is `bc_vservo_insert_v2b`, not the v4
   expert-relative-loss checkpoint from `hanoi_vservo_errloss_1.0`.
8. **Page text is stale.** It mentions “insert ss4 -> rescue fallback”; the
   code has no fallback and uses v2b.

## Certification tests before trusting the 1.0

1. `SLOW=1` A/B (cache/skip off), fresh seed base, n>=200.
2. `POLICY_ONLY=1` ablation (N=1, transport servos off, fast flags off) to
   quantify the hand-coded contribution.
3. Full multi-move Hanoi with the size rule, frozen n=200.
4. Seed audit: 896250+ vs the v2b/P0/P1/P2 training seeds.
5. Continuous no-reset run if real-world deployment is the target.

## Verdict

No evidence of GT leakage, retry, Q-filter, or fallback. The 1.0 is honest for
this **42-episode live single-move session**, but it is a **hybrid vision
controller with 5x transport action averaging and the fast vision schedule**,
using the **v2b** insert checkpoint. It should not be presented as a pure-model,
full-Hanoi, n=200 1.0 until the A/B and policy-only ablations pass.
