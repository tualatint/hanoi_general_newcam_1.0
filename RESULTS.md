# Results

## v1.2 — single transport `sem7_sgd180` + jitter/slip interlocks

Source: `results/webserver_8786_jitter_session.log` (audited server stdout).

| metric | value |
|---|---:|
| episodes at capture | **120** |
| success | **120** |
| SR | **100%** |
| Wilson 95% CI | 96.9% – 100% |
| mean steps | 57.0 |
| switch types | `handoff_vision` 120 |
| failures / fallbacks / corrections | 0 / 0 / 0 |

| level | episodes |
|---|---:|
| 0 | 48 |
| 1 | 47 |
| 2 | 25 |

| disk | episodes |
|---|---:|
| disk_s | 70 |
| disk_m | 29 |
| disk_l | 21 |

Summary JSON: `results/jitter_summary.json`.

This is a **live session**, not a frozen n=200 holdout. Every episode is
counted; no retry, fallback, Q-filter, or best-of-N. Transport actions are the
mean of 5 policy samples; insert is single-sample.

## v1.1 — single transport `gen_v2_uni_sem7`

Source: `results/webserver_8786_sem7_session.log`.

| metric | value |
|---|---:|
| episodes at capture | 16 |
| success | 16 |
| SR | 100% |
| Wilson 95% CI | 80.6% – 100% |
| mean steps | 57.1 |
| switch types | `handoff_vision` 16 |

Levels 0:8, 1:6, 2:2; disks s:12, m:4. Preserved in the v1.1 GitHub Release.

## v1.0 — three-transport legacy

Source: `results/webserver_8786_session.log`.

| metric | value |
|---|---:|
| episodes | 42 |
| success | 42 |
| SR | 100% |
| Wilson 95% CI | 91.6% – 100% |
| mean steps | 63.0 |

Levels 0 (21), 1 (14), 2 (7). Preserved in the v1.0 GitHub Release.
