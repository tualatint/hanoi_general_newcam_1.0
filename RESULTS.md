# Results

## v1.1 — single transport (`gen_v2_uni_sem7`)

Source: `results/webserver_8786_sem7_session.log` (audited server stdout).

| metric | value |
|---|---:|
| episodes at capture | **16** |
| success | **16** |
| SR | **100%** |
| Wilson 95% CI | 80.6% – 100% |
| mean steps | 57.1 |
| switch types | `handoff_vision` 16 |
| failures / fallbacks / corrections | 0 / 0 / 0 |

| level | episodes |
|---|---:|
| 0 | 8 |
| 1 | 6 |
| 2 | 2 |

| disk | episodes |
|---|---:|
| disk_s | 12 |
| disk_m | 4 |
| disk_l | 0 |

Summary JSON: `results/sem7_summary.json`.

This is a **live session** with a small sample; every episode is counted and
there is no retry, fallback, Q-filter, or best-of-N. Transport actions are the
mean of 5 policy samples; insert is single-sample.

## v1.0 — three-transport legacy

Source: `results/webserver_8786_session.log`.

| metric | value |
|---|---:|
| episodes | 42 |
| success | 42 |
| SR | 100% |
| Wilson 95% CI | 91.6% – 100% |
| mean steps | 63.0 |
| switch types | `handoff_vision` 42 |

Levels: 0 (21), 1 (14), 2 (7). This version used the P0/P1/P2 transport
specialists and is preserved in the v1.0 GitHub Release.
