# Results — 8786 live session

Source: `results/webserver_8786_session.log` (the audited server's stdout).

| metric | value |
|---|---:|
| episodes | **42** |
| success | **42** |
| SR | **100%** |
| Wilson 95% CI | 91.6% – 100% |
| mean steps | 63.0 |
| switch types | `handoff_vision` 42 |
| failures / fallbacks / corrections | 0 / 0 / 0 |

Task coverage:

| level | episodes |
|---|---:|
| 0 | 21 |
| 1 | 14 |
| 2 | 7 |

| disk | episodes |
|---|---:|
| disk_s | 22 |
| disk_m | 16 |
| disk_l | 4 |

Summary JSON: `results/session_summary.json`.

This is a **live session**, not a frozen n=200 holdout. The 42 episodes are a
continuous seed loop from 896250; every episode is counted, and there is no
retry, fallback, Q-filter, or best-of-N selection. Transport actions are the
mean of 5 policy samples (`N_SAMPLES=5`); insert is single-sample.
