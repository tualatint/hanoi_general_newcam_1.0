# Weights — v1.2 (single transport, sem7_sgd180)

Two DP3 checkpoints are attached to the **v1.2 GitHub Release** as release
assets (not Git LFS). They are the unmodified files from the running 8786
server.

| asset | path | bytes | sha256 |
|---|---|---|---|
| `bc_vservo_transport_sem7_sgd180.ckpt` | `runs/bc_vservo_transport_sem7_sgd180/best.ckpt` | 621,197,444 | `723f8941d7acee453ef2306a045d2cd0c86b44f8194c291495adbc508b0d541b` |
| `bc_vservo_insert_v2b.ckpt` | `runs/bc_vservo_insert_v2b/best.ckpt` | 621,195,460 | `a60a187ece19b81029226ee539a55a8378dde1c50fb0f193f23e2e3d9a2a104a` |

```bash
./download_weights.sh
# or:
sha256sum -c CHECKSUMS.sha256
```

Release history:
- **v1.0**: three transport specialists (P0/P1/P2) + insert v2b.
- **v1.1**: single `gen_v2_uni_sem7` transport + insert v2b.
- **v1.2**: single `sem7_sgd180` transport + jitter/slip interlocks + insert v2b.
