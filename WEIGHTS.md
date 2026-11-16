# Weights — v1.1 (single transport)

Two DP3 checkpoints are attached to the **v1.1 GitHub Release** as release
assets (not Git LFS). They are the unmodified files from the running 8786
server.

| asset | path | bytes | sha256 |
|---|---|---|---|
| `bc_vservo_transport_gen_v2_uni_sem7.ckpt` | `runs/bc_vservo_transport_gen_v2_uni_sem7/best.ckpt` | 621,196,164 | `8aa5953b2c6199c676928553a2bace86cfe5a87f19568e81c489a0cc7ae518d8` |
| `bc_vservo_insert_v2b.ckpt` | `runs/bc_vservo_insert_v2b/best.ckpt` | 621,195,460 | `a60a187ece19b81029226ee539a55a8378dde1c50fb0f193f23e2e3d9a2a104a` |

```bash
./download_weights.sh
# or:
sha256sum -c CHECKSUMS.sha256
```

The **v1.0 release** remains available with the older three-transport stack
(`bc_vservo_transport_bal_v2_p0/p1/p2`) plus the same insert v2b.
