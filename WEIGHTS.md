# Weights

Four DP3 checkpoints are attached to the **v1.0 GitHub Release** as release
assets (not Git LFS). They are the unmodified files from the audited 8786
server.

| asset | path | bytes | sha256 |
|---|---:|---|
| `bc_vservo_insert_v2b.ckpt` | `runs/bc_vservo_insert_v2b/best.ckpt` | 621,195,460 | `a60a187ece19b81029226ee539a55a8378dde1c50fb0f193f23e2e3d9a2a104a` |
| `bc_vservo_transport_bal_v2_p0.ckpt` | `runs/bc_vservo_transport_bal_v2_p0/best.ckpt` | 621,196,484 | `50f530bc4370ba9b52bd32b31988d8239a43524413b9c91ebe46f0fcde6ec3e3` |
| `bc_vservo_transport_bal_v2_p1.ckpt` | `runs/bc_vservo_transport_bal_v2_p1/best.ckpt` | 621,196,676 | `316a31af176a7fea233cd5fa7e208ae66fada8e072f8f42294c92bcae7171809` |
| `bc_vservo_transport_bal_v2_p2.ckpt` | `runs/bc_vservo_transport_bal_v2_p2/best.ckpt` | 621,196,740 | `bc8cd1c68e0ea8b54139462949589db320130d9b41d1d55eddad9322a4ebfa43` |

```bash
./download_weights.sh
# or:
sha256sum -c CHECKSUMS.sha256
```

The insert checkpoint is **v2b**, not the v4 expert-relative-loss checkpoint
from `hanoi_vservo_errloss_1.0`.
