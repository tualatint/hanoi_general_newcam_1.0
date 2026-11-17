#!/usr/bin/env python
"""Latency + action-match benchmark: teacher 10-step vs distilled 1-step."""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

REPO = os.environ.get(
    "HANOI_REPO", "/media/j15/hdd2/RL-100/hanoi_vservo_2026")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from rl_100.common.ckpt_loader import build_policy_from_ckpt  # noqa: E402
from rl_100.common.pytorch_util import dict_apply  # noqa: E402
from rl_100.dataset.arx5_dataset import Arx5GraspDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--max-episodes", type=int, default=20)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    teacher, _ = build_policy_from_ckpt(args.teacher, dev, use_ema=True)
    student, scfg = build_policy_from_ckpt(args.student, dev, use_ema=True)
    print(f"teacher steps={teacher.num_inference_steps} "
          f"student steps={student.num_inference_steps} "
          f"steps_offset={scfg['policy']['noise_scheduler']['steps_offset']}")
    ds = Arx5GraspDataset(
        zarr_path=list(args.data), horizon=5, pad_before=1, pad_after=0,
        seed=42, val_ratio=0.0, max_train_episodes=args.max_episodes,
        obs_keys=["state", "action", "point_cloud"])
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=1)))
    obs = dict_apply(batch["obs"], lambda x: x.to(dev))
    for name, pol in (("teacher", teacher), ("student", student)):
        for _ in range(3):
            pol.predict_action(obs)
        torch.cuda.synchronize()
        t0 = time.time()
        out = None
        for _ in range(args.iters):
            out = pol.predict_action(obs)["action"]
        torch.cuda.synchronize()
        ms = (time.time() - t0) / args.iters * 1000.0
        print(f"{name}: {ms:.2f} ms/chunk  action_mean={out.abs().mean().item():.4f}")
        if name == "teacher":
            teacher_action = out.clone()
    mse = torch.mean((out - teacher_action) ** 2).item()
    print(f"action MSE student vs teacher: {mse:.6f}")


if __name__ == "__main__":
    main()
