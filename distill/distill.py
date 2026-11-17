#!/usr/bin/env python
"""One-step DDIM distillation for the hanoi transport / insert DP3 policies.

Follows the RL-100 teacher/student same-noise MSE pattern
(ref_official/RL-100/rl_100/policy/rl100_3d.py::compute_flow_distill_loss),
adapted to the local DDIM sample-prediction policy.

Teacher: 10-step DDIM from noise -> x0 target.
Student: same noise, timestep t=90 (the teacher's first timestep), predicts x0.
Only the student UNet trains; encoder + normalizer are frozen.

The saved student cfg sets num_inference_steps=1 and steps_offset=90, so the
existing policy/server predict_action uses one step at t=90 and returns the
student x0.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = os.environ.get(
    "HANOI_REPO", "/media/j15/hdd2/RL-100/hanoi_vservo_2026")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from rl_100.common.ckpt_loader import build_policy_from_ckpt  # noqa: E402
from rl_100.common.pytorch_util import dict_apply  # noqa: E402
from rl_100.dataset.arx5_dataset import Arx5GraspDataset  # noqa: E402


def _to_device(batch, device):
    return dict_apply(batch, lambda x: x.to(device))


def _ema_update(ema_policy, student_policy, decay):
    with torch.no_grad():
        for pe, ps in zip(ema_policy.parameters(), student_policy.parameters()):
            pe.mul_(decay).add_(ps, alpha=1.0 - decay)
        for be, bs in zip(ema_policy.buffers(), student_policy.buffers()):
            if be.shape == bs.shape:
                be.copy_(bs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="max optimizer steps per epoch (0 = full loader)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--max-episodes", type=int, default=500)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print(f"[distill] teacher={args.teacher}", flush=True)
    teacher, cfg = build_policy_from_ckpt(args.teacher, device, use_ema=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.obs_encoder.eval()

    # Student = deepcopy of the teacher; only the UNet trains.
    student = copy.deepcopy(teacher)
    student.model.requires_grad_(True)
    student.model.train()
    student.obs_encoder.eval()
    for p in student.obs_encoder.parameters():
        p.requires_grad_(False)
    for p in student.normalizer.parameters():
        p.requires_grad_(False)
    ema = copy.deepcopy(student)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    # Teacher rollout settings: 10 DDIM steps [90..0]; student sees t=90.
    teacher.noise_scheduler.set_timesteps(teacher.num_inference_steps)
    t_start = int(teacher.noise_scheduler.timesteps[0].item())
    print(f"[distill] teacher steps={teacher.num_inference_steps} "
          f"t_start={t_start}", flush=True)

    dataset = Arx5GraspDataset(
        zarr_path=list(args.data),
        horizon=5,
        pad_before=1,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=args.max_episodes,
        obs_keys=["state", "action", "point_cloud"],
    )
    print(f"[distill] samples={len(dataset)}", flush=True)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, persistent_workers=False,
        pin_memory=True, drop_last=True)

    optimizer = torch.optim.AdamW(
        student.model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.95, 0.999))
    total_steps = args.epochs * (
        args.max_steps if args.max_steps > 0 else len(loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=args.lr * 0.05)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    global_step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        run_loss = 0.0
        n = 0
        for step, batch in enumerate(loader):
            if args.max_steps and step >= args.max_steps:
                break
            batch = _to_device(batch, device)
            with torch.no_grad():
                cond_data, cond_mask, local_cond, global_cond, _ = \
                    teacher.obs2feature(batch["obs"], fix_encoder=True)
                noise = torch.randn_like(cond_data)
                traj = noise.clone()
                for t in teacher.noise_scheduler.timesteps:
                    traj[cond_mask] = cond_data[cond_mask]
                    out = teacher.model(
                        sample=traj, timestep=t,
                        local_cond=local_cond, global_cond=global_cond)
                    traj = teacher.noise_scheduler.step(
                        out, t, traj).prev_sample
                target = traj.detach()

            B = target.shape[0]
            ts = torch.full((B,), t_start, dtype=torch.long, device=device)
            pred = student.model(
                sample=noise, timestep=ts,
                local_cond=None, global_cond=global_cond)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                student.model.parameters(), args.grad_clip)
            optimizer.step()
            sched.step()
            _ema_update(ema, student, args.ema_decay)
            global_step += 1
            run_loss += float(loss)
            n += 1
            if global_step % args.log_every == 0:
                print(f"[distill] ep {epoch + 1}/{args.epochs} "
                      f"step {step + 1} loss {run_loss / n:.6f} "
                      f"lr {sched.get_last_lr()[0]:.2e} "
                      f"({time.time() - t0:.0f}s)", flush=True)

        print(f"[distill] epoch {epoch + 1} mean loss {run_loss / max(n, 1):.6f}",
              flush=True)
        # Save last + EMA student every epoch.
        cfg_out = copy.deepcopy(cfg)
        cfg_out["policy"]["num_inference_steps"] = 1
        cfg_out["policy"]["noise_scheduler"]["steps_offset"] = t_start
        payload = {
            "cfg": cfg_out,
            "state_dicts": {
                "model": student.state_dict(),
                "ema_model": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            "pickles": {"global_step": global_step, "epoch": epoch},
        }
        outdir = os.path.dirname(os.path.abspath(args.out))
        last = os.path.join(outdir, "last.ckpt")
        torch.save(payload, last)
        torch.save(payload, args.out)
        print(f"[distill] saved {args.out} (last={last})", flush=True)

    print("[distill] done", flush=True)


if __name__ == "__main__":
    main()
