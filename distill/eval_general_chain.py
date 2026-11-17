#!/usr/bin/env python
"""Full general single-move chain: transport -> (handoff/correction) -> insert.

The transport -> insert switch is the same vision-only rule as the fixed-task
chain:
- handoff: |e_xy| < 5 mm and |e_z| < 4 mm for 5 consecutive steps, or the
  env's GT transport success (the handoff box);
- correction: after the transport budget, if the disk is still held and
  |e_xy| < 30 mm, |e_z| < 15 mm (the near-miss state).

Success = the general env's seated-ring predicate on the sampled target.
Failed insert states are saved with qpos/qvel/task for correction demos.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INC = os.environ.get(
    "HANOI_INC", "/media/j15/hdd2/RL-100/hanoi_flowrl_2026")
sys.path.insert(0, os.path.join(INC, "src"))
sys.path.insert(0, INC)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_100.common.ckpt_loader import build_policy_from_ckpt  # noqa: E402
from rl_100.common.pytorch_util import dict_apply  # noqa: E402
from general_env import GeneralVservoEnv  # noqa: E402
from camera_pose import apply_camera_pose  # noqa: E402
from transport_residual_servo import install as install_transport_servo  # noqa: E402

install_transport_servo(GeneralVservoEnv)


def obs_tensor(hist, device, n_ch=None, sem_class=True):
    stack = {
        "agent_pos": np.stack([o["agent_pos"] for o in hist])[None],
        "point_cloud": np.stack([o["point_cloud"] for o in hist])[None],
    }
    if n_ch is not None:
        stack["point_cloud"] = stack["point_cloud"][..., :int(n_ch)]
    if not sem_class and stack["point_cloud"].shape[-1] > 6:
        # policy trained with the binary ghost flag: remap the semantic class
        # (0 other / 1 ghost / 2 object) back to a ghost flag.
        pc = stack["point_cloud"]
        pc[..., 6] = (np.abs(pc[..., 6] - 1.0) < 0.5).astype(pc.dtype)
    return dict_apply(stack, lambda x: torch.from_numpy(x).to(
        device=device).float())


def act(policy, hist, device, n_samples=1, n_ch=None, sem_class=True):
    with torch.no_grad():
        if n_samples <= 1:
            return policy.predict_action(
                obs_tensor(hist, device, n_ch, sem_class))["action"][0].cpu().numpy()
        samples = [
            policy.predict_action(obs_tensor(hist, device, n_ch, sem_class))["action"][0]
            .cpu().numpy()
            for _ in range(n_samples)
        ]
        return np.mean(samples, axis=0)


def policy_semantic(cfg):
    """True when a checkpoint was trained on the 0/1/2 semantic class channel."""
    try:
        zp = cfg.get("dataset", {}).get("zarr_path", "")
        if isinstance(zp, (list, tuple)):
            return any("_sem7" in str(x) for x in zp)
        return "_sem7" in str(zp)
    except Exception:
        return False


def policy_pc_channels(cfg):
    """Point-cloud channel count expected by a checkpoint's policy."""
    try:
        pc = cfg["task"]["shape_meta"]["obs"]["point_cloud"]
        if isinstance(pc, dict):
            pc = pc.get("shape", pc)
        return int(pc[1])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport-ckpt", required=True)
    ap.add_argument("--p2-transport-ckpt", default="",
                    help="optional transport policy specialized for dst peg 2")
    ap.add_argument("--p0-transport-ckpt", default="",
                    help="optional transport policy specialized for dst peg 0")
    ap.add_argument("--p1-transport-ckpt", default="",
                    help="optional transport policy specialized for dst peg 1")
    ap.add_argument("--insert-ckpt", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed-base", type=int, default=830000)
    ap.add_argument("--transport-steps", type=int, default=90)
    ap.add_argument("--insert-steps", type=int, default=120)
    ap.add_argument("--fallback-insert-ckpt", default="",
                    help="insert policy used after the primary insert fails")
    ap.add_argument("--fallback-steps", type=int, default=180)
    ap.add_argument("--save-failures", default="",
                    help="save qpos/qvel/task of failed insert stages")
    ap.add_argument("--vision-only-handoff", action="store_true",
                    help="do not switch on the simulator's GT transport-success "
                         "flag; switch only on the vision near-hold rule or the "
                         "timeout correction gate (real-robot-faithful)")
    ap.add_argument("--handoff-xy-mm", type=float, default=5.0,
                    help="raw vision |ex|,|ey| threshold for the handoff")
    ap.add_argument("--handoff-z-mm", type=float, default=4.0,
                    help="raw vision |ez| threshold for the handoff")
    ap.add_argument("--handoff-hold", type=int, default=4,
                    help="consecutive near steps required to hand off")
    ap.add_argument("--no-grip-guard", dest="grip_guard",
                    action="store_false",
                    help="disable the transport grip safety interlock")
    ap.add_argument("--grip-guard", action="store_true", default=True,
                    help="never open the gripper while a disk is held and the "
                         "vision XY error is large (transport safety interlock; "
                         "proprioception + vision only)")
    ap.add_argument("--grip-guard-mm", type=float, default=20.0)
    ap.add_argument("--correction-xy-mm", type=float, default=30.0,
                    help="timeout near-miss gate: |e_xy| threshold")
    ap.add_argument("--correction-z-mm", type=float, default=15.0,
                    help="timeout near-miss gate: |e_z| threshold")
    ap.add_argument("--save-timeout", default="",
                    help="save held transport-timeout states for insert-correction "
                         "collection")
    ap.add_argument("--exec-steps", type=int, default=4,
                    help="actions executed per predicted chunk")
    ap.add_argument("--insert-exec-steps", type=int, default=0,
                    help="actions executed per insert chunk (0 = use "
                         "--exec-steps); shorter chunks give tighter "
                         "closed-loop correction near the peg")
    ap.add_argument("--n-samples", type=int, default=1,
                    help="average this many diffusion samples per chunk "
                         "(variance reduction near marginal grasps)")
    ap.add_argument("--insert-samples", type=int, default=0,
                    help="samples averaged in the insert stage (0 = use "
                         "--n-samples); the insert policy is the fragile "
                         "single-sample stage")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--camera-pose", default="",
                    help="saved free-camera JSON applied to cam_pc")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    pol_t, cfg_t = build_policy_from_ckpt(
        args.transport_ckpt, device, use_ema=True)
    pol_t2 = None
    if args.p2_transport_ckpt:
        pol_t2, cfg_t2 = build_policy_from_ckpt(
            args.p2_transport_ckpt, device, use_ema=True)
        pol_t2.eval()
    else:
        cfg_t2 = None
    pol_t0 = None
    if args.p0_transport_ckpt:
        pol_t0, cfg_t0 = build_policy_from_ckpt(
            args.p0_transport_ckpt, device, use_ema=True)
        pol_t0.eval()
    else:
        cfg_t0 = None
    pol_t1 = None
    if args.p1_transport_ckpt:
        pol_t1, cfg_t1 = build_policy_from_ckpt(
            args.p1_transport_ckpt, device, use_ema=True)
        pol_t1.eval()
    else:
        cfg_t1 = None
    spec_t = {0: pol_t0, 1: pol_t1, 2: pol_t2}
    spec_ch = {0: policy_pc_channels(cfg_t0),
               1: policy_pc_channels(cfg_t1),
               2: policy_pc_channels(cfg_t2)}
    spec_sem = {0: policy_semantic(cfg_t0),
                1: policy_semantic(cfg_t1),
                2: policy_semantic(cfg_t2)}
    n_ch_t = policy_pc_channels(cfg_t)
    sem_t = policy_semantic(cfg_t)
    pol_i, cfg_i = build_policy_from_ckpt(
        args.insert_ckpt, device, use_ema=True)
    n_ch_i = policy_pc_channels(cfg_i)
    pol_fb = None
    if args.fallback_insert_ckpt:
        pol_fb, cfg_fb = build_policy_from_ckpt(
            args.fallback_insert_ckpt, device, use_ema=True)
        pol_fb.eval()
        n_ch_fb = policy_pc_channels(cfg_fb)
    else:
        n_ch_fb = None
    pol_t.eval()
    pol_i.eval()
    n_obs = int(pol_t.n_obs_steps)
    env = GeneralVservoEnv(
        max_episode_steps=(args.transport_steps + args.insert_steps
                           + args.fallback_steps + 20),
        n_points=1024, pc_resolution=160, seed=args.seed_base,
        min_dist=5, goal_peg=2, reward_mode="dense", skill_mode=True,
        skill_stage="transport", use_contact_obs=False,
        insert_template_path=None, ghost_z_mode="handoff")
    if args.camera_pose:
        apply_camera_pose(env, args.camera_pose)
    rows = []
    for i in range(args.n):
        seed = args.seed_base + i
        env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.skill_stage = "transport"
        env.ghost_z_mode = "handoff"
        env.reset_integral()
        obs = env.reset()
        task = tuple(int(x) for x in env.task)
        init = tuple(int(x) for x in env._initial_state)
        hist = deque([obs] * n_obs, maxlen=n_obs)
        tstep = 0
        switched = False
        reason = ""
        t_done = False
        info = {}
        near_hold = 0
        gt_handoff = False
        while tstep < args.transport_steps and not switched:
            pol_tr = pol_t
            ch_tr = n_ch_t
            sem_tr = sem_t
            spec = spec_t.get(int(task[2]))
            if spec is not None:
                pol_tr = spec
                ch_tr = spec_ch.get(int(task[2])) or n_ch_t
                sem_tr = spec_sem.get(int(task[2]), False)
            acts = act(pol_tr, hist, device, args.n_samples, ch_tr, sem_tr)
            for a in acts[:args.exec_steps]:
                if (args.grip_guard and env._servo_valid > 0
                        and 0.030 < env._get_gripper_width() < 0.082):
                    e0 = env._servo_err_raw
                    if np.hypot(float(e0[0]), float(e0[1])) > \
                            args.grip_guard_mm * 1e-3:
                        a = np.asarray(a).copy()
                        a[3] = 1.0   # keep holding the disk during the carry
                obs, _r, t_done, info = env.step(a)
                hist.append(obs)
                tstep += 1
                # use the *raw* vision error: the smoothed error lags the
                # moving disk and lets the handoff fire with a larger physical
                # offset than intended.
                e = env._servo_err_raw
                valid = env._servo_valid > 0
                near = (valid
                        and abs(float(e[0])) < args.handoff_xy_mm * 1e-3
                        and abs(float(e[1])) < args.handoff_xy_mm * 1e-3
                        and abs(float(e[2])) < args.handoff_z_mm * 1e-3)
                near_hold = near_hold + 1 if near else 0
                if near_hold >= args.handoff_hold:
                    switched, reason = True, "handoff_vision"
                    handoff_err = (e[:3] * 1e3).round(2).tolist()
                    break
                if t_done and not args.vision_only_handoff:
                    switched, reason = True, "handoff_gt"
                    break
                if t_done:
                    gt_handoff = True
                if tstep >= args.transport_steps:
                    held = env._get_gripper_width() < 0.085
                    if (valid and held
                            and float(np.hypot(e[0], e[1]))
                            < args.correction_xy_mm * 1e-3
                            and abs(float(e[2]))
                            < args.correction_z_mm * 1e-3):
                        switched, reason = True, "correction"
                        handoff_err = (e[:3] * 1e3).round(2).tolist()
                    elif args.save_timeout and held:
                        os.makedirs(args.save_timeout, exist_ok=True)
                        np.savez(
                            os.path.join(args.save_timeout,
                                         f"general_chain_entry_{seed}.npz"),
                            qpos=env.data.qpos.copy(),
                            qvel=env.data.qvel.copy(),
                            seed=np.int64(seed),
                            task=np.asarray(task, dtype=np.int64),
                            initial_state=np.asarray(init, dtype=np.int64),
                            post_axis=(env._post_axis.copy()
                                       if env._post_axis is not None
                                       else np.full(2, np.nan)),
                            table_z=np.float64(env._table_z
                                               if env._table_z is not None
                                               else np.nan),
                            post_z_min=np.float64(env._post_z_min
                                                  if env._post_z_min is not None
                                                  else np.nan),
                            stage=np.array("insert_entry"),
                peg_offsets=(env._peg_offsets
                             if getattr(env, "_peg_offsets", None)
                             is not None else np.zeros((3, 2))))
                    break
            # In vision-only mode the simulator's done flag must not end the
            # episode: it is privileged state that the deployed system does
            # not have.  The GT transport predicate ends the episode on the
            # *entry* into the handoff box, before the vision estimate (which
            # lags the moving disk by a few mm) has converged, so breaking
            # here turns physically-correct handoffs into transport_fail.
            # Keep controlling until the vision handoff fires or the budget
            # expires.  This only removes privileged information; the policy
            # observation is unchanged.
            if switched or (t_done and not args.vision_only_handoff):
                break
        if not switched:
            if args.save_failures:
                os.makedirs(args.save_failures, exist_ok=True)
                np.savez(
                    os.path.join(args.save_failures,
                                 f"general_chain_transport_{seed}.npz"),
                    qpos=env.data.qpos.copy(),
                    qvel=env.data.qvel.copy(),
                    seed=np.int64(seed),
                    task=np.asarray(task, dtype=np.int64),
                    initial_state=np.asarray(init, dtype=np.int64),
                    post_axis=(env._post_axis.copy()
                               if env._post_axis is not None
                               else np.full(2, np.nan)),
                    table_z=np.float64(env._table_z
                                       if env._table_z is not None
                                       else np.nan),
                    post_z_min=np.float64(env._post_z_min
                                          if env._post_z_min is not None
                                          else np.nan),
                    stage=np.array("transport_fail"),
                    peg_offsets=(env._peg_offsets
                                 if getattr(env, "_peg_offsets", None)
                                 is not None else np.zeros((3, 2))))
            rows.append({"seed": seed, "task": list(task), "success": False,
                         "switch": "transport_fail",
                         "gt_handoff": gt_handoff,
                         "transport_steps": tstep, "insert_steps": 0})
            print(f"[gen-chain {i+1}/{args.n}] seed={seed} task={task} "
                  f"transport_fail", flush=True)
            continue

        # switch to the seated ghost and reset the integral
        env.ghost_z_mode = "seated"
        env.skill_stage = "insert"
        env.reset_integral()
        env._centered_once = False
        env._phi_prev = None
        obs = env._get_obs()
        entry_qpos = env.data.qpos.copy()
        entry_qvel = env.data.qvel.copy()
        entry_post = (env._post_axis.copy() if env._post_axis is not None
                      else np.full(2, np.nan))
        entry_table = (float(env._table_z) if env._table_z is not None
                       else float("nan"))
        entry_zmin = (float(env._post_z_min)
                      if env._post_z_min is not None else float("nan"))
        hist = deque([obs] * n_obs, maxlen=n_obs)
        ins_samples = (args.insert_samples if args.insert_samples > 0
                       else args.n_samples)
        ins_exec = (args.insert_exec_steps if args.insert_exec_steps > 0
                    else args.exec_steps)
        istep = 0
        i_done = False
        i_info = {}
        while istep < args.insert_steps and not i_done:
            acts = act(pol_i, hist, device, ins_samples, n_ch_i, False)
            for a in acts[:ins_exec]:
                obs, _r, i_done, i_info = env.step(a)
                hist.append(obs)
                istep += 1
                if i_done:
                    break
        succ = bool(i_info.get("success", False))
        fb_used = False
        if not succ and pol_fb is not None:
            obs = env._get_obs()
            hist = deque([obs] * n_obs, maxlen=n_obs)
            fstep = 0
            f_done = False
            f_info = {}
            while fstep < args.fallback_steps and not f_done:
                acts = act(pol_fb, hist, device, ins_samples, n_ch_fb, False)
                for a in acts[:ins_exec]:
                    obs, _r, f_done, f_info = env.step(a)
                    hist.append(obs)
                    fstep += 1
                    istep += 1
                    if f_done:
                        break
            if f_info:
                i_info = f_info
            succ = bool(i_info.get("success", False))
            fb_used = True
        e = env._servo_err_raw
        if args.save_failures and not succ:
            os.makedirs(args.save_failures, exist_ok=True)
            # Entry state (the near-miss handoff the insert policy had to
            # recover from) -> correction demos.
            np.savez(
                os.path.join(args.save_failures,
                             f"general_chain_entry_{seed}.npz"),
                qpos=entry_qpos, qvel=entry_qvel,
                seed=np.int64(seed),
                task=np.asarray(task, dtype=np.int64),
                initial_state=np.asarray(init, dtype=np.int64),
                post_axis=np.asarray(entry_post, dtype=np.float64),
                table_z=np.float64(entry_table),
                post_z_min=np.float64(entry_zmin),
                stage=np.array("insert_entry"),
                peg_offsets=(env._peg_offsets
                             if getattr(env, "_peg_offsets", None)
                             is not None else np.zeros((3, 2))))
            # Final insert state (hover / jam) -> rescue demos.
            np.savez(
                os.path.join(args.save_failures,
                             f"general_chain_{seed}.npz"),
                qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
                seed=np.int64(seed),
                task=np.asarray(task, dtype=np.int64),
                initial_state=np.asarray(init, dtype=np.int64),
                post_axis=(env._post_axis.copy()
                           if env._post_axis is not None
                           else np.full(2, np.nan)),
                table_z=np.float64(env._table_z
                                   if env._table_z is not None
                                   else float("nan")),
                post_z_min=np.float64(env._post_z_min
                                      if env._post_z_min is not None
                                      else float("nan")),
                stage=np.array("insert"),
                servo_err=np.asarray(e, dtype=np.float32))
        rows.append({"seed": seed, "task": list(task), "success": succ,
                     "switch": reason, "fallback": fb_used,
                     "gt_handoff": gt_handoff,
                     "handoff_err_mm": handoff_err,
                     "transport_steps": tstep, "insert_steps": istep,
                     "final_err_mm": (e[:3] * 1e3).round(2).tolist()})
        print(f"[gen-chain {i+1}/{args.n}] seed={seed} task={task} "
              f"switch={reason} t={tstep} i={istep} success={succ}",
              flush=True)
    k = sum(r["success"] for r in rows)
    print(f"[general-chain] {k}/{len(rows)} = {k/max(1,len(rows)):.4f}")
    for reason in sorted({r["switch"] for r in rows}):
        sub = [r for r in rows if r["switch"] == reason]
        print(f"  {reason}: {sum(r['success'] for r in sub)}/{len(sub)}")
    for key, idx in (("disk", 0), ("dst", 2), ("level", 3)):
        groups = {}
        for r in rows:
            g = r["task"][idx]
            a, b = groups.get(g, (0, 0))
            groups[g] = (a + int(r["success"]), b + 1)
        print(f"  by {key}: " + "  ".join(
            f"{g}:{a}/{b}" for g, (a, b) in sorted(groups.items())))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"transport_ckpt": args.transport_ckpt,
                       "insert_ckpt": args.insert_ckpt,
                       "n": len(rows), "success": k,
                       "sr": k / max(1, len(rows)), "rows": rows}, f)
        print(f"wrote {args.out}")
    env.close()


if __name__ == "__main__":
    main()
