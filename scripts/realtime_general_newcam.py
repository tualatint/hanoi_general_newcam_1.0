#!/usr/bin/env python
"""Realtime webserver for the general (random disk -> random peg) chain.

transport ss -> vision/GT handoff or correction switch -> insert ss4 ->
rescue fallback.  Vision-only 18-D observation; the page overlays the
magenta ghost goal and the cyan vision-estimated disk centre.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import time
import traceback
from collections import deque

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INC = os.environ.get("HANOI_INC", ROOT)
sys.path.insert(0, os.path.join(INC, "src"))
sys.path.insert(0, INC)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from flask import Flask, Response, jsonify, request  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from rl_100.common.ckpt_loader import build_policy_from_ckpt  # noqa: E402
from rl_100.common.pytorch_util import dict_apply  # noqa: E402
from general_env import GeneralVservoEnv  # noqa: E402
from transport_residual_servo import install as install_transport_servo  # noqa: E402

install_transport_servo(GeneralVservoEnv)

app = Flask(__name__)

TRANSPORT_CKPT = ""
INSERT_CKPT = os.environ.get(
    "INSERT_CKPT",
    os.path.join(ROOT, "runs", "bc_vservo_insert_v2b", "best.ckpt"))
# Per-destination specialists (selected by the task's destination peg).
FALLBACK_CKPT = ""
P0_TRANSPORT_CKPT = os.environ.get(
    "P0_CKPT",
    os.path.join(ROOT, "runs", "bc_vservo_transport_bal_v2_p0",
                 "best.ckpt"))
P1_TRANSPORT_CKPT = os.environ.get(
    "P1_CKPT",
    os.path.join(ROOT, "runs", "bc_vservo_transport_bal_v2_p1",
                 "best.ckpt"))
P2_TRANSPORT_CKPT = os.environ.get(
    "P2_CKPT",
    os.path.join(ROOT, "runs", "bc_vservo_transport_bal_v2_p2",
                 "best.ckpt"))

TRANSPORT_STEPS = 120
INSERT_STEPS = 120
FALLBACK_STEPS = 0
MAX_STEPS = TRANSPORT_STEPS + INSERT_STEPS + 20
HANDOFF_XY_MM = 5.0
HANDOFF_Z_MM = 4.0
HANDOFF_HOLD = 4
EXEC_STEPS = int(os.environ.get("EXEC_STEPS", "4"))
INSERT_EXEC_STEPS = int(os.environ.get("INSERT_EXEC_STEPS", "2"))

_lock = threading.Lock()
_last_frame = None
_frame_seq = 0
_rollout_thread = None
_stop_event = None
_device = "cuda:0"
_policies = {}
_status = {
    "running": False, "loop": False, "mode": "move",
    "stage": "-", "seed": None, "episode": 0, "ep_done": 0,
    "ep_success": 0, "sr": None, "step": 0, "max_steps": MAX_STEPS,
    "success": None, "last_result": None, "last_steps": None,
    "vision_valid": None, "vision_err_mm": None, "vision_tilt_deg": None,
    "handoff": None, "switch": None, "fallback": False,
    "task": None, "message": "idle", "ckpt": None,
    "n_points": 1024, "video_size": 512,
}


def _update(**kw):
    with _lock:
        _status.update(kw)


def _snapshot():
    with _lock:
        return dict(_status)


def _load_policy(path):
    path = os.path.abspath(path)
    if path in _policies:
        return _policies[path][0]
    policy, cfg = build_policy_from_ckpt(path, _device, use_ema=True)
    policy.eval()
    _policies[path] = (policy, cfg)
    print(f"[server] loaded {path}", flush=True)
    return policy


def _project(points, env, width, height):
    if env._pc_renderer is None or points is None or len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    cam_id = env._cam_pc_id
    R_x180 = np.array([[1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0, 0.0, -1.0]], dtype=np.float64)
    R_cw = env.data.cam_xmat[cam_id].reshape(3, 3) @ R_x180
    t_cw = env.data.cam_xpos[cam_id]
    p_cam = (np.asarray(points, dtype=np.float64) - t_cw) @ R_cw
    f = height / (2.0 * np.tan(
        np.radians(env.model.cam_fovy[cam_id]) / 2.0))
    z = p_cam[:, 2]
    ok = z > 1e-6
    u = np.full(len(points), -1.0)
    v = np.full(len(points), -1.0)
    u[ok] = f * p_cam[ok, 0] / z[ok] + width / 2.0
    v[ok] = f * p_cam[ok, 1] / z[ok] + height / 2.0
    return np.stack([u, v], axis=-1)


def _render_overlay(env, renderer, width, height):
    renderer.update_scene(env.data, camera=env._cam_pc_id)
    pil = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(pil)
    uv = _project(env._ghost_points(), env, width, height)
    for u, v in uv:
        if 0 <= u < width and 0 <= v < height:
            draw.ellipse([u - 2, v - 2, u + 2, v + 2],
                         outline=(255, 0, 255), fill=(255, 0, 255))
    if env._disk_centre is not None:
        uv = _project(env._disk_centre[None], env, width, height)
        if len(uv) and 0 <= uv[0, 0] < width and 0 <= uv[0, 1] < height:
            u, v = float(uv[0, 0]), float(uv[0, 1])
            draw.line([u - 7, v, u + 7, v], fill=(0, 255, 255), width=2)
            draw.line([u, v - 7, u, v + 7], fill=(0, 255, 255), width=2)
    return pil


def _encode(pil):
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=82)
    return buf.getvalue()


def _vision_status(env):
    e = np.asarray(env._servo_err, dtype=np.float64)
    return {
        "vision_valid": bool(env._servo_valid > 0),
        "vision_err_mm": [round(float(e[0]) * 1e3, 2),
                          round(float(e[1]) * 1e3, 2),
                          round(float(e[2]) * 1e3, 2)],
        "vision_tilt_deg": round(float(np.degrees(np.hypot(e[3], e[4]))), 2),
    }


def _policy_act(policy, obs_hist, n=None):
    stack = {
        "agent_pos": np.stack([o["agent_pos"] for o in obs_hist])[None],
        "point_cloud": np.stack([o["point_cloud"] for o in obs_hist])[None],
    }
    od = dict_apply(stack, lambda x: torch.from_numpy(x).to(
        device=_device).float())
    if n is None:
        n = int(os.environ.get("N_SAMPLES", "1"))
    with torch.no_grad():
        if n <= 1:
            return policy.predict_action(od)["action"][0].cpu().numpy()
        return np.mean(
            [policy.predict_action(od)["action"][0].cpu().numpy()
             for _ in range(n)], axis=0)


def _task_text(env):
    if env.task is None:
        return None
    d, src, dst, level = env.task
    return {
        "disk": env.vision_disk, "disk_idx": int(d),
        "src": int(src), "dst": int(dst), "level": int(level),
    }


def _rollout_worker(seed, max_steps, video_size, loop, stop_event):
    global _last_frame, _frame_seq
    try:
        _update(running=True, loop=bool(loop), mode="move", seed=int(seed),
                max_steps=int(max_steps), video_size=int(video_size),
                episode=0, ep_done=0, ep_success=0, sr=None, step=0,
                success=None, last_result=None, last_steps=None,
                stage="-", handoff=None, switch=None, fallback=False,
                task=None, message="loading policies", ckpt=INSERT_CKPT)
        pol_t = _load_policy(TRANSPORT_CKPT) if TRANSPORT_CKPT else None
        pol_i = _load_policy(INSERT_CKPT)
        pol_fb = _load_policy(FALLBACK_CKPT) if FALLBACK_CKPT else None
        pol_t2 = _load_policy(P2_TRANSPORT_CKPT) if P2_TRANSPORT_CKPT else None
        pol_t0 = _load_policy(P0_TRANSPORT_CKPT) if P0_TRANSPORT_CKPT else None
        pol_t1 = _load_policy(P1_TRANSPORT_CKPT) if P1_TRANSPORT_CKPT else None
        spec_t = {}
        if pol_t0 is not None:
            spec_t[0] = pol_t0
        if pol_t1 is not None:
            spec_t[1] = pol_t1
        if pol_t2 is not None:
            spec_t[2] = pol_t2
        n_obs = int((pol_t or pol_i).n_obs_steps)

        env = GeneralVservoEnv(
            max_episode_steps=int(max_steps), n_points=1024,
            pc_resolution=160, seed=int(seed), min_dist=5, goal_peg=2,
            reward_mode="dense", skill_mode=True,
            skill_stage="transport", use_contact_obs=False,
            insert_template_path=None, ghost_z_mode="handoff")
        renderer = mujoco.Renderer(env.model, height=int(video_size),
                                   width=int(video_size))

        def push_frame():
            global _last_frame, _frame_seq
            pil = _render_overlay(env, renderer, int(video_size),
                                  int(video_size))
            _last_frame = _encode(pil)
            _frame_seq += 1

        ep_done = ep_success = 0
        ep_index = 0
        _update(message="running")
        while not stop_event.is_set():
            ep_seed = int(seed) + (ep_index if loop else 0)
            ep_index += 1 if loop else 0
            env.seed(ep_seed)
            np.random.seed(ep_seed)
            torch.manual_seed(ep_seed)
            env.skill_stage = "transport"
            env.ghost_z_mode = "handoff"
            env.reset_integral()
            obs = env.reset()
            task = _task_text(env)
            hist = deque([obs] * n_obs, maxlen=n_obs)
            stage = "transport"
            switch = None
            push_frame()
            _update(episode=ep_done + 1, step=0, stage=stage,
                    success=None, last_result=None, message="running",
                    handoff=None, switch=None, fallback=False, task=task,
                    **_vision_status(env))
            step = 0
            done = False
            info = {}
            transport_steps = 0
            insert_steps = 0
            fallback_steps = 0
            switched = False
            handoff_hold = 0
            while step < int(max_steps) and not stop_event.is_set():
                if stage == "insert":
                    policy = pol_i
                elif (env.task is not None
                      and spec_t.get(int(env.task[2])) is not None):
                    policy = spec_t[int(env.task[2])]
                else:
                    policy = pol_t
                acts = _policy_act(
                    policy, hist,
                    n=(int(os.environ.get("INSERT_SAMPLES", "1"))
                       if stage == "insert" else None))
                n_exec = (INSERT_EXEC_STEPS if stage == "insert"
                          else EXEC_STEPS)
                for a in acts[:n_exec]:
                    if (stage == "transport" and env._servo_valid > 0
                            and 0.030 < env._get_gripper_width() < 0.082):
                        e0 = np.asarray(env._servo_err_raw)
                        if float(np.hypot(e0[0], e0[1])) > 0.020:
                            a = np.asarray(a).copy()
                            a[3] = 1.0   # keep holding during the carry
                    obs, _r, done, info = env.step(a)
                    hist.append(obs)
                    step += 1
                    if stage == "transport":
                        transport_steps += 1
                    elif stage == "insert":
                        insert_steps += 1
                    elif stage == "fallback":
                        fallback_steps += 1
                    push_frame()
                    _update(step=step, stage=stage,
                            **_vision_status(env))
                    if stage == "transport":
                        # raw vision error: the smoothed one lags the moving
                        # disk and lets the handoff fire too far off-centre.
                        e = np.asarray(env._servo_err_raw, dtype=np.float64)
                        valid = env._servo_valid > 0
                        near = (valid
                                and abs(e[0]) < HANDOFF_XY_MM * 1e-3
                                and abs(e[1]) < HANDOFF_XY_MM * 1e-3
                                and abs(e[2]) < HANDOFF_Z_MM * 1e-3)
                        handoff_hold = (handoff_hold + 1) if near else 0
                        _update(handoff=bool(near) or handoff_hold > 0)
                        if handoff_hold >= HANDOFF_HOLD:
                            switch = "handoff_vision"
                        # NOTE: no simulator GT-success switch; deployment is
                        # vision-only.
                        elif transport_steps >= TRANSPORT_STEPS:
                            held = env._get_gripper_width() < 0.085
                            if (valid and held
                                    and float(np.hypot(e[0], e[1])) < 0.030
                                    and abs(float(e[2])) < 0.015):
                                switch = "correction"
                            else:
                                switch = "transport_fail"
                        if switch is not None:
                            switched = True
                            done = False
                            info = {}
                            if switch == "transport_fail":
                                _update(message="transport timeout")
                                done = True
                            else:
                                env.ghost_z_mode = "seated"
                                env.skill_stage = "insert"
                                env.reset_integral()
                                env._centered_once = False
                                env._phi_prev = None
                                obs = env._get_obs()
                                hist = deque([obs] * n_obs, maxlen=n_obs)
                                stage = "insert"
                                push_frame()
                                _update(stage=stage, switch=switch,
                                        handoff=True,
                                        message=f"{switch}; inserting")
                            break
                    elif stage == "insert":
                        if not done and insert_steps >= INSERT_STEPS:
                            # no rescue model: the same vision-only insert
                            # policy gets the fixed budget, then the episode
                            # is a failure (matches the held-out eval).
                            done = True
                            _update(message="insert timeout")
                            break
                    elif stage == "fallback":
                        if not done and fallback_steps >= FALLBACK_STEPS:
                            done = True
                            _update(message="rescue timeout")
                            break
                    # In transport, the simulator done flag is privileged
                    # information and also fires on *entry* into the handoff
                    # box, before the vision estimate (which lags the moving
                    # ring) has converged.  Ignore it; only the vision
                    # handoff/correction switch or the step budget ends the
                    # transport stage.
                    if (done and stage != "transport") or stop_event.is_set():
                        break
                if (done and stage != "transport") or stop_event.is_set():
                    break
            success = bool(info.get("success", False)) and (
                stage in ("insert", "fallback"))
            if not switched:
                success = False
            ep_done += 1
            ep_success += int(success)
            _update(ep_done=ep_done, ep_success=ep_success,
                    sr=(ep_success / ep_done) if ep_done else None,
                    success=success, last_steps=step,
                    last_result=("success" if success else "failure"),
                    message=("success" if success else "failure"))
            print(f"[server] seed={ep_seed} task={task} steps={step} "
                  f"stage={stage} switch={switch} success={success} "
                  f"sr={ep_success}/{ep_done}", flush=True)
            if stop_event.is_set():
                break
            if not loop:
                break
            time.sleep(0.5)
        renderer.close()
        env.close()
        _update(running=False, message="stopped")
    except Exception as e:
        print(f"[server] ERROR: {e!r}", flush=True)
        traceback.print_exc()
        _update(running=False, message=f"error: {e}")
    finally:
        stop_event.set()


def _start_rollout(seed, max_steps, video_size, loop):
    global _rollout_thread, _stop_event, _last_frame, _frame_seq
    old = _rollout_thread
    if old is not None and old.is_alive():
        _stop_event.set()
        old.join(timeout=5)
    ev = threading.Event()
    _stop_event = ev
    _last_frame = None
    _frame_seq = 0
    _rollout_thread = threading.Thread(
        target=_rollout_worker,
        args=(seed, max_steps, video_size, loop, ev),
        daemon=True)
    _rollout_thread.start()


@app.route("/")
def index():
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Hanoi — general new-camera vision-only servo</title>
<style>
 :root{color-scheme:dark}
 body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:0;
      padding:22px 26px 40px;background:#101318;color:#e8edf3}
 h1{font-size:21px;margin:0 0 4px;font-weight:650}
 .sub{color:#8fa3b8;font-size:13px;margin-bottom:18px}
 .layout{display:flex;gap:26px;align-items:flex-start;flex-wrap:wrap}
 .panel{background:#171c23;border:1px solid #2a3340;border-radius:12px;
        padding:16px 18px}
 .controls{flex:1 1 340px;min-width:320px;max-width:460px}
 .stage{flex:0 0 auto}
 .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:8px 0}
 label{font-size:13px;color:#b9c8d8}
 select,input{font-size:13px;padding:6px 8px;border-radius:6px;
              border:1px solid #394556;background:#10151b;color:#e8edf3}
 button{font-size:14px;padding:7px 14px;border-radius:8px;
        border:1px solid #3c4a5d;background:#232c38;color:#e8edf3;
        cursor:pointer}
 #btn-start{background:#1e6b3a;border-color:#2f9a55}
 #btn-stop{background:#7a2d2d;border-color:#b34a4a}
 img#stream{width:512px;height:512px;background:#000;border-radius:10px;
            border:1px solid #2a3340;display:block}
 .badge{display:inline-block;padding:3px 10px;border-radius:999px;
        font-size:13px;font-weight:700}
 .badge.running{background:#1d4f86;color:#eaf4ff}
 .badge.success{background:#1e6b3a;color:#eaffef}
 .badge.failure{background:#7a2d2d;color:#ffecec}
 .badge.idle{background:#3a4350;color:#cfd8e3}
 .metric{display:flex;justify-content:space-between;gap:12px;font-size:14px;
         padding:5px 0;border-bottom:1px solid #222b36}
 .metric span:last-child{font-variant-numeric:tabular-nums;color:#f0f6ff}
 .key{color:#8fa3b8}
 .model{font-family:ui-monospace,Menlo,monospace;font-size:12px;
        color:#9fb4c7;word-break:break-all}
 .legend{font-size:12px;color:#9fb4c7;line-height:1.8;margin-top:10px}
 .legend i{display:inline-block;width:10px;height:10px;border-radius:50%;
           margin-right:5px}
 #errbox{font-family:ui-monospace,Menlo,monospace;font-size:13px;
         background:#10151b;border:1px solid #2a3340;border-radius:8px;
         padding:8px 10px;margin-top:10px;white-space:pre}
</style></head><body>
<h1>Hanoi — general new-camera vision-only servo</h1>
<div class="sub">Random top disk -> random target peg (any disk, source,
target, level; no Hanoi size rule). The magenta ring is the synthetic goal
pose; the cyan cross is the vision-estimated moving-disk centre. The policy
sees only the point cloud + the 5-D vision error + integral; no simulator
object pose.</div>
<div class="layout">
 <div class="panel stage">
  <img id="stream" src="/stream" alt="live stream">
  <div class="row" style="justify-content:space-between">
   <span id="badge" class="badge idle">idle</span>
   <span class="legend" id="counter">session: 0 eps</span>
  </div>
  <div id="errbox">error: —</div>
 </div>
 <div class="panel controls">
  <div class="row"><label>Seed</label>
   <input id="seed" type="number" value="836001" style="width:110px">
   <label>Max steps</label>
   <input id="max_steps" type="number" value="390" style="width:80px">
   <label>Size</label>
   <select id="video_size"><option>384</option>
    <option selected>512</option><option>640</option></select></div>
  <div class="row"><label><input id="loop" type="checkbox" checked>
   infinite loop</label></div>
  <div class="row"><button id="btn-start">Start</button>
   <button id="btn-stop">Stop</button></div>
  <div class="row"><span class="key">Task</span></div>
  <div class="model" id="model">—</div>
  <div class="metric"><span class="key">Episode</span><span id="m_ep">—</span></div>
  <div class="metric"><span class="key">Stage</span><span id="m_stage">—</span></div>
  <div class="metric"><span class="key">Switch</span><span id="m_switch">—</span></div>
  <div class="metric"><span class="key">Step</span><span id="m_step">—</span></div>
  <div class="metric"><span class="key">Handoff reached</span><span id="m_handoff">—</span></div>
  <div class="metric"><span class="key">Vision valid</span><span id="m_valid">—</span></div>
  <div class="metric"><span class="key">Error ex,ey,ez (mm)</span><span id="m_err">—</span></div>
  <div class="metric"><span class="key">Disk tilt (deg)</span><span id="m_tilt">—</span></div>
  <div class="metric"><span class="key">Last episode</span><span id="m_last">—</span></div>
  <div class="metric"><span class="key">Session success</span><span id="m_sr">—</span></div>
  <div class="legend">
   <i style="background:#ff00ff"></i>ghost goal ring (virtual block)<br>
   <i style="background:#00ffff"></i>vision-estimated disk centre<br>
   transport -> handoff/correction -> insert ss4 -> rescue fallback.</div>
 </div>
</div>
<script>
const badge=document.getElementById('badge');
const counter=document.getElementById('counter');
const errbox=document.getElementById('errbox');
function setBadge(t,c){badge.textContent=t;badge.className='badge '+(c||'idle');}
async function start(){
 const body={seed:parseInt(document.getElementById('seed').value)||836001,
  max_steps:parseInt(document.getElementById('max_steps').value)||390,
  video_size:parseInt(document.getElementById('video_size').value)||512,
  loop:document.getElementById('loop').checked};
 setBadge('starting…','running');
 const r=await fetch('/start',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const j=await r.json();
 if(!r.ok){setBadge('error: '+j.error,'failure');return;}
 document.getElementById('stream').src='/stream?t='+Date.now();
}
async function stop(){setBadge('stopping…','running');
 await fetch('/stop',{method:'POST'});}
document.getElementById('btn-start').onclick=start;
document.getElementById('btn-stop').onclick=stop;
async function refresh(){
 try{
  const j=await (await fetch('/state')).json();
  if(j.running){
   if(j.success===true)setBadge('success','success');
   else if(j.success===false&&j.last_result==='failure')setBadge('failure','failure');
   else setBadge(j.message||'running','running');
  } else setBadge(j.message||'idle','idle');
  document.getElementById('m_ep').textContent=j.episode||'—';
  document.getElementById('m_stage').textContent=j.stage||'—';
  document.getElementById('m_switch').textContent=j.switch||'—';
  document.getElementById('m_step').textContent=(j.step||0)+' / '+j.max_steps;
  document.getElementById('m_handoff').textContent=
   j.handoff===null||j.handoff===undefined?'—':(j.handoff?'yes':'no');
  document.getElementById('m_valid').textContent=
   j.vision_valid===null||j.vision_valid===undefined?'—':
   (j.vision_valid?'yes':'no');
  document.getElementById('m_err').textContent=j.vision_err_mm?
   j.vision_err_mm.map(x=>x.toFixed(2)).join(', '):'—';
  document.getElementById('m_tilt').textContent=
   j.vision_tilt_deg===null?'—':j.vision_tilt_deg;
  document.getElementById('m_last').textContent=j.last_result?
   j.last_result+(j.last_steps?' ('+j.last_steps+' steps)':''):'—';
  document.getElementById('m_sr').textContent=j.ep_done?
   (j.ep_success+' / '+j.ep_done+'  ('+Math.round(100*j.sr)+'%)'):'—';
  const t=j.task||{};
  document.getElementById('model').textContent=t.disk?
   (t.disk+'  peg '+t.src+' -> peg '+t.dst+'  level '+t.level):'—';
  errbox.textContent='error: '+(j.vision_err_mm?
   '['+j.vision_err_mm.map(x=>x.toFixed(2)).join(', ')+'] mm  '+
   'tilt='+(j.vision_tilt_deg??'—')+' deg  valid='+
   (j.vision_valid?'yes':'no'):'—');
  counter.textContent='session: '+(j.ep_done||0)+' eps · '+
   (j.mode||'')+' · '+j.n_points+' pts · '+j.device;
 }catch(e){}
}
setInterval(refresh,400);refresh();
</script></body></html>"""


@app.route("/start", methods=["POST"])
def start():
    try:
        data = request.get_json(force=True)
        seed = int(data.get("seed", 836001))
        max_steps = int(data.get("max_steps", MAX_STEPS))
        video_size = int(data.get("video_size", 512))
        loop = bool(data.get("loop", True))
        _start_rollout(seed, max_steps, video_size, loop)
        return jsonify({"ok": True, "seed": seed})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/stop", methods=["POST"])
def stop():
    if _stop_event is not None:
        _stop_event.set()
    return jsonify({"ok": True})


@app.route("/state")
def state():
    s = _snapshot()
    s["running"] = bool(_rollout_thread is not None
                        and _rollout_thread.is_alive())
    return jsonify(s)


def _gen_frames():
    last = -1
    while True:
        with _lock:
            frame = _last_frame
            seq = _frame_seq
        if frame is not None and seq != last:
            last = seq
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + frame + b"\r\n")
        else:
            time.sleep(0.05)


@app.route("/stream")
def stream():
    return Response(_gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def main():
    global _device
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8783)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=836001)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--video-size", type=int, default=512)
    ap.add_argument("--no-loop", action="store_true")
    ap.add_argument("--autostart", action="store_true")
    args = ap.parse_args()
    _device = args.device
    if args.autostart:
        _start_rollout(args.seed, args.max_steps, args.video_size,
                       not args.no_loop)
    print(f"[server] http://{args.host}:{args.port}/  device={args.device} "
          f"seed={args.seed} autostart={args.autostart}", flush=True)
    app.run(host=args.host, port=args.port, threaded=True,
            debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
