"""VservoEnv -- Hanoi with a vision-estimated ghost goal and servo error.

The incumbent ``HanoiEnv`` is read-only.  This subclass adds three things,
all computed from the same depth+RGB camera that produces the policy cloud:

1. **Ghost ring**: the next disk's goal pose, drawn as 64 synthetic points on
   the target post.  The post axis and table height are *estimated from the
   yellow post points*, not read from ``env.data`` / ``PEG_XY``.
2. **Servo error**: ``[post_x - disk_x, post_y - disk_y, ghost_z - disk_z,
   normal_x, normal_y]`` plus a validity flag.  The disk pose comes from the
   vision-only ICP estimator; the ghost pose comes from the observed post.
   The systematic camera-frame offset cancels because both use the same
   cloud.
3. **Dense vservo reward**: a potential on the same vision error, with a
   two-phase insert shape (centre first, then descend) and a centred bonus.

No simulator object pose is used by the observation or the dense reward.
The sparse task/lift/insert events still use the simulator predicates, since
they are episode-level task events rather than the servo signal.
"""
from __future__ import annotations

from typing import Optional

import os

import numpy as np

from rl_100.env.arx5.arx5_env import _fps_downsample
from hanoi_env import (DISK_HALF_T, FINGER_TO_PAD_Z, PEG_XY,
                       TABLE_Z, HanoiEnv)
from vision_pose import (
    estimate_disk, estimate_peg_axis, estimate_table_z,
    recolor_robot_primaries, sample_disk_model)

GHOST_G = 64
GHOST_RGB = np.array([255.0, 0.0, 255.0], dtype=np.float32)  # magenta
# FAST: the post and the table are static in this fixture.  Estimate them
# only during a short warm-up (or while missing) and reuse the cached
# geometry; the moving-disk estimate still runs every vision frame.
FAST_STATIC_CACHE = os.environ.get("VSERVO_STATIC_CACHE", "1") != "0"


class VservoEnv(HanoiEnv):
    """HanoiEnv + vision-only ghost goal, servo error and dense reward."""

    N_GHOST = GHOST_G
    N_ERR = 5          # [ex, ey, ez, nx, ny]
    N_VALID = 1
    N_INT = 3          # integral of the position error (I-controller state)
    INT_BETA = 0.90
    INT_MAX = 0.02     # m, anti-windup
    INT_GATE = 0.005   # m, only integrate the steady-state error
    # reward weights (metres / radians)
    W_XY = 10.0
    W_Z = 4.0
    W_TILT = 1.5
    CENTER_XY_TOL = 0.0015
    CENTER_Z_TOL = 0.004
    CENTER_TILT_TOL = np.deg2rad(4.0)
    SAFE_XY_TOL = 0.004
    DESCENT_PENALTY = 0.010
    CENTER_BONUS = 0.20
    # temporal filters
    ALPHA_POST = 0.20
    ALPHA_DISK = 0.60

    def __init__(self, *args, vision_disk: str = "disk_s",
                 ghost_level: int = 0, ghost_z_mode: str = "seated",
                 handoff_z: float = 0.268, use_vision: bool = True,
                 home_jitter: float = 0.0, **kwargs):
        if ghost_z_mode not in ("seated", "handoff"):
            raise ValueError(
                f"ghost_z_mode must be seated/handoff, got {ghost_z_mode!r}")
        # must exist before super().__init__ in case the base calls self.seed
        self._episode_seed = None
        kwargs.setdefault("use_point_cloud", True)
        super().__init__(*args, **kwargs)
        n_grey = recolor_robot_primaries(self.model)
        self.vision_disk = str(vision_disk)
        self.ghost_level = int(ghost_level)
        self.ghost_z_mode = str(ghost_z_mode)
        self.handoff_z = float(handoff_z)
        self.use_vision = bool(use_vision)
        self.home_jitter = float(home_jitter)
        # Fast evaluation path (see hanoi_vservo_errloss_fast_1.0):
        #  - static_cache: post axis / table height are static per episode;
        #    estimate them only on the first 5 vision frames, then reuse.
        #  - vision_skip: the policy consumes 2 observations at the start of
        #    each 4-action chunk; compute the cloud+vision only on the last
        #    two actions of the chunk (steps 3 and 0 mod 4) and reuse the
        #    cached observation otherwise, so the two-frame history at every
        #    policy call is still fresh and consecutive.
        self.static_cache = os.environ.get("VSERVO_STATIC_CACHE", "0") == "1"
        self.vision_skip = os.environ.get("VSERVO_VISION_SKIP", "0") == "1"
        self._obs_cache = None
        self.pc_channels = 7  # xyz + rgb + ghost flag
        self._vision_rng = np.random.default_rng(7)
        self._ghost_local = sample_disk_model(
            self.vision_disk, n=GHOST_G, rng=self._vision_rng)
        self._reset_vision()
        # print once, useful in collection logs
        print(f"[VservoEnv] recoloured {n_grey} robot primaries; "
              f"disk={self.vision_disk} ghost_level={self.ghost_level} "
              f"mode={self.ghost_z_mode} pc_channels={self.pc_channels}")

    # ------------------------------------------------------------------
    # vision state
    # ------------------------------------------------------------------
    def _set_arm_home(self):
        super()._set_arm_home()
        if self.home_jitter > 0.0:
            for i, adr in enumerate(self._joint_qpos_adrs):
                dq = float(self.np_random.uniform(-self.home_jitter,
                                                  self.home_jitter))
                q = float(np.clip(self.data.qpos[adr] + dq,
                                  self._joint_lows[i], self._joint_highs[i]))
                self.data.qpos[adr] = q
                self.data.ctrl[self._joint_actuator_ids[i]] = q

    def _reset_vision(self):
        self._obs_cache = None
        self._post_axis: Optional[np.ndarray] = None
        self._post_z_min: Optional[float] = None
        self._table_z: Optional[float] = None
        self._disk_centre: Optional[np.ndarray] = None
        self._disk_normal: Optional[np.ndarray] = None
        self._servo_err = np.zeros(self.N_ERR, dtype=np.float32)
        self._servo_err_raw = np.zeros(self.N_ERR, dtype=np.float32)
        self._servo_valid = 0.0
        self._servo_integral = np.zeros(self.N_INT, dtype=np.float32)
        self._ghost_centre: Optional[np.ndarray] = None
        self._phi_prev: Optional[float] = None
        self._centered_once = False
        self._vision_frames = 0
        self._vision_valid_frames = 0
        self._invalid_run = 0
        self.vision_info = {}
        # optional learned estimator correction (vision+proprioception only;
        # the coefficients were fitted offline with GT labels)
        self._vision_calib = None
        cal_path = os.environ.get("VSERVO_CALIB_JSON", "")
        if cal_path and os.path.exists(cal_path):
            try:
                import json
                with open(cal_path) as f:
                    self._vision_calib = json.load(f)
            except Exception:
                self._vision_calib = None

    def reset_integral(self):
        """Reset the error integral at a skill/target switch."""
        self._servo_integral = np.zeros(self.N_INT, dtype=np.float32)

    def seed(self, seed=None):
        # remember the episode seed so reset() can reseed the vision RANSAC
        self._episode_seed = seed
        return super().seed(seed)

    def reset(self):
        self._reset_vision()
        # The vision estimator's RNG used to be created once in __init__ and
        # advanced across episodes, so the same seed produced a different
        # trajectory depending on how many episodes had run before it (e.g.
        # live seed 980010 was the 11th episode, a standalone rerun was the
        # 1st).  Reseed it per episode: same seed -> same RANSAC stream.
        base = 7 if self._episode_seed is None else int(self._episode_seed) + 7
        self._vision_rng = np.random.default_rng(base)
        return super().reset()

    # ------------------------------------------------------------------
    # point cloud: dense render -> vision -> ghost points -> FPS
    # ------------------------------------------------------------------
    def _render_point_cloud(self):
        if self._pc_renderer is None:
            return np.zeros((self.n_points, self.pc_channels),
                            dtype=np.float32)
        cam_id = self._cam_pc_id
        self._pc_renderer.enable_depth_rendering()
        self._pc_renderer.update_scene(self.data, camera=cam_id)
        depth = self._pc_renderer.render().astype(np.float32)
        self._pc_renderer.disable_depth_rendering()
        self._pc_renderer.update_scene(self.data, camera=cam_id)
        rgb = self._pc_renderer.render()

        H, W = depth.shape
        f = H / (2.0 * np.tan(np.radians(self.model.cam_fovy[cam_id]) / 2.0))
        cx, cy = W / 2.0, H / 2.0
        v, u = np.mgrid[0:H, 0:W]
        x = (u - cx) * depth / f
        y = (v - cy) * depth / f
        pts_cam = np.stack([x, y, depth], axis=-1).reshape(-1, 3)
        mask = (depth > 0.001) & (depth < self._pc_far)
        mask = mask.reshape(-1)
        if not np.any(mask):
            return np.zeros((self.n_points, self.pc_channels),
                            dtype=np.float32)
        pts_cam = pts_cam[mask]
        R_x180 = np.array([[1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0],
                           [0.0, 0.0, -1.0]], dtype=np.float32)
        R_cw = self.data.cam_xmat[cam_id].reshape(3, 3).astype(np.float32) @ R_x180
        t_cw = self.data.cam_xpos[cam_id].astype(np.float32)
        pts_world = pts_cam @ R_cw.T + t_cw
        colors = rgb.reshape(-1, 3)[mask].astype(np.float32)

        xmin, ymin, zmin, xmax, ymax, zmax = self.pc_crop_bounds
        keep = ((pts_world[:, 0] >= xmin) & (pts_world[:, 0] <= xmax)
                & (pts_world[:, 1] >= ymin) & (pts_world[:, 1] <= ymax)
                & (pts_world[:, 2] >= zmin) & (pts_world[:, 2] <= zmax))
        pts_world = pts_world[keep]
        colors = colors[keep]
        dense = np.concatenate([pts_world, colors], axis=-1)

        # vision runs on the dense cloud *before* FPS
        self._update_vision(dense)

        # reserve ghost slots: FPS the real scene to n_points - N_GHOST
        n_real = self.n_points - self.N_GHOST
        n_table = int(round(0.10 * n_real))
        is_table = pts_world[:, 2] < 0.205
        table_pts = dense[is_table]
        obj_pts = dense[~is_table]
        if len(table_pts) > n_table:
            table_pts = _fps_downsample(table_pts, n_table, seed=1)
        n_obj = n_real - len(table_pts)
        if len(obj_pts) > n_obj:
            obj_pts = _fps_downsample(obj_pts, n_obj, seed=0)
        real = np.concatenate([obj_pts, table_pts], axis=0)
        if len(real) < n_real:
            pad = np.zeros((n_real - len(real), 6), dtype=np.float32)
            real = np.concatenate([real, pad], axis=0)
        real = real[:n_real].astype(np.float32)

        ghost = self._ghost_points()
        out = np.zeros((self.n_points, self.pc_channels), dtype=np.float32)
        out[:n_real, :6] = real
        out[n_real:, :3] = ghost
        out[n_real:, 3:6] = GHOST_RGB
        out[n_real:, 6] = 1.0
        return out

    def _ghost_points(self) -> np.ndarray:
        if self._ghost_centre is None:
            return np.zeros((self.N_GHOST, 3), dtype=np.float32)
        return (self._ghost_local + self._ghost_centre).astype(np.float32)

    # ------------------------------------------------------------------
    # vision update
    # ------------------------------------------------------------------
    def _update_vision(self, dense: np.ndarray):
        self._vision_frames += 1
        hint = (self._post_axis if self._post_axis is not None
                else np.asarray(PEG_XY[self.goal_peg], dtype=np.float64))
        if (not getattr(self, "static_cache", False)
                or self._vision_frames <= 5
                or self._post_axis is None or self._table_z is None):
            post = estimate_peg_axis(dense, hint, rng=self._vision_rng)
            table = estimate_table_z(dense, rng=self._vision_rng)
        else:
            post = None
            table = None
        disk = estimate_disk(dense, self.vision_disk, rng=self._vision_rng)
        if table is not None:
            tz = table["table_z"]
            if self._table_z is None or abs(tz - self._table_z) < 0.005:
                self._table_z = (tz if self._table_z is None
                                 else (1 - self.ALPHA_POST) * self._table_z
                                 + self.ALPHA_POST * tz)
        # reject bad geometry fits: a coarse fit without a converged ICP, a
        # high RMS, too few matched points, a narrow visible arc, or a jump.
        bad = disk is None
        if not bad:
            bad = ("icp_rms" not in disk
                   or float(disk["icp_rms"]) > 0.0030
                   or int(disk.get("icp_kept", 0)) < 15
                   or float(disk.get("arc_coverage_deg", 0.0)) < 55.0)
        if not bad and self._table_z is not None:
            c = disk["centre"]
            bad = not (0.05 < c[0] < 0.60 and -0.30 < c[1] < 0.42
                       and self._table_z - 0.02 < c[2]
                       < self._table_z + 0.20)
        # the jump gate stops single-frame outliers, but once the disk has
        # been lost for a few frames (fast carry / occlusion) the next good
        # fit must be accepted, otherwise the error stays frozen for the
        # rest of the episode
        if not bad and self._disk_centre is not None:
            reacq = int(os.environ.get("VSERVO_REACQ_FRAMES", "5"))
            if reacq <= 0 or getattr(self, "_invalid_run", 0) < reacq:
                bad = float(np.linalg.norm(
                    disk["centre"] - self._disk_centre)) > 0.060
        if bad:
            self._servo_valid = 0.0
            self._invalid_run = getattr(self, "_invalid_run", 0) + 1
            self.vision_info = {
                "valid": False, "reason": "disk_quality",
                "post_ok": post is not None,
                "table_ok": table is not None,
                "icp_rms": (float(disk.get("icp_rms", np.nan))
                            if disk is not None else np.nan),
                "icp_n": (int(disk.get("icp_kept", 0))
                          if disk is not None else 0),
                "arc_deg": (float(disk.get("arc_coverage_deg", np.nan))
                            if disk is not None else np.nan),
                "n_frames": self._vision_frames,
            }
            return
        # learned estimator correction: bias is a function of the raw
        # estimate, the EE pose and the gripper width only (no GT at runtime)
        cal = getattr(self, "_vision_calib", None)
        if cal is not None:
            m = cal.get("models", {}).get(self.vision_disk)
            # The correction is fitted on held-ring frames; only apply it
            # while a ring is actually between the fingers (otherwise the
            # open-grip approach frames are extrapolated).
            gate = float(os.environ.get("VSERVO_CALIB_GRIP_GATE", "0.085"))
            # The fit is valid near the handoff pose above the target post;
            # gate it to that region so the gross carry does not extrapolate.
            xy_gate = float(os.environ.get("VSERVO_CALIB_XY_GATE", "0.030"))
            near_post = True
            if xy_gate > 0.0 and self._post_axis is not None:
                near_post = (float(np.hypot(
                    float(disk["centre"][0]) - float(self._post_axis[0]),
                    float(disk["centre"][1]) - float(self._post_axis[1])))
                    < xy_gate)
            if (m is not None
                    and near_post
                    and getattr(self, "_finger_site_id", None) is not None
                    and (gate <= 0.0
                         or float(self._get_gripper_width()) < gate)):
                W = np.asarray(m["W"], dtype=np.float64)
                ee = np.asarray(
                    self.data.site(self._finger_site_id).xpos,
                    dtype=np.float64)
                feat = np.array([
                    float(disk["centre"][0]), float(disk["centre"][1]),
                    float(disk["centre"][2]), float(ee[0]), float(ee[1]),
                    float(ee[2]), float(self._get_gripper_width()), 1.0])
                disk["centre"] = (
                    np.asarray(disk["centre"], dtype=np.float64)
                    + W @ feat)
        self._disk_centre = disk["centre"]
        self._disk_normal = disk["normal"]
        self._invalid_run = 0

        if post is not None:
            axis = post["axis_xy"]
            if self._post_axis is None:
                self._post_axis = axis
            elif np.linalg.norm(axis - self._post_axis) < 0.005:
                a = self.ALPHA_POST
                self._post_axis = (1 - a) * self._post_axis + a * axis
            zmin = post["z_min"]
            self._post_z_min = (zmin if self._post_z_min is None
                                else (1 - self.ALPHA_POST) * self._post_z_min
                                + self.ALPHA_POST * zmin)
        # fallback only when the table plane is not visible for a long time
        if self._table_z is None and self._post_z_min is not None:
            self._table_z = self._post_z_min
        if self._post_axis is None or self._table_z is None:
            self._servo_valid = 0.0
            self.vision_info = {
                "valid": False, "reason": "post/table", "disk_ok": True,
                "n_frames": self._vision_frames,
            }
            return

        if self.skill_stage == "grasp":
            # Grasp-stage visual servo: drive the finger site to the grasp
            # pose above/beside the moving ring (the expert's grasp pose is
            # disk_centre + (FINGER_TO_PAD_Z, 0, DISK_HALF_T + 0.5 mm)).
            goal = np.array([
                float(self._disk_centre[0]) + FINGER_TO_PAD_Z,
                float(self._disk_centre[1]),
                float(self._disk_centre[2]) + DISK_HALF_T + 0.0015,
            ], dtype=np.float64)
            finger = self.data.site(self._finger_site_id).xpos
            err = np.array([goal[0] - finger[0], goal[1] - finger[1],
                            goal[2] - finger[2], 0.0, 0.0],
                           dtype=np.float32)
            self._servo_err_raw = err.copy()
            if self._servo_valid > 0:
                a = self.ALPHA_DISK
                self._servo_err = (1 - a) * self._servo_err + a * err
            else:
                self._servo_err = err
            self._servo_valid = 1.0
            e3 = self._servo_err[:3]
            if float(np.linalg.norm(e3)) < self.INT_GATE:
                self._servo_integral = np.clip(
                    self.INT_BETA * self._servo_integral + e3,
                    -self.INT_MAX, self.INT_MAX).astype(np.float32)
            else:
                self._servo_integral = (
                    self.INT_BETA * self._servo_integral).astype(np.float32)
            self._vision_valid_frames += 1
            self.vision_info = {
                "valid": True, "err": self._servo_err.copy(),
                "disk_centre": self._disk_centre.copy(),
                "finger": finger.copy(), "grasp_goal": goal,
                "table_z": float(self._table_z),
                "n_frames": self._vision_frames,
                "n_valid": self._vision_valid_frames,
            }
            return
        if self.ghost_z_mode == "handoff":
            # handoff height above the *observed* table, so the global cloud
            # z offset cancels
            # The expert's pre-insert height is clamped just above the post
            # cap (PEG_TOP_Z + 20 mm) for every target level, so the handoff
            # ghost is level-independent; only the seated ghost uses the level.
            ghost_z = self._table_z + (self.handoff_z - TABLE_Z)
        else:
            ghost_z = (self._table_z + DISK_HALF_T
                       + self.ghost_level * 2.0 * DISK_HALF_T)
        self._ghost_centre = np.array(
            [self._post_axis[0], self._post_axis[1], ghost_z],
            dtype=np.float64)
        err = np.array([
            self._ghost_centre[0] - self._disk_centre[0],
            self._ghost_centre[1] - self._disk_centre[1],
            self._ghost_centre[2] - self._disk_centre[2],
            self._disk_normal[0],
            self._disk_normal[1],
        ], dtype=np.float32)
        self._servo_err_raw = err.copy()
        # temporal smoothing of the disk-derived error
        if self._servo_valid > 0:
            a = self.ALPHA_DISK
            self._servo_err = (1 - a) * self._servo_err + a * err
        else:
            self._servo_err = err
        self._servo_valid = 1.0
        # integral of the position error: the policy sees the accumulated
        # error, so it can learn I-controller-like correction over time.
        # Gate to the near-goal phase to avoid winding up during the gross
        # transport / descent; decay outside the gate.
        e3 = self._servo_err[:3]
        if float(np.linalg.norm(e3)) < self.INT_GATE:
            self._servo_integral = np.clip(
                self.INT_BETA * self._servo_integral + e3,
                -self.INT_MAX, self.INT_MAX).astype(np.float32)
        else:
            self._servo_integral = (
                self.INT_BETA * self._servo_integral).astype(np.float32)
        self._vision_valid_frames += 1
        self.vision_info = {
            "valid": True,
            "err": self._servo_err.copy(),
            "err_raw": err,
            "disk_centre": self._disk_centre.copy(),
            "post_axis": self._post_axis.copy(),
            "ghost_centre": self._ghost_centre.copy(),
            "table_z": float(self._table_z),
            "disk_tilt_deg": float(np.degrees(disk["tilt_rad"])),
            "icp_rms": float(disk.get("icp_rms", np.nan)),
            "arc_deg": float(disk.get("arc_coverage_deg", np.nan)),
            "post_arc_deg": (float(np.degrees(post["arc_coverage_rad"]))
                             if post is not None else np.nan),
            "n_frames": self._vision_frames,
            "n_valid": self._vision_valid_frames,
        }

    # ------------------------------------------------------------------
    # observation: append the vision-only error
    # ------------------------------------------------------------------
    def _obs_need_fresh(self):
        # The skip pattern assumes a 4-action chunk with a 2-observation
        # history.  The insert stage runs 2-action chunks, where skipping
        # would feed the policy two identical (stale) frames per call, so it
        # always refreshes there.
        if getattr(self, "skill_stage", "") != "transport":
            return True
        sc = int(getattr(self, "step_count", 0))
        if sc % 4 in (0, 3):
            return True
        e = self._servo_err
        # keep the servo error fresh near the handoff so the stage-switch
        # counting is identical to the every-step pipeline
        if (self.skill_stage != "insert" and self._servo_valid > 0
                and float(np.hypot(e[0], e[1])) < 0.030
                and abs(float(e[2])) < 0.030):
            return True
        return False

    def _get_obs(self):
        if (self.vision_skip and self._obs_cache is not None
                and not self._obs_need_fresh()):
            return self._obs_cache
        obs = super()._get_obs()
        obs["agent_pos"] = np.concatenate([
            np.asarray(obs["agent_pos"], dtype=np.float32),
            self._servo_err.astype(np.float32),
            np.array([self._servo_valid], dtype=np.float32),
            self._servo_integral.astype(np.float32),
        ]).astype(np.float32)
        if self.vision_skip:
            self._obs_cache = obs
        return obs

    # ------------------------------------------------------------------
    # dense vision reward
    # ------------------------------------------------------------------
    def _vservo_phi(self) -> Optional[float]:
        if self._servo_valid <= 0 or self._ghost_centre is None:
            return None
        e = self._servo_err.astype(np.float64)
        d_xy = float(np.hypot(e[0], e[1]))
        dz = abs(float(e[2]))
        tilt = float(np.hypot(e[3], e[4]))
        # two-phase insert shape: no z progress while laterally off-centre
        z_term = dz if d_xy <= self.SAFE_XY_TOL else 0.0
        return -(self.W_XY * d_xy + self.W_Z * z_term
                 + self.W_TILT * tilt)

    def _compute_reward(self):
        reward, done, info = super()._compute_reward()
        # remove the simulator-frame carry shaping; use the vision potential
        carried = float(info.get("reward_carry", 0.0))
        reward -= carried
        info["reward_carry"] = 0.0
        info["reward_vservo"] = 0.0

        phi = self._vservo_phi()
        if phi is not None:
            if self._phi_prev is None:
                self._phi_prev = phi
            r = phi - self._phi_prev
            self._phi_prev = phi
            info["reward_vservo"] = float(r)
            reward += r
            # centered subgoal: bonus once per episode
            e = self._servo_err.astype(np.float64)
            d_xy = float(np.hypot(e[0], e[1]))
            dz = abs(float(e[2]))
            tilt = float(np.hypot(e[3], e[4]))
            if (not self._centered_once
                    and d_xy <= self.CENTER_XY_TOL
                    and dz <= self.CENTER_Z_TOL
                    and tilt <= self.CENTER_TILT_TOL):
                self._centered_once = True
                reward += self.CENTER_BONUS
                info["centered"] = True
            # premature-descent penalty: moving down while off-centre
            # (insert only; the transport ghost is high above the post)
            if (self.ghost_z_mode == "seated"
                    and d_xy > self.SAFE_XY_TOL
                    and self._ghost_centre is not None
                    and self._disk_centre is not None
                    and self._disk_centre[2]
                    < self._ghost_centre[2] + 0.010):
                reward -= self.DESCENT_PENALTY
                info["descent_penalty"] = self.DESCENT_PENALTY
        info["vision_valid"] = bool(self._servo_valid > 0)
        info["vision_err"] = self._servo_err.tolist()
        info["vision_frames"] = self._vision_frames
        info["vision_valid_frames"] = self._vision_valid_frames
        return reward, done, info
