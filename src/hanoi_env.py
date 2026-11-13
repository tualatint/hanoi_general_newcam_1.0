"""HanoiEnv -- 3 pegs, 3 ring disks, ARX L5 top-down pick-and-place.

Task
----
Move all three disks to peg C (``goal_peg=2``), largest at the bottom and
smallest on top.  The initial state is sampled from the legal Hanoi states
whose optimal solution is at least ``min_dist`` moves long (default 5), so a
single episode is a long sequence of 5-7 pick-and-place moves (~200-300 env
steps) instead of the multiball task's single pick-place (~30-45 steps).

Implementation
--------------
Subclasses ``SphereBoxEnv`` (which provides the validated MetaWorld 4-dim
EE-delta action interface, the 9-dim agent state and the depth->point-cloud
pipeline) and swaps in ``hanoi_scene.xml`` after the parent caches the
single-sphere model, exactly like ``SphereBoxMultiballEnv``.  The grasp
orientation is the top-down rotation ``R_TOP`` (site x = -z world, jaw along
+y); feasibility was verified in simulation before this file was written.

No file in the shared ``multiball/`` or ``RL-100/`` trees is modified.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

import mujoco

from rl_100.env.arx5.sphere_box_env import SphereBoxEnv
from rl_100.env.arx5.arx5_env import _fps_downsample
from rl_100.env.arx5.ik import IKSolver

import hanoi_solver as S

HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "hanoi_scene.xml")
INSERT_TEMPLATE_PATH = os.path.join(HERE, "hanoi_insert_templates_train.npz")

TABLE_Z = 0.19
DISK_THICKNESS = 0.020
DISK_HALF_T = DISK_THICKNESS / 2.0
DISK_NAMES = ("disk_l", "disk_m", "disk_s")   # index 0(large) .. 2(small)
PEG_X = 0.30
PEG_YS = (-0.074, 0.03, 0.134)  # 20% tighter than (-0.10, 0.03, 0.16)
PEG_XY = np.array([[PEG_X, y] for y in PEG_YS], dtype=np.float64)

# Bodies excluded from the point cloud (floor, backdrop, table legs and the
# invisible analytic contact plane).  The table top itself is kept as a sparse
# 10% subset by the parent renderer; the pegs and disks are kept in full.
PC_EXCLUDE_NAMES = (
    "floor", "bg_ceiling", "bg_back", "bg_left", "bg_right",
    "table_leg1", "table_leg2", "table_leg3", "table_leg4",
    "table_top_objplane",
)

# Top-down grasp frame: site x = approach (down), site y = jaw opening (+y),
# site z = finger_site offset (+x).  Columns are the site axes in world coords.
R_TOP = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
], dtype=float)

# Finger-site -> pad-centre offset along site z (1.47564 cm from the robot XML).
FINGER_TO_PAD_Z = 0.0147564
# Pad geometry: the 4 cm pad long axis is site x (vertical for R_TOP), so the
# pad bottom is 2 cm below the finger_site.  Grasp at finger_z = disk_top
# puts the pad bottom flush with the disk bottom.
PAD_HALF_LEN = 0.020


class HanoiEnv(SphereBoxEnv):
    # ---- interface / task constants ----
    MW_GRASP_ROTATION = R_TOP
    PC_CROP_BOUNDS = (-0.08, -0.32, 0.10, 0.48, 0.42, 0.55)
    # Home arm configuration for the top-down frame (solved by IKSolver in
    # __init__; this is the verified fallback for finger (0.30, 0.03, 0.33)).
    MW_HOME_JOINTS = np.array(
        [0.1050, 1.6450, 1.6000, -1.5260, 0.0, 0.1050], dtype=float)

    # ---- dense skill reward (reward_mode="dense") ----
    LIFT_BONUS = 0.5
    INSERT_BONUS = 0.5
    SHAPING_GAIN = 0.1
    PHI_CELL = 0.03
    LIFT_CLEAR = 0.025
    PLACE_XY_TOL = 0.010
    PLACE_Z_TOL = 0.006
    SKILL_SUCCESS_REWARD = 1.0
    LIFT_HOLD_STEPS = 5
    # A ring threaded at the bottom of the post counts as done even if the
    # gripper is still closed around it (the insert policy sometimes keeps
    # holding after seating; requiring jaws-open made the episode continue
    # and the rescue would pull the seated ring back out).
    SEATED_HOLD_STEPS = 3

    # ---- handoff envelope (shared by BOTH skills) ----
    # The transport target and the insert-start distribution must be the
    # *same* box; otherwise transport can terminate anywhere in a wide window
    # (easy 100% SR) while the insert expert is trained to correct a much
    # smaller error.  ``make_insert_templates.py`` samples exactly this box,
    # so the two experts meet on one distribution:
    #   xy: |dx|, |dy| <= HANDOFF_XY_TOL
    #   z : held disk centre in [HANDOFF_Z, HANDOFF_Z + HANDOFF_Z_UP]
    # HANDOFF_Z is the measured disk-centre height at the scripted CENTER
    # phase (the finger setpoint there is 0.270).
    HANDOFF_Z = 0.268
    HANDOFF_XY_TOL = 0.005
    HANDOFF_Z_UP = 0.004

    # Hold inside the envelope for a few steps, mirroring LIFT_HOLD_STEPS.
    # With hold=1 the episode ends on the *transition* into the box, so the
    # recorded data (states at action time) contains no in-box frame and BC
    # only learns the approach velocity -- the old transport policy overshot
    # the peg by 10-20 mm and descended to z=0.265 (below the insert start
    # range).  Holding lets the scripted expert servo the disk to the centre
    # and gives BC in-box frames that teach it to settle.
    TRANSPORT_HOLD_STEPS = 5
    TRANSPORT_XY_TOL = HANDOFF_XY_TOL
    TRANSPORT_Z_MIN = HANDOFF_Z
    TRANSPORT_Z_MAX = HANDOFF_Z + HANDOFF_Z_UP

    def __init__(self, render_size=84, max_episode_steps=800, n_points=512,
                 image_size=84, verbose=False, seed=42, use_point_cloud=True,
                 use_image=False, pc_resolution=160, min_dist=5,
                 goal_peg=2, jitter_xy=0.0002, jitter_yaw_deg=1.0,
                 reward_mode="task", skill_mode=False, skill_stage="none",
                 insert_template_path: Optional[str] = None,
                 use_contact_obs: bool = False,
                 mw_action_scale=None, mw_max_step=None, mw_n_substeps=None):
        if reward_mode not in ("task", "dense"):
            raise ValueError(f"reward_mode must be 'task' or 'dense', "
                             f"got {reward_mode!r}")
        if skill_stage not in ("none", "lift", "grasp", "transport",
                              "insert", "lift_insert"):
            raise ValueError(f"skill_stage must be none/lift/transport/"
                             f"insert/lift_insert, got {skill_stage!r}")
        if skill_mode and skill_stage == "none":
            skill_stage = "lift_insert"
        self.min_dist = int(min_dist)
        self.goal_peg = int(goal_peg)
        self.reward_mode = str(reward_mode)
        self.skill_stage = str(skill_stage)
        self.skill_mode = self.skill_stage != "none"
        self._insert_template_path = (insert_template_path
                                      or INSERT_TEMPLATE_PATH)
        self.use_contact_obs = bool(use_contact_obs)
        self._goal_state = tuple([self.goal_peg] * S.N_DISKS)
        self._jitter_xy = float(jitter_xy)
        self._jitter_yaw = np.deg2rad(float(jitter_yaw_deg))
        super().__init__(
            render_size=render_size,
            max_episode_steps=max_episode_steps,
            n_points=n_points,
            image_size=image_size,
            verbose=verbose,
            seed=seed,
            use_point_cloud=use_point_cloud,
            use_image=use_image,
            pc_resolution=pc_resolution,
            mw_action_scale=mw_action_scale,
            mw_max_step=mw_max_step,
            mw_n_substeps=mw_n_substeps,
        )
        self._swap_model(XML_PATH)
        # Solve a reachable top-down home pose from the known good seed.
        self._home_q = self._solve_home()
        self.MW_HOME_JOINTS = self._home_q
        self._initial_state: Optional[Tuple[int, int, int]] = None
        self._dist_prev: Optional[int] = None
        self._init_dist: Optional[int] = None
        self._knocked = False
        self._success_prev = False
        self._insert_templates = None
        self._reset_shaping()

    def _reset_shaping(self):
        """Per-episode one-time skill flags and carry potential anchor."""
        self._lifted_once = {n: False for n in DISK_NAMES}
        self._inserted_once = {n: False for n in DISK_NAMES}
        self._carry_disk = None
        self._phi_anchor = None
        self._lift_hold = 0
        self._transport_hold = 0
        self._seated_hold = 0

    CONTACT_DIM = 34

    def contact_features(self) -> np.ndarray:
        """34-D contact/torque feedback for the contact-aware insert policy.

        Groups: left pad wrench (6), right pad (6), disk_s contacts (6),
        peg2 contacts (6), gripper actuator forces (2), six joint torques (6),
        contact count (1), max penetration (1).  Contact wrenches come from
        ``mj_contactForce`` (contact frame); the torques are the 'virtual
        current' analog on the ARX.
        """
        left = np.zeros(6)
        right = np.zeros(6)
        disk = np.zeros(6)
        peg = np.zeros(6)
        max_pen = 0.0
        ncon = int(self.data.ncon)
        res = np.zeros(6, dtype=np.float64)
        for i in range(ncon):
            c = self.data.contact[i]
            n1 = self.model.geom(c.geom1).name or ""
            n2 = self.model.geom(c.geom2).name or ""
            res[:] = 0.0
            mujoco.mj_contactForce(self.model, self.data, i, res)
            max_pen = max(max_pen, -float(c.dist))
            if "left_finger_pad" in n1 or "left_finger_pad" in n2:
                left += res
            if "right_finger_pad" in n1 or "right_finger_pad" in n2:
                right += res
            if "disk_s" in n1 or "disk_s" in n2:
                disk += res
            if "peg2" in n1 or "peg2" in n2:
                peg += res
        grip = float(self.data.actuator_force[self._gripper_actuator_id])
        grip2 = (float(self.data.actuator_force[self._gripper2_actuator_id])
                 if self._gripper2_actuator_id is not None else 0.0)
        torques = np.asarray(
            [self.data.qfrc_actuator[a] for a in self._joint_dof_adrs],
            dtype=np.float64)
        return np.concatenate([
            left, right, disk, peg, [grip, grip2], torques,
            [float(ncon), max_pen],
        ]).astype(np.float32)

    def _get_obs(self):
        obs = super()._get_obs()
        if self.use_contact_obs:
            obs["agent_pos"] = np.concatenate([
                np.asarray(obs["agent_pos"], dtype=np.float32),
                self.contact_features(),
            ]).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    # model swap
    # ------------------------------------------------------------------
    def _swap_model(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._pc_far = float(self.model.vis.map.zfar * self.model.stat.extent)

        self._disk_body_ids = {
            name: self.model.body(name).id for name in DISK_NAMES}
        self._object_body_ids = dict(self._disk_body_ids)
        self._obj_body_id = self._disk_body_ids["disk_l"]
        self._current_object = "disk_l"

        self._ee_site_id = self.model.site("ee_site").id
        self._grasp_site_id = self.model.site("grasp_site").id
        self._finger_site_id = self.model.site("finger_site").id
        try:
            self._finger_site_id = self.model.site("finger_site").id
        except KeyError:
            self._finger_site_id = self._grasp_site_id

        self._joint_actuator_ids = [
            self.model.actuator(f"joint{i}").id for i in range(1, 7)]
        self._gripper_actuator_id = self.model.actuator("gripper").id
        try:
            self._gripper2_actuator_id = self.model.actuator("gripper2").id
        except KeyError:
            self._gripper2_actuator_id = None

        self._joint_qpos_adrs = [
            self.model.jnt_qposadr[self.model.joint(f"joint{i}").id]
            for i in range(1, 7)]
        self._gripper_qpos_adr = self.model.jnt_qposadr[
            self.model.joint("joint7").id]
        self._gripper2_qpos_adr = self.model.jnt_qposadr[
            self.model.joint("joint8").id]
        self._joint_dof_adrs = [
            self.model.jnt_dofadr[self.model.joint(f"joint{i}").id]
            for i in range(1, 7)]
        self._joint_lows = np.array([r[0] for r in self.JOINT_RANGES])
        self._joint_highs = np.array([r[1] for r in self.JOINT_RANGES])
        self.ik = IKSolver(self) if IKSolver is not None else None

        self._cam_pc_id = self.model.camera("cam_pc").id
        self._pc_exclude_geom_ids = set()
        for i in range(self.model.ngeom):
            if self.model.geom(i).name in PC_EXCLUDE_NAMES:
                self._pc_exclude_geom_ids.add(i)
        # The parent init created a renderer for the original sphere_box
        # scene.  Close it before opening the renderer for the swapped model;
        # leaving two GL contexts alive makes MuJoCo's segmentation readback
        # return garbage on some EGL/OSMesa drivers.
        for attr in ("_renderer", "_pc_renderer"):
            old = getattr(self, attr, None)
            if old is not None:
                try:
                    old.close()
                except Exception:
                    pass
            setattr(self, attr, None)
        if self.use_point_cloud:
            self._pc_renderer = mujoco.Renderer(
                self.model, height=self.pc_resolution,
                width=self.pc_resolution)

    def _solve_home(self) -> np.ndarray:
        """Top-down home at finger (0.30, 0.03, 0.33); fall back to the
        verified joint vector if the numerical IK stalls."""
        if self.ik is None:
            return np.array(self.MW_HOME_JOINTS, dtype=float)
        finger = np.array([0.30, 0.03, 0.33])
        grasp = finger - R_TOP[:, 0] * 0.05 - R_TOP[:, 2] * FINGER_TO_PAD_Z
        guess = np.array([0.10, 1.60, 1.60, -1.50, 0.0, 0.10])
        q = self.ik.solve(grasp, target_mat=R_TOP, guess=guess,
                          use_grasp_site=True, max_restarts=4)
        if q is None:
            q = np.array(self.MW_HOME_JOINTS, dtype=float)
        return np.asarray(q, dtype=float)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def disk_center(self, name: str) -> np.ndarray:
        return self.data.body(self._disk_body_ids[name]).xpos.copy()

    def disk_quat(self, name: str) -> np.ndarray:
        return self.data.body(self._disk_body_ids[name]).xquat.copy()

    def disk_positions(self) -> Dict[str, np.ndarray]:
        return {n: self.disk_center(n) for n in DISK_NAMES}

    def peg_center(self, peg: int) -> np.ndarray:
        return np.array([PEG_XY[peg, 0], PEG_XY[peg, 1], TABLE_Z],
                        dtype=np.float64)

    def disks_on_peg(self, peg: int, tol: float = 0.025) -> List[str]:
        """Disk names whose xy is within ``tol`` of the peg, bottom-to-top."""
        out = []
        for name in DISK_NAMES:
            p = self.disk_center(name)
            if (np.hypot(p[0] - PEG_XY[peg, 0], p[1] - PEG_XY[peg, 1]) < tol
                    and TABLE_Z - 0.02 < p[2] < 0.36):
                out.append(name)
        return sorted(out, key=lambda n: float(self.disk_center(n)[2]))

    def top_disk(self, peg: int, tol: float = 0.025) -> Optional[str]:
        on = self.disks_on_peg(peg, tol)
        return on[-1] if on else None

    def state_from_disks(self, tol: float = 0.025) -> Optional[Tuple[int, int, int]]:
        """Assign every disk to its nearest peg; None if any disk is in
        transit / off the table."""
        state = []
        for name in DISK_NAMES:
            p = self.disk_center(name)
            d = np.hypot(PEG_XY[:, 0] - p[0], PEG_XY[:, 1] - p[1])
            j = int(np.argmin(d))
            if d[j] > tol or p[2] < TABLE_Z - 0.02 or p[2] > 0.36:
                return None
            state.append(j)
        return tuple(state)

    def distance_to_goal(self) -> Optional[int]:
        st = self.state_from_disks()
        if st is None:
            return None
        return S.distance(st, goal=self._goal_state)

    def n_on_goal(self) -> int:
        n = 0
        for d, name in enumerate(DISK_NAMES):
            p = self.disk_center(name)
            if np.hypot(p[0] - PEG_XY[self.goal_peg, 0],
                        p[1] - PEG_XY[self.goal_peg, 1]) < 0.020:
                n += 1
        return n

    def _upright(self, name: str) -> bool:
        q = self.disk_quat(name)  # wxyz
        w, x, y, z = (float(v) for v in q)
        # The ring is rotationally symmetric about its hole axis (body z).
        # Using |qw| alone wrongly rejects a flat disk that settled with a
        # large yaw (e.g. a dropped ring at yaw 108 deg has qw=0.59 but is
        # perfectly flat).  Check the world direction of the body z axis.
        zz = 1.0 - 2.0 * (x * x + y * y)
        return bool(abs(zz) > 0.97)

    def _is_seated(self, name: str) -> bool:
        """Ring threaded on the goal post at the legal height and at rest.

        Unlike ``_placed_on_peg`` this does NOT require the jaws to be open:
        the insert policy often seats the ring and keeps holding it.  The
        caller additionally requires a short dwell to reject transients.
        """
        p = self.disk_center(name)
        if self._goal_peg_xy_dist(name) > 0.002:
            return False
        if not self._upright(name):
            return False
        if abs(float(p[2]) - self._support_z(name)) > self.PLACE_Z_TOL:
            return False
        dof = self.model.body(self._disk_body_ids[name]).dofadr[0]
        if np.linalg.norm(self.data.qvel[dof:dof + 6]) > 0.10:
            return False
        return True

    def success(self) -> bool:
        """All three disks on the goal peg, in order, at rest, released."""
        st = self.state_from_disks(tol=0.020)
        if st is None or st != self._goal_state:
            return False
        if self._knocked:
            return False
        if self._get_gripper_width() < 0.075:
            return False
        for d, name in enumerate(DISK_NAMES):
            p = self.disk_center(name)
            if abs(p[2] - (TABLE_Z + DISK_HALF_T + d * DISK_THICKNESS)) > 0.006:
                return False
            if not self._upright(name):
                return False
            dof = self.model.body(self._disk_body_ids[name]).dofadr[0]
            if np.linalg.norm(self.data.qvel[dof:dof + 6]) > 0.15:
                return False
        return True

    # ------------------------------------------------------------------
    # reset / init state
    # ------------------------------------------------------------------
    def _reset_insert(self):
        """Start from a saved pre-insert state (disk held just above peg C)."""
        if self._insert_templates is None:
            d = np.load(self._insert_template_path)
            self._insert_templates = (d["qpos"], d["qvel"], d["src"])
        qpos, qvel, src = self._insert_templates
        i = int(self.np_random.randint(0, len(qpos)))
        self.data.qpos[:] = qpos[i]
        self.data.qvel[:] = qvel[i]
        mujoco.mj_forward(self.model, self.data)
        for k, adr in enumerate(self._joint_qpos_adrs):
            self.data.ctrl[self._joint_actuator_ids[k]] = self.data.qpos[adr]
        grip = float(self.data.qpos[self._gripper_qpos_adr])
        grip2 = float(self.data.qpos[self._gripper2_qpos_adr])
        self.data.ctrl[self._gripper_actuator_id] = grip
        if self._gripper2_actuator_id is not None:
            self.data.ctrl[self._gripper2_actuator_id] = grip2
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._initial_state = (int(src[i]), int(src[i]), self.goal_peg)
        self._dist_prev = self.distance_to_goal()
        self._init_dist = self._dist_prev
        self._grasp_success = False
        # The disk is already held above C: do not pay the lift bonus again.
        self._lifted_once["disk_s"] = True
        self._lift_hold = self.LIFT_HOLD_STEPS
        self._seated_hold = 0
        return self._get_obs()

    def insert_info(self) -> Dict[str, object]:
        """Sub-goal description for the insert-only expert."""
        return {
            "disk": "disk_s", "disk_idx": 2,
            "src": int(self._initial_state[0])
            if self._initial_state is not None else -1,
            "dst": int(self.goal_peg),
            "grasp": np.zeros(3),
            "final_z": float(TABLE_Z + DISK_HALF_T),
            "preinsert_z": float(self.disk_center("disk_s")[2]),
            "release_z": float(TABLE_Z + DISK_HALF_T + 0.003),
            "peg_x": float(PEG_XY[self.goal_peg, 0]),
            "peg_y": float(PEG_XY[self.goal_peg, 1]),
            "n_dst": 0,
        }

    def _sample_initial_state(self) -> Tuple[int, int, int]:
        if self.skill_stage in ("lift", "transport", "lift_insert"):
            # Single-move skill stage: the full tower starts on A or B and
            # the top (small) disk must be inserted on the goal peg C.
            src = int(self.np_random.randint(0, 2))
            return (src, src, src)
        return S.sample_state(self.np_random, min_dist=self.min_dist,
                              goal=self._goal_state)

    def _set_arm_home(self):
        for i, adr in enumerate(self._joint_qpos_adrs):
            self.data.qpos[adr] = self._home_q[i]
            self.data.ctrl[self._joint_actuator_ids[i]] = self._home_q[i]
        self.data.qpos[self._gripper_qpos_adr] = self.GRIPPER_OPEN
        self.data.qpos[self._gripper2_qpos_adr] = -self.GRIPPER_OPEN
        self.data.ctrl[self._gripper_actuator_id] = self.GRIPPER_OPEN
        if self._gripper2_actuator_id is not None:
            self.data.ctrl[self._gripper2_actuator_id] = -self.GRIPPER_OPEN

    def _place_initial_disks(self, state: Tuple[int, int, int],
                             jitter: bool = True):
        # One shared xy offset per peg keeps each initial stack aligned, plus a
        # tiny per-disk offset so the policy does not see pixel-identical
        # scenes.  The hole is 10% larger than the post (0.7 mm radial
        # clearance), so the total jitter is kept at ~0.2 mm.
        if jitter:
            peg_off = self.np_random.uniform(
                -self._jitter_xy, self._jitter_xy, size=(S.N_PEGS, 2))
        else:
            peg_off = np.zeros((S.N_PEGS, 2), dtype=np.float64)
        for d, name in enumerate(DISK_NAMES):
            peg = int(state[d])
            level = sum(1 for j in range(d) if int(state[j]) == peg)
            z = (TABLE_Z + DISK_HALF_T + 0.0002
                 + level * DISK_THICKNESS)
            if jitter:
                x = PEG_XY[peg, 0] + peg_off[peg, 0] + self.np_random.uniform(
                    -0.00002, 0.00002)
                y = PEG_XY[peg, 1] + peg_off[peg, 1] + self.np_random.uniform(
                    -0.00002, 0.00002)
                yaw = self.np_random.uniform(-self._jitter_yaw,
                                             self._jitter_yaw)
            else:
                x, y, yaw = PEG_XY[peg, 0], PEG_XY[peg, 1], 0.0
            body = self.model.body(self._disk_body_ids[name])
            adr = self.model.jnt_qposadr[body.jntadr[0]]
            self.data.qpos[adr:adr + 3] = [x, y, z]
            self.data.qpos[adr + 3:adr + 7] = [
                np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
            dof = body.dofadr[0]
            self.data.qvel[dof:dof + 6] = 0.0

    def _zero_disk_velocity(self):
        for name in DISK_NAMES:
            dof = self.model.body(self._disk_body_ids[name]).dofadr[0]
            self.data.qvel[dof:dof + 6] = 0.0

    def reset(self):
        self.step_count = 0
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self._knocked = False
        self._success_prev = False
        self._seated_hold = 0
        self._reset_shaping()
        if self.skill_stage == "insert":
            return self._reset_insert()

        # The exact-contact ring stack is stable for almost every jitter draw,
        # but a rare draw can make the top disk slide off during the settle
        # (observed for eval seed 700002).  Retry with a fresh state; if the
        # settle still fails, place the intended state exactly and skip the
        # physics settle so the episode always starts from a legal state.
        settled = False
        for _attempt in range(3):
            self._initial_state = self._sample_initial_state()
            self._set_arm_home()
            self._place_initial_disks(self._initial_state, jitter=True)
            mujoco.mj_forward(self.model, self.data)
            for _ in range(200):
                mujoco.mj_step(self.model, self.data)
            self._zero_disk_velocity()
            mujoco.mj_forward(self.model, self.data)
            if self.state_from_disks() == self._initial_state:
                settled = True
                break
        if not settled:
            self._place_initial_disks(self._initial_state, jitter=False)
            self._zero_disk_velocity()
            mujoco.mj_forward(self.model, self.data)
            if self.state_from_disks() != self._initial_state:
                raise RuntimeError(
                    "hanoi reset could not create a legal initial state: "
                    f"wanted {self._initial_state}, got "
                    f"{self.state_from_disks()}")

        self._dist_prev = self.distance_to_goal()
        self._init_dist = self._dist_prev
        self._grasp_success = False
        return self._get_obs()

    # ------------------------------------------------------------------
    # reward / termination
    # ------------------------------------------------------------------
    def _support_z(self, name):
        """Centre height of the disk if it were resting on the stack below."""
        p = self.disk_center(name)
        level = 0
        for other in DISK_NAMES:
            if other == name:
                continue
            q = self.disk_center(other)
            if (np.hypot(q[0] - p[0], q[1] - p[1]) < 0.020
                    and q[2] < p[2] - 0.005):
                level += 1
        return TABLE_Z + DISK_HALF_T + level * DISK_THICKNESS

    def _is_lifted(self, name):
        """Disk is more than LIFT_CLEAR above its current support."""
        return (float(self.disk_center(name)[2])
                > self._support_z(name) + self.LIFT_CLEAR)

    def _placed_on_peg(self, name):
        """Disk is upright, released, at rest, and at a legal stack height on
        *any* peg.  This is the per-disk insertion predicate used by the
        dense skill reward (the task success predicate is stricter: all three
        on the goal peg)."""
        p = self.disk_center(name)
        d = np.hypot(PEG_XY[:, 0] - p[0], PEG_XY[:, 1] - p[1])
        if float(np.min(d)) > self.PLACE_XY_TOL:
            return False
        support = self._support_z(name)
        if abs(float(p[2]) - support) > self.PLACE_Z_TOL:
            return False
        if not self._upright(name):
            return False
        dof = self.model.body(self._disk_body_ids[name]).dofadr[0]
        if np.linalg.norm(self.data.qvel[dof:dof + 6]) > 0.10:
            return False
        # Released: the jaws are open (same threshold as the success check).
        if self._get_gripper_width() < 0.075:
            return False
        return True

    def _goal_peg_xy_dist(self, name):
        p = self.disk_center(name)
        return float(np.hypot(p[0] - PEG_XY[self.goal_peg, 0],
                              p[1] - PEG_XY[self.goal_peg, 1]))

    def _skill_success(self):
        """Skill-stage termination (lift-only or insert)."""
        if self.skill_stage == "lift":
            return (self._lift_hold >= self.LIFT_HOLD_STEPS
                    and self._is_lifted("disk_s")
                    and self._get_gripper_width() < 0.075)
        if self.skill_stage == "transport":
            p = self.disk_center("disk_s")
            near = (abs(float(p[0]) - PEG_XY[self.goal_peg, 0])
                    < self.TRANSPORT_XY_TOL
                    and abs(float(p[1]) - PEG_XY[self.goal_peg, 1])
                    < self.TRANSPORT_XY_TOL
                    and self.TRANSPORT_Z_MIN < float(p[2])
                    < self.TRANSPORT_Z_MAX)
            return (self._transport_hold >= self.TRANSPORT_HOLD_STEPS
                    and near and self._is_lifted("disk_s")
                    and self._get_gripper_width() < 0.075)
        # Seated and stationary for a few steps: done even if the jaws are
        # still closed around the ring.
        if self._is_seated("disk_s"):
            self._seated_hold += 1
        else:
            self._seated_hold = 0
        if self._seated_hold >= self.SEATED_HOLD_STEPS:
            return True
        return (self._placed_on_peg("disk_s")
                and self._goal_peg_xy_dist("disk_s") <= self.PLACE_XY_TOL)

    def _compute_reward(self):
        disks = self.disk_positions()
        knocked = False
        for name, p in disks.items():
            if (p[2] < TABLE_Z - 0.035 or p[0] < 0.05 or p[0] > 0.67
                    or p[1] < -0.45 or p[1] > 0.45):
                knocked = True
        self._knocked = self._knocked or knocked

        st = self.state_from_disks()
        dist = S.distance(st, goal=self._goal_state) if st is not None else None
        r_task = 0.0
        if (dist is not None and self._dist_prev is not None
                and dist < self._dist_prev):
            r_task = 0.1 * float(self._dist_prev - dist)
        if dist is not None:
            self._dist_prev = dist

        r_lift = r_insert = r_carry = 0.0
        if self.reward_mode == "dense":
            lifted_now = []
            for name in DISK_NAMES:
                lifted = self._is_lifted(name)
                if lifted:
                    lifted_now.append(name)
                    if not self._lifted_once[name]:
                        r_lift += self.LIFT_BONUS
                        self._lifted_once[name] = True
                # Insertion is only rewarded after the robot has lifted this
                # disk, so the initial stacks never pay the bonus.
                if self.skill_stage in ("lift", "transport"):
                    goal_insert = False
                elif self.skill_mode:
                    goal_insert = (name == "disk_s"
                                   and self._goal_peg_xy_dist(name)
                                   <= self.PLACE_XY_TOL)
                else:
                    goal_insert = True
                if (self._lifted_once[name] and not self._inserted_once[name]
                        and goal_insert and self._placed_on_peg(name)):
                    r_insert += self.INSERT_BONUS
                    self._inserted_once[name] = True

            # Speed-neutral quantized potential: while carrying, phi is the
            # lifted disk's xy distance to the goal peg; otherwise phi is the
            # EE-to-nearest-disk xy distance (approach).  One cell per step.
            carry_disk = (max(lifted_now,
                              key=lambda n: float(self.disk_center(n)[2]))
                          if lifted_now else None)
            if carry_disk != self._carry_disk:
                self._carry_disk = carry_disk
                self._phi_anchor = None
            if carry_disk is not None:
                phi = self._goal_peg_xy_dist(carry_disk)
            else:
                ee = self.data.site(self._finger_site_id).xpos
                phi = min(float(np.hypot(ee[0] - q[0], ee[1] - q[1]))
                          for q in disks.values())
            if self._phi_anchor is None:
                self._phi_anchor = phi
            if self._phi_anchor - phi >= self.PHI_CELL:
                r_carry = self.SHAPING_GAIN * self.PHI_CELL
                self._phi_anchor -= self.PHI_CELL

        if self.skill_stage == "lift":
            if (self._is_lifted("disk_s")
                    and self._get_gripper_width() < 0.075):
                self._lift_hold += 1
            else:
                self._lift_hold = 0
        if self.skill_stage == "transport":
            p = self.disk_center("disk_s")
            near = (abs(float(p[0]) - PEG_XY[self.goal_peg, 0])
                    < self.TRANSPORT_XY_TOL
                    and abs(float(p[1]) - PEG_XY[self.goal_peg, 1])
                    < self.TRANSPORT_XY_TOL
                    and self.TRANSPORT_Z_MIN < float(p[2])
                    < self.TRANSPORT_Z_MAX)
            if (near and self._is_lifted("disk_s")
                    and self._get_gripper_width() < 0.075):
                self._transport_hold += 1
            else:
                self._transport_hold = 0

        reward = r_task + r_lift + r_insert + r_carry

        task_succ = False if self.skill_mode else self.success()
        succ = self._skill_success() if self.skill_mode else task_succ
        if succ:
            reward += (self.SKILL_SUCCESS_REWARD if self.skill_mode else 1.0)
            done = True
            self._grasp_success = True
        else:
            done = bool(knocked or self.step_count >= self.max_episode_steps)
            if knocked:
                reward -= 0.5
        self._success_prev = succ

        info = {
            "success": bool(succ),
            "grasp_success": bool(succ),
            "progress": (int(self._init_dist - dist)
                         if (dist is not None and self._init_dist is not None)
                         else 0),
            "dist_to_goal": int(dist) if dist is not None else -1,
            "state": list(st) if st is not None else None,
            "n_on_goal": int(self.n_on_goal()),
            "knocked": bool(self._knocked),
            "disks": {k: v.tolist() for k, v in disks.items()},
            "initial_state": list(self._initial_state)
            if self._initial_state is not None else None,
            "in_box": bool(succ),  # compatibility with the generic eval code
            "reward_mode": self.reward_mode,
            "reward_task": r_task,
            "reward_lift": r_lift,
            "reward_insert": r_insert,
            "reward_carry": r_carry,
            "lifted": [n for n in DISK_NAMES if self._is_lifted(n)],
            "placed": [n for n in DISK_NAMES if self._placed_on_peg(n)],
        }
        return reward, done, info

    # ------------------------------------------------------------------
    # faster point cloud (2 renders instead of depth+rgb+segmentation)
    # ------------------------------------------------------------------
    def _render_point_cloud(self):
        """Depth + RGB -> world points -> crop -> z-based table mask -> FPS.

        The shared parent renderer also does a segmentation pass, but this
        scene's only semantic distinction is table vs object and that is
        available from z: the table top is at 0.19 m while every disk, peg and
        robot part we care about is at/above 0.205 m.  Dropping the
        segmentation pass cuts demo-collection time by ~1/3 on OSMesa.
        """
        if self._pc_renderer is None:
            return np.zeros((self.n_points, 6), dtype=np.float32)
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
            return np.zeros((self.n_points, 6), dtype=np.float32)
        pts_cam = pts_cam[mask]
        R_x180 = np.array([[1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0],
                           [0.0, 0.0, -1.0]], dtype=np.float32)
        R_cw = self.data.cam_xmat[cam_id].reshape(3, 3).astype(np.float32) @ R_x180
        t_cw = self.data.cam_xpos[cam_id].astype(np.float32)
        pts_world = pts_cam @ R_cw.T + t_cw

        xmin, ymin, zmin, xmax, ymax, zmax = self.pc_crop_bounds
        keep = ((pts_world[:, 0] >= xmin) & (pts_world[:, 0] <= xmax) &
                (pts_world[:, 1] >= ymin) & (pts_world[:, 1] <= ymax) &
                (pts_world[:, 2] >= zmin) & (pts_world[:, 2] <= zmax))
        pts_world = pts_world[keep]
        colors = rgb.reshape(-1, 3)[mask][keep].astype(np.float32)
        if len(pts_world) == 0:
            return np.zeros((self.n_points, 6), dtype=np.float32)

        pts = np.concatenate([pts_world, colors], axis=-1)
        is_table = pts_world[:, 2] < 0.205
        table_pts = pts[is_table]
        obj_pts = pts[~is_table]
        n_table = int(round(0.10 * self.n_points))
        if len(table_pts) > n_table:
            table_pts = _fps_downsample(table_pts, n_table, seed=1)
        n_obj = self.n_points - len(table_pts)
        if len(obj_pts) > n_obj:
            obj_pts = _fps_downsample(obj_pts, n_obj, seed=0)
        pts = np.concatenate([obj_pts, table_pts], axis=0)
        if len(pts) < self.n_points:
            pad = np.zeros((self.n_points - len(pts), 6), dtype=np.float32)
            pts = np.concatenate([pts, pad], axis=0)
        return pts.astype(np.float32)

    # parent hook: not used (reset is overridden), kept for API compatibility
    def _randomize_sphere_pose(self):
        pass
