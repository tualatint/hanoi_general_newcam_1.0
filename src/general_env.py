"""GeneralVservoEnv -- random top disk -> random target peg, no Hanoi rule.

At each reset:
- sample an arbitrary arrangement of the 3 disks on the 3 pegs (no size rule);
- pick a random non-empty source peg and its top disk;
- pick a different target peg;
- the target level is the number of disks already on that peg (0/1/2);
- the ghost is placed on the target post at that level.

The moving disk and target/level are set on the env so the existing
``VservoEnv`` vision/ghost machinery works unchanged (the disk radius and
ghost height vary per episode).  The base ``_compute_reward`` still has some
disk_s-specific sparse terms; they are not used by BC and the termination is
overridden here.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

from hanoi_env import (  # noqa: F401
    DISK_HALF_T, DISK_NAMES, PEG_XY, TABLE_Z, HanoiEnv)
from vservo_env import GHOST_G, VservoEnv
from vision_pose import sample_disk_model

# Matches expert_hanoi.PEG_TOP_Z (TABLE_Z + 0.060): the expert clamps the
# pre-insert height just above the post cap for every target level.
PEG_TOP_Z = TABLE_Z + 0.060


class GeneralVservoEnv(VservoEnv):
    """One move: top disk of a random source peg -> random target peg."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = None  # (disk_idx, src_peg, dst_peg, target_level)
        self._gen_hold = 0
        pose_path = os.environ.get("HANOI_CAMERA_POSE", "")
        if pose_path:
            from camera_pose import apply_camera_pose
            apply_camera_pose(self, pose_path)

    # ------------------------------------------------------------------
    # random task / initial arrangement
    # ------------------------------------------------------------------
    def _sample_initial_state(self):
        state = tuple(int(self.np_random.randint(0, 3)) for _ in range(3))
        nonempty = [p for p in range(3) if p in state]
        if not nonempty:  # cannot happen for 3 disks x 3 pegs, kept safe
            state = (0, 0, 0)
            nonempty = [0]
        src = nonempty[int(self.np_random.randint(0, len(nonempty)))]
        src_disk = max(d for d in range(3) if state[d] == src)
        others = [p for p in range(3) if p != src]
        dst = others[int(self.np_random.randint(0, len(others)))]
        level = sum(1 for d in range(3) if state[d] == dst)

        self.task = (int(src_disk), int(src), int(dst), int(level))
        self.vision_disk = DISK_NAMES[src_disk]
        self.goal_peg = int(dst)
        self.ghost_level = int(level)
        self._goal_state = tuple(
            dst if d == src_disk else state[d] for d in range(3))
        # ghost ring model follows the moving disk's outer radius
        self._ghost_local = sample_disk_model(
            self.vision_disk, n=GHOST_G, rng=self._vision_rng)
        self._gen_hold = 0
        return state

    # ------------------------------------------------------------------
    # task setup / handoff-state restore (insert demos and chain eval)
    # ------------------------------------------------------------------
    def set_task(self, disk_idx, src, dst, level, initial_state=None):
        """Install a task without sampling a new initial arrangement."""
        self.task = (int(disk_idx), int(src), int(dst), int(level))
        d, _src, dst, level = self.task
        self.vision_disk = DISK_NAMES[d]
        self.goal_peg = int(dst)
        self.ghost_level = int(level)
        if initial_state is not None:
            init = tuple(int(x) for x in initial_state)
            self._goal_state = tuple(
                dst if i == d else init[i] for i in range(3))
        else:
            self._goal_state = (dst, dst, dst)
        # ghost ring model follows the moving disk's outer radius
        self._ghost_local = sample_disk_model(
            self.vision_disk, n=GHOST_G, rng=self._vision_rng)
        self._gen_hold = 0

    def insert_info(self):
        """Task-aware sub-goal description for the insert-only expert."""
        if self.task is None:
            return super().insert_info()
        d, src, dst, level = self.task
        name = DISK_NAMES[d]
        final_z = TABLE_Z + DISK_HALF_T + level * 2.0 * DISK_HALF_T
        return {
            "disk": name, "disk_idx": int(d), "src": int(src),
            "dst": int(dst), "grasp": np.zeros(3, dtype=np.float64),
            "final_z": float(final_z),
            "preinsert_z": float(self.disk_center(name)[2]),
            "release_z": float(final_z + 0.003),
            "peg_x": float(PEG_XY[int(dst), 0]),
            "peg_y": float(PEG_XY[int(dst), 1]),
            "n_dst": int(level),
        }

    def restore_handoff(self, qpos, qvel, task, initial_state=None,
                        post_axis=None, table_z=None, post_z_min=None):
        """Restore a held pre-insert state and install its general task.

        The policy observation is still the vision-only 18-D agent_pos; the
        saved qpos/qvel/task are used only to initialise the simulator state
        for demo collection / evaluation.  ``post_axis`` / ``table_z`` are
        vision estimates from the reset frame (arm at home, moving disk still
        on the source peg); they let the insert skill start when the held disk
        occludes the target post.  They are not simulator GT.
        """
        d, src, dst, level = (int(x) for x in task)
        name = DISK_NAMES[d]
        self.data.qpos[:] = np.asarray(qpos, dtype=np.float64)
        self.data.qvel[:] = np.asarray(qvel, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)
        for k, adr in enumerate(self._joint_qpos_adrs):
            self.data.ctrl[self._joint_actuator_ids[k]] = self.data.qpos[adr]
        grip = float(self.data.qpos[self._gripper_qpos_adr])
        grip2 = float(self.data.qpos[self._gripper2_qpos_adr])
        self.data.ctrl[self._gripper_actuator_id] = grip
        if self._gripper2_actuator_id is not None:
            self.data.ctrl[self._gripper2_actuator_id] = grip2
        if initial_state is not None:
            self._initial_state = tuple(int(x) for x in initial_state)
        else:
            self._initial_state = (src, src, dst)
        # The env.reset() before restore sampled a different arrangement, so
        # the previous disk centre / servo error belong to another scene.  The
        # estimator's 60 mm jump gate would then reject every restored frame.
        self._reset_vision()
        self.set_task(d, src, dst, level,
                      initial_state=self._initial_state)
        if post_axis is not None and np.all(np.isfinite(post_axis)):
            self._post_axis = np.asarray(
                post_axis, dtype=np.float64).copy()
        if table_z is not None and np.isfinite(table_z):
            self._table_z = float(table_z)
        if post_z_min is not None and np.isfinite(post_z_min):
            self._post_z_min = float(post_z_min)
        self._reset_shaping()
        self._lifted_once[name] = True
        self._grasp_success = False
        self._seated_hold = 0
        self._servo_integral = np.zeros(self.N_INT, dtype=np.float32)
        self._centered_once = False
        self._phi_prev = None
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def distance_to_goal(self):
        st = self.state_from_disks()
        if st is None or self.task is None:
            return None
        return 0 if int(st[self.task[0]]) == int(self.task[2]) else 1

    # ------------------------------------------------------------------
    # general success predicates
    # ------------------------------------------------------------------
    def _skill_success(self):
        if self.task is None:
            return False
        d, _src, dst, level = self.task
        name = DISK_NAMES[d]
        if self.skill_stage == "grasp":
            # the ring is grasped and lifted clear of its stack
            ok = self._is_lifted(name)
            self._gen_hold = self._gen_hold + 1 if ok else 0
            return (self._gen_hold >= self.LIFT_HOLD_STEPS
                    and self._get_gripper_width() < 0.085)
        if self.skill_stage == "transport":
            p = self.disk_center(name)
            # pre-insert height is the same for every target level (clamped
            # above the post cap); the level only affects the seated ghost
            z_lo = self.HANDOFF_Z
            z_hi = z_lo + self.HANDOFF_Z_UP
            near = (abs(float(p[0]) - PEG_XY[dst, 0])
                    < self.TRANSPORT_XY_TOL
                    and abs(float(p[1]) - PEG_XY[dst, 1])
                    < self.TRANSPORT_XY_TOL
                    and z_lo < float(p[2]) < z_hi
                    and self._is_lifted(name)
                    # fully open is 0.088; a held disk_l is ~0.076, which the
                    # old 0.075 "closed" threshold wrongly classified as open
                    and self._get_gripper_width() < 0.085)
            self._gen_hold = self._gen_hold + 1 if near else 0
            return self._gen_hold >= self.TRANSPORT_HOLD_STEPS
        # seated on the target stack (insert / full move)
        if self._is_seated(name):
            self._gen_hold += 1
        else:
            self._gen_hold = 0
        if self._gen_hold >= self.SEATED_HOLD_STEPS:
            return True
        return (self._placed_on_peg(name)
                and self._goal_peg_xy_dist(name) <= self.PLACE_XY_TOL)
