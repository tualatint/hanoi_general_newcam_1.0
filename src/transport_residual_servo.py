"""Vision-only residual servos and safety interlocks for the general chain.

The policy handles the gross motion; these helpers act on the same 18-D
vision observation the policy sees (estimated disk centre, virtual ghost,
EE pose, gripper width).  Nothing here reads simulator ground truth, so the
whole stack is deployable on a real robot.

Main pieces:
* terminal transport servo: proportional delta on the handoff-ghost error;
* insert terminal servo: centre-first, speed-capped descent to the seated
  ghost, with an anti-deadlock unlock and an early release above the seat;
* grip safety: never open while carrying, settle after the first closure;
* lift gate: no lateral carry until the ring is clear of the source post.
"""
from __future__ import annotations

import os

import numpy as np

# Action -> displacement scale (m per unit action).  Measured on the current
# 2051 camera/arm setup by applying a fixed action and reading the finger-site
# displacement: 0.20 action -> 13.6 mm on every axis (mw_action_scale 0.10 is
# the command, the PD/IK tracking lag gives ~0.068).  The old fixed-task zarr
# values (0.039/0.032/0.007) under-estimated z by ~10x, which made the insert
# z servo saturate and slam the ring into the post cap.
SERVO_S = np.array([
    float(x) for x in
    os.environ.get("TRANSPORT_SERVO_S", "0.065,0.065,0.065").split(",")
], dtype=np.float64)
SERVO_K = float(os.environ.get("TRANSPORT_SERVO_K", "0.05"))
SERVO_GATE = float(os.environ.get("TRANSPORT_SERVO_GATE", "0.020"))  # m xy
SERVO_Z_GATE = float(os.environ.get("TRANSPORT_SERVO_Z_GATE", "0.008"))
SERVO_CLIP = float(os.environ.get("TRANSPORT_SERVO_CLIP", "0.20"))
# Grasp approach assist (opt-in, vision + proprioception only).  Some
# jittered scenes make the policy hover above the source ring with the jaws
# open (it never commits to the grasp); after GA_STEPS aligned hover steps
# this drives the jaws down to the expert grasp pose and closes them.
GRASP_ASSIST = os.environ.get("TRANSPORT_GRASP_ASSIST", "0") != "0"
GA_XY = float(os.environ.get("TRANSPORT_GRASP_ASSIST_XY_MM", "0.020"))
GA_REL = float(os.environ.get("TRANSPORT_GRASP_ASSIST_REL_MM", "0.010"))
GA_STEPS = int(os.environ.get("TRANSPORT_GRASP_ASSIST_STEPS", "30"))
GA_AZ = float(os.environ.get("TRANSPORT_GRASP_ASSIST_AZ", "-0.12"))
GA_PAD_Z = float(os.environ.get("TRANSPORT_FINGER_TO_PAD_Z", "0.0147564"))
GA_DISK_HALF_T = 0.010
# Grip-force cap: once the jaws contact a ring, keep closing commands from
# squeezing it out (the pads are flat and a ring pinched above its mid-plane
# is ejected downward).  Holds the current opening via the position target.
GRIP_HOLD = os.environ.get("TRANSPORT_GRIP_HOLD", "0") != "0"
GRIP_HOLD_OPEN = float(os.environ.get("TRANSPORT_GRIPPER_OPEN", "0.044"))
# Slip reset: after a confirmed grasp, if the jaw width collapses below the
# ring diameter the ring is no longer held.  Release the closure latch so
# the grasp assist can re-approach and close again (vision + proprioception).
SLIP_RESET = os.environ.get("TRANSPORT_SLIP_RESET", "0") != "0"
# Grasp commit: at the expert grasp pose, force the jaws closed instead of
# letting the policy re-open without the ring (observed: jaws stop at 64 mm
# on a 52 mm ring, then the policy lifts empty-handed).
GRASP_COMMIT = os.environ.get("TRANSPORT_GRASP_COMMIT", "0") != "0"
# Pre-grasp alignment servo: while the jaws are open, servo the finger site
# to the expert grasp pose (disk centre + FINGER_TO_PAD_Z in x) and veto a
# closure that happens more than GRASP_ALIGN_MM off-centre.  Off-centre
# grasps tilt the ring against the 0.67 mm-clearance post and it binds
# during the lift (the residual sigma=12 failure).
GRASP_ALIGN = os.environ.get("TRANSPORT_GRASP_ALIGN", "0") != "0"
GRASP_ALIGN_MM = float(os.environ.get(
    "TRANSPORT_GRASP_ALIGN_MM", "0.006"))
# Lift-stall reset: a slightly tilted small ring can bind on the 60 mm post
# while held; if the vision disk z does not rise while the servo still asks
# for a large lift (ez > 20 mm), open, back off and re-grasp once.
LIFT_STALL = os.environ.get("TRANSPORT_LIFT_STALL", "0") != "0"
LS_STEPS = int(os.environ.get("TRANSPORT_LIFT_STALL_STEPS", "12"))
LS_DZ = float(os.environ.get("TRANSPORT_LIFT_STALL_DZ_MM", "0.002"))
RING_OD = {"disk_l": 0.076, "disk_m": 0.064, "disk_s": 0.052}


def _ring_od(env) -> float:
    """Outer diameter of the ring currently being moved (vision task)."""
    return RING_OD.get(getattr(env, "vision_disk", ""), 0.076)
INSERT_S = np.array([
    float(x) for x in
    os.environ.get("INSERT_SERVO_S", "0.065,0.065,0.065").split(",")
], dtype=np.float64)
# Descent speed cap (normalised action) so the proportional servo cannot slam
# the ring into the post cap when the vertical error is tens of mm.
INSERT_ZMAX = float(os.environ.get("INSERT_ZMAX", "0.06"))
INSERT_ZMAX_NEAR = float(os.environ.get("INSERT_ZMAX_NEAR", "0.03"))
INSERT_ZNEAR = float(os.environ.get("INSERT_ZNEAR", "0.010"))


def _cap_dz(az: float, ez: float) -> float:
    lim = INSERT_ZMAX_NEAR if abs(ez) < INSERT_ZNEAR else INSERT_ZMAX
    return float(np.clip(az, -lim, 0.30))

# Lift-before-carry interlock.  The disks are rings: if the policy starts the
# lateral carry while the ring is still around the source post, the ring jams
# on the post (disk stuck at the source) or is levered off the fingers (drop).
# While a disk is held and the vision vertical error says it is below the
# handoff ghost, suppress lateral motion and keep lifting.  Hysteresis avoids
# chattering near the target height.  Vision-only: uses ``_servo_err_raw``.
LIFT_GATE_ON = float(os.environ.get("TRANSPORT_LIFT_GATE_MM", "0.008"))
# release only once the ring is ~10 mm above the handoff ghost (the expert
# starts the lateral carry at z ~= 276-316, i.e. 8-48 mm above the ghost)
LIFT_GATE_OFF = float(os.environ.get("TRANSPORT_LIFT_GATE_OFF_MM", "-0.010"))
LIFT_MIN_AZ = float(os.environ.get("TRANSPORT_LIFT_MIN_AZ", "0.25"))

# Carry-high rule: while the disk is still far from the target peg (|e_xy|)
# never let the policy descend below the carry ceiling (10 mm above the
# handoff ghost).  The scripted expert crosses the pegs at z ~= 285-290 and
# only descends once it is centred; the BC policy instead descended to the
# handoff height too early, the ring clipped a post top and was levered out
# of the fingers.
CARRY_XY = float(os.environ.get("TRANSPORT_CARRY_XY_MM", "0.015"))
CARRY_UP = float(os.environ.get("TRANSPORT_CARRY_UP_MM", "0.010"))
CARRY_AZ = float(os.environ.get("TRANSPORT_CARRY_AZ", "0.0"))

# Grasp-height window: the BC policy sometimes closes the jaws while still
# tens of mm above the ring (grasping air, then knocking the stack over) or
# too deep.  Only allow the first closure when the finger site is within
# [LO, HI] mm of the vision-estimated disk top (the expert closes at
# +2 mm for disk_s on the table and +8 mm for disk_m at level 1).
GRASP_HI = float(os.environ.get("TRANSPORT_GRASP_GATE_MM", "0.010"))
GRASP_LO = float(os.environ.get("TRANSPORT_GRASP_GATE_LO_MM", "0.0"))
GRASP_CENTER_K = float(os.environ.get("TRANSPORT_GRASP_CENTER_K", "0.5"))


def install(env_cls) -> None:
    if getattr(env_cls, "_transport_residual_servo", False):
        return
    orig_step = env_cls.step
    orig_reset = env_cls.reset

    def _reset(self, *args, **kwargs):
        self._lift_gate_active = False
        self._grasp_closed = False
        self._last_grip_w = 1.0
        self._ins_stall = 0
        self._ins_ez_prev = None
        self._early_rel = 0
        self._closed_once = False
        self._settle_left = 0
        self._center_wait = 0
        self._wedge_stall = 0
        self._wedge_prev = None
        self._wedge_rel = 0
        self._ga_hover = 0
        self._ga_engaged = False
        self._ls_z = None
        self._ls_count = 0
        self._ls_used = 0
        return orig_reset(self, *args, **kwargs)

    def _step(self, action):
        a = np.asarray(action, dtype=np.float64).copy()
        e_raw0 = getattr(self, "_servo_err_raw", None)
        # Centre-first descent permission for the insert stage.  The vision
        # error is noisy (~0.5-1 mm), so a strictly sub-mm centre gate can
        # deadlock: the servo never measures "centred" and z never unlocks
        # (observed: ring held 22 mm above the seat for 120 steps).  Allow a
        # capped descent after INSERT_CENTER_MAX_WAIT centring steps.
        ins_ok = True
        if (getattr(self, "skill_stage", "") == "insert"
                and float(getattr(self, "_servo_valid", 0.0)) > 0.0):
            es = np.asarray(getattr(self, "_servo_err",
                                    np.zeros(3, dtype=np.float32)),
                            dtype=np.float64)
            rxy_now = float(np.hypot(es[0], es[1]))
            cxy = os.environ.get("INSERT_CENTER_XY")
            if cxy is None or cxy == "":
                cxy = min(float(os.environ.get(
                            "INSERT_SERVO_CENTER_XY", "0.0015")),
                          float(os.environ.get(
                            "INSERT_TERMINAL_CENTER_XY", "0.0015")))
            cxy = float(cxy)
            maxw = int(os.environ.get("INSERT_CENTER_MAX_WAIT", "14"))
            if cxy > 0.0 and rxy_now > cxy:
                self._center_wait = int(getattr(self, "_center_wait", 0)) + 1
                ins_ok = (maxw > 0 and self._center_wait >= maxw)
            else:
                self._center_wait = 0
        if (os.environ.get("TRANSPORT_SERVO", "0") != "0"
                and getattr(self, "skill_stage", "") == "transport"
                and float(getattr(self, "_servo_valid", 0.0)) > 0.0
                and e_raw0 is not None
                and 0.030 < float(self._get_gripper_width()) < _ring_od(self) + 0.004):
            # raw (not smoothed) error: the EMA lags and lets a stale large
            # z error saturate the correction
            e = np.asarray(e_raw0, dtype=np.float64)
            if (np.hypot(e[0], e[1]) < SERVO_GATE
                    and abs(float(e[2])) < SERVO_Z_GATE):
                corr = np.clip(SERVO_K * e[:2] / SERVO_S[:2], -SERVO_CLIP,
                               SERVO_CLIP)
                a[:2] = np.clip(a[:2] + corr, -1.0, 1.0)
                if os.environ.get("TRANSPORT_SERVO_Z", "0") != "0":
                    az = float(np.clip(SERVO_K * e[2] / SERVO_S[2],
                                       -SERVO_CLIP, SERVO_CLIP))
                    a[2] = float(np.clip(a[2] + az, -1.0, 1.0))
        # Explicit terminal-phase visual servo: near the handoff ghost, replace
        # the policy's Cartesian delta with a proportional delta on the same
        # vision error.  The policy handles the gross transport; the servo
        # eliminates the terminal steady-state error.  Vision-only.
        k_term = float(os.environ.get("TRANSPORT_TERMINAL_SERVO", "0"))
        if (k_term > 0.0
                and getattr(self, "skill_stage", "") == "transport"
                and float(getattr(self, "_servo_valid", 0.0)) > 0.0
                and e_raw0 is not None
                and 0.030 < float(self._get_gripper_width()) < _ring_od(self) + 0.004):
            e = np.asarray(e_raw0, dtype=np.float64)
            gate_xy = float(os.environ.get(
                "TRANSPORT_TERMINAL_GATE", "0.080"))
            gate_z = float(os.environ.get(
                "TRANSPORT_TERMINAL_Z_GATE", "0.020"))
            if (np.hypot(e[0], e[1]) < gate_xy
                    and abs(float(e[2])) < gate_z):
                # cap the lateral command: a 0.5-gain step on a 60 mm error
                # would otherwise demand ~50 mm of EE motion in one step and
                # can fling a held ring
                xy_clip = float(os.environ.get(
                    "TRANSPORT_TERMINAL_XY_CLIP", "0.25"))
                a[0] = float(np.clip(k_term * e[0] / SERVO_S[0],
                                     -xy_clip, xy_clip))
                a[1] = float(np.clip(k_term * e[1] / SERVO_S[1],
                                     -xy_clip, xy_clip))
                a[2] = _cap_dz(
                    float(np.clip(k_term * e[2] / SERVO_S[2], -1.0, 1.0)),
                    float(e[2]))
        if (os.environ.get("TRANSPORT_GRASP_GATE", "0") != "0"
                and getattr(self, "skill_stage", "") == "transport"
                and float(getattr(self, "_servo_valid", 0.0)) > 0.0
                and not getattr(self, "_grasp_closed", False)
                and getattr(self, "_disk_centre", None) is not None
                and self._finger_site_id is not None):
            fz = float(self.data.site(self._finger_site_id).xpos[2])
            top = float(self._disk_centre[2]) + 0.010
            rel = fz - top
            if rel < GRASP_LO:
                a[3] = -1.0
                a[2] = max(float(a[2]), 0.05)
            elif float(a[3]) > 0.0 and rel > GRASP_HI:
                a[3] = -1.0
                a[2] = min(float(a[2]), -0.05)
            elif float(a[3]) > 0.0:
                a[2] = 0.0
        # Grasp approach assist: if the open jaws stay aligned over the
        # vision-estimated source ring but the policy keeps hovering without
        # committing, drive the expert grasp pose (site = disk centre +
        # FINGER_TO_PAD_Z in x, disk top + 1.5 mm) and close there.
        if (GRASP_ASSIST
                and getattr(self, "skill_stage", "") == "transport"
                and float(getattr(self, "_servo_valid", 0.0)) > 0.0
                and not getattr(self, "_grasp_closed", False)
                and getattr(self, "_disk_centre", None) is not None
                and self._finger_site_id is not None):
            site = self.data.site(self._finger_site_id).xpos
            dc = np.asarray(self._disk_centre, dtype=np.float64)
            gx = float(dc[0]) + GA_PAD_Z
            gy = float(dc[1])
            gz = float(dc[2]) + GA_DISK_HALF_T + 0.0015
            dxy = float(np.hypot(site[0] - gx, site[1] - gy))
            rel = float(site[2] - gz)
            w = float(self._get_gripper_width())
            if dxy < GA_XY and rel > GA_REL and w > _ring_od(self) + 0.004:
                self._ga_hover = int(getattr(self, "_ga_hover", 0)) + 1
            elif rel <= GA_REL:
                self._ga_hover = 0
            if self._ga_hover >= GA_STEPS:
                self._ga_engaged = True
            if getattr(self, "_ga_engaged", False):
                a[0] = float(np.clip(0.5 * (gx - site[0]) / 0.065,
                                     -0.20, 0.20))
                a[1] = float(np.clip(0.5 * (gy - site[1]) / 0.065,
                                     -0.20, 0.20))
                a[3] = -1.0
                a[2] = GA_AZ
                if rel <= 0.006:
                    a[3] = 1.0
                    a[2] = 0.0
                if w < _ring_od(self) + 0.004:
                    self._ga_engaged = False
                    self._ga_hover = 0
        if (GRASP_ALIGN
                and getattr(self, "skill_stage", "") == "transport"
                and float(getattr(self, "_servo_valid", 0.0)) > 0.0
                and not getattr(self, "_grasp_closed", False)
                and getattr(self, "_disk_centre", None) is not None
                and self._finger_site_id is not None):
            site3 = self.data.site(self._finger_site_id).xpos
            dc3 = np.asarray(self._disk_centre, dtype=np.float64)
            gx3 = float(dc3[0]) + GA_PAD_Z
            gy3 = float(dc3[1])
            dxy3 = float(np.hypot(site3[0] - gx3, site3[1] - gy3))
            w3 = float(self._get_gripper_width())
            if w3 > _ring_od(self) + 0.004 and GRASP_ALIGN_MM < dxy3 < 0.030:
                a[0] = float(np.clip(0.5 * (gx3 - site3[0]) / 0.065,
                                     -0.25, 0.25))
                a[1] = float(np.clip(0.5 * (gy3 - site3[1]) / 0.065,
                                     -0.25, 0.25))
                if float(a[3]) > 0.0:
                    a[3] = -1.0
                if float(site3[2] - (dc3[2] + GA_DISK_HALF_T + 0.0015)) > 0.010:
                    a[2] = min(float(a[2]), -0.05)
        if (GRASP_COMMIT
                and getattr(self, "skill_stage", "") == "transport"
                and float(getattr(self, "_servo_valid", 0.0)) > 0.0
                and not getattr(self, "_grasp_closed", False)
                and getattr(self, "_disk_centre", None) is not None
                and self._finger_site_id is not None):
            site2 = self.data.site(self._finger_site_id).xpos
            dc2 = np.asarray(self._disk_centre, dtype=np.float64)
            gx2 = float(dc2[0]) + GA_PAD_Z
            gy2 = float(dc2[1])
            gz2 = float(dc2[2]) + GA_DISK_HALF_T + 0.0015
            dxy2 = float(np.hypot(site2[0] - gx2, site2[1] - gy2))
            rel2 = float(site2[2] - gz2)
            od2 = RING_OD.get(getattr(self, "vision_disk", ""), 0.076)
            w2 = float(self._get_gripper_width())
            if (dxy2 < GA_XY and -0.004 < rel2 < 0.010
                    and w2 > od2 + 0.004):
                a[3] = 1.0
                a[2] = min(float(a[2]), 0.0)
        e_raw = getattr(self, "_servo_err_raw", None)
        if (os.environ.get("TRANSPORT_LIFT_GATE", "0") != "0"
                and getattr(self, "skill_stage", "") == "transport"
                and float(getattr(self, "_servo_valid", 0.0)) > 0.0
                and e_raw is not None
                and 0.030 < float(self._get_gripper_width()) < _ring_od(self) + 0.004):
            ez = float(np.asarray(e_raw, dtype=np.float64)[2])
            active = bool(getattr(self, "_lift_gate_active", False))
            if active and ez < LIFT_GATE_OFF:
                active = False
            elif not active and ez > LIFT_GATE_ON:
                active = True
            self._lift_gate_active = active
            if active:
                a[:2] = 0.0
                a[2] = max(float(a[2]), LIFT_MIN_AZ)
        else:
            self._lift_gate_active = False
        if (os.environ.get("TRANSPORT_CARRY_GATE", "0") != "0"
                and getattr(self, "skill_stage", "") == "transport"
                and float(getattr(self, "_servo_valid", 0.0)) > 0.0
                and e_raw is not None
                and 0.030 < float(self._get_gripper_width()) < _ring_od(self) + 0.004):
            e = np.asarray(e_raw, dtype=np.float64)
            exy = float(np.hypot(e[0], e[1]))
            if exy > CARRY_XY and float(e[2]) > -CARRY_UP:
                a[2] = max(float(a[2]), CARRY_AZ)
        # Track the first grasp (independent of the optional force-close):
        # a real grasp stops the jaws in the 30-82 mm band; closing on air
        # keeps shrinking (88 -> 7 mm), so require the width to be stable.
        if getattr(self, "skill_stage", "") == "transport":
            w = float(self._get_gripper_width())
            w_prev = float(getattr(self, "_last_grip_w", 1.0))
            # Contact width is the ring outer diameter (plus 4 mm): the old
            # 82 mm upper bound marked the grasp "closed" while the jaws were
            # still 30 mm from a small ring, so the force-hold/grip-hold
            # interlocks stalled the closure on air.
            ring_od = {"disk_l": 0.076, "disk_m": 0.064,
                       "disk_s": 0.052}.get(
                           getattr(self, "vision_disk", ""), 0.076)
            geom_ok = True
            if (getattr(self, "_disk_centre", None) is not None
                    and self._finger_site_id is not None):
                site_g = self.data.site(self._finger_site_id).xpos
                dc_g = np.asarray(self._disk_centre, dtype=np.float64)
                dxy_g = float(np.hypot(
                    site_g[0] - (float(dc_g[0]) + GA_PAD_Z),
                    site_g[1] - float(dc_g[1])))
                rel_g = float(site_g[2] - (float(dc_g[2])
                                           + GA_DISK_HALF_T + 0.0015))
                # the ring must actually be between the pads, otherwise a
                # free closure settling at ~50 mm would latch "grasped" on air
                geom_ok = dxy_g < 0.010 and abs(rel_g) < 0.012
            if not getattr(self, "_grasp_closed", False):
                if (ring_od - 0.006 < w < min(0.082, ring_od + 0.004)
                        and abs(w - w_prev) < 0.002 and geom_ok):
                    self._grasp_closed = True
            if (SLIP_RESET and getattr(self, "_grasp_closed", False)
                    and w < ring_od - 0.010):
                # ring slipped out of the jaws: reopen and let the assist
                # re-approach the vision-estimated ring
                self._grasp_closed = False
                # Re-engage the grasp assist immediately: the policy tends to
                # stay high after a slip and would not re-approach within the
                # remaining budget.
                self._ga_hover = GA_STEPS
                self._ga_engaged = True
                a[3] = -1.0
            self._last_grip_w = w
        # Pre-grasp xy centering: put the open jaws over the vision-estimated
        # ring before closing.  Off-centre grasps tilt the ring against the
        # source post, it jams and the arm cannot lift (the dominant
        # residual failure).
        if (os.environ.get("TRANSPORT_GRASP_CENTER", "0") != "0"
                and getattr(self, "skill_stage", "") == "transport"
                and float(getattr(self, "_servo_valid", 0.0)) > 0.0
                and not getattr(self, "_grasp_closed", False)
                and getattr(self, "_disk_centre", None) is not None
                and float(self._get_gripper_width()) > 0.082):
            site = self.data.site(self._finger_site_id).xpos
            dx = float(self._disk_centre[0]) - float(site[0])
            dy = float(self._disk_centre[1]) - float(site[1])
            rel_c = float(site[2]) - (float(self._disk_centre[2]) + 0.010)
            if np.hypot(dx, dy) < 0.030 and rel_c > 0.008:
                a[0] = float(np.clip(a[0] + GRASP_CENTER_K * dx / SERVO_S[0],
                                     -0.5, 0.5))
                a[1] = float(np.clip(a[1] + GRASP_CENTER_K * dy / SERVO_S[1],
                                     -0.5, 0.5))
        # Never release a grasped disk during transport (optional).
        if (LIFT_STALL
                and getattr(self, "skill_stage", "") == "transport"
                # only when the jaws hold a ring LARGER than the target
                # (wrong-ring grasp of the disk below in a stack); a normal
                # grasp has width ~= the target ring OD
                and float(self._get_gripper_width()) > _ring_od(self) + 0.008
                and getattr(self, "_disk_centre", None) is not None
                and getattr(self, "_servo_err", None) is not None
                and float(np.asarray(self._servo_err)[2]) > 0.020):
            z_now = float(np.asarray(self._disk_centre)[2])
            if self._ls_z is not None and (z_now - self._ls_z) > LS_DZ:
                self._ls_count = 0
            else:
                self._ls_count = int(getattr(self, "_ls_count", 0)) + 1
            self._ls_z = z_now
            if self._ls_count >= LS_STEPS and self._ls_used < 2:
                # ring bound on the post: reopen, lift, re-approach
                self._ls_used += 1
                self._ls_count = 0
                self._grasp_closed = False
                self._ga_hover = GA_STEPS
                self._ga_engaged = True
                a[3] = -1.0
                a[2] = 0.10
        if (os.environ.get("TRANSPORT_FORCE_CLOSE", "0") != "0"
                and getattr(self, "skill_stage", "") == "transport"
                and getattr(self, "_grasp_closed", False)):
            a[3] = 1.0
        if (GRIP_HOLD
                and getattr(self, "skill_stage", "") == "transport"
                and getattr(self, "_grasp_closed", False)):
            w_hold = float(self._get_gripper_width())
            if 0.030 < w_hold < 0.082:
                a[3] = float(np.clip(w_hold / GRIP_HOLD_OPEN - 1.0,
                                     -0.9, 0.9))
        k_in = float(os.environ.get("INSERT_SERVO", "0"))
        if (k_in > 0.0
                and getattr(self, "skill_stage", "") == "insert"
                and float(getattr(self, "_servo_valid", 0.0)) > 0.0):
            e = np.asarray(self._servo_err, dtype=np.float64)
            rxy = float(np.hypot(e[0], e[1]))
            if rxy < 0.050 and abs(float(e[2])) < 0.100:
                a[:2] = a[:2] + k_in * e[:2] / INSERT_S[:2]
                if not ins_ok:
                    # centre first: never descend while off-centre
                    a[2] = 0.0
                else:
                    a[2] = _cap_dz(
                        float(a[2]) + k_in * e[2] / INSERT_S[2],
                        float(e[2]))
                a[:3] = np.clip(a[:3], -1.0, 1.0)
        # Explicit insert terminal servo: near the seat, drive the ring to the
        # virtual target with a proportional delta on the same vision error.
        # No centre-first freeze: the xy and z terms act together.
        k_it = float(os.environ.get("INSERT_TERMINAL_SERVO", "0"))
        if (k_it > 0.0
                and getattr(self, "skill_stage", "") == "insert"
                and float(getattr(self, "_servo_valid", 0.0)) > 0.0):
            e = np.asarray(self._servo_err, dtype=np.float64)
            gxy = float(os.environ.get("INSERT_TERMINAL_GATE", "0.010"))
            gz = float(os.environ.get("INSERT_TERMINAL_Z_GATE", "0.030"))
            if (np.hypot(e[0], e[1]) < gxy
                    and abs(float(e[2])) < gz):
                a[0] = float(np.clip(k_it * e[0] / INSERT_S[0],
                                     -1.0, 1.0))
                a[1] = float(np.clip(k_it * e[1] / INSERT_S[1],
                                     -1.0, 1.0))
                if not ins_ok:
                    a[2] = 0.0
                else:
                    a[2] = _cap_dz(
                        float(k_it * e[2] / INSERT_S[2]), float(e[2]))
        # Early release on the seat approach: the expert opens the jaws ~3 mm
        # above the seat and lets the ring drop onto the rounded post cap,
        # which self-centres it.  Pressing a held ring onto the cap can wedge
        # it and eject it sideways (observed: ring flung 380 mm, 90 deg tilt).
        # Vision + proprioception only.
        release_eta = float(os.environ.get(
            "INSERT_RELEASE_EARLY_MM", "0"))
        if (release_eta > 0.0
                and getattr(self, "skill_stage", "") == "insert"
                and float(getattr(self, "_servo_valid", 0.0)) > 0.0):
            e = np.asarray(self._servo_err, dtype=np.float64)
            w = float(self._get_gripper_width())
            rxy = float(np.hypot(e[0], e[1]))
            if (w > 0.030 and rxy < 0.0010
                    and -release_eta < float(e[2]) < 0.0010):
                self._early_rel = 1
            if getattr(self, "_early_rel", 0):
                a[3] = -1.0
                a[2] = 0.03
                if w > 0.082:
                    self._early_rel = 0
        # Terminal release: the BC insert policy sometimes keeps the jaws
        # closed while pressing a centred ring onto the post cap, so the ring
        # cannot slide down (the fingers hold it).  The scripted expert has an
        # OPEN phase for this; detect the vertical stall from the vision error
        # (no z progress while centred) and execute the release: open the
        # jaws and lift, letting the ring drop onto the peg.  Vision-only.
        if (os.environ.get("INSERT_RELEASE", "0") != "0"
                and getattr(self, "skill_stage", "") == "insert"
                and float(getattr(self, "_servo_valid", 0.0)) > 0.0):
            e = np.asarray(self._servo_err, dtype=np.float64)
            w = float(self._get_gripper_width())
            rxy = float(np.hypot(e[0], e[1]))
            if w > 0.030 and rxy < 0.010 and abs(float(e[2])) < 0.010:
                prev = getattr(self, "_ins_ez_prev", None)
                if prev is not None and abs(float(e[2]) - prev) < 0.0008:
                    self._ins_stall = getattr(self, "_ins_stall", 0) + 1
                else:
                    self._ins_stall = 0
                self._ins_ez_prev = float(e[2])
                if self._ins_stall >= 10:
                    a[3] = -1.0
                    a[2] = max(float(a[2]), 0.08)
            else:
                self._ins_stall = 0
                self._ins_ez_prev = None
        # Wedge release: a ring grasped with a few degrees of tilt can jam on
        # the post cylinder above the seat (observed: 8 deg at handoff, ring
        # held 38 mm above the seat for the whole 120-step budget).  Opening
        # the jaws while lifting from that state lets the ring drop and
        # self-level on the post; it seats within ~5 steps.  Only trigger
        # when the ring is centred, the descent has stalled, and the ring is
        # around the post (8-45 mm above the seat), never while it is still
        # above the post cap.  Vision + proprioception only.
        if (os.environ.get("INSERT_WEDGE_RELEASE", "1") != "0"
                and getattr(self, "skill_stage", "") == "insert"
                and float(getattr(self, "_servo_valid", 0.0)) > 0.0):
            e = np.asarray(self._servo_err, dtype=np.float64)
            w = float(self._get_gripper_width())
            rxy = float(np.hypot(e[0], e[1]))
            ez = float(e[2])
            wzmax = float(os.environ.get(
                "INSERT_WEDGE_ZMAX_MM", "0.045"))
            wxy = float(os.environ.get(
                "INSERT_WEDGE_XY_MM", "0.0020"))
            if (w > 0.030 and rxy < wxy
                    and 0.008 < -ez < wzmax):
                prev = getattr(self, "_wedge_prev", None)
                if prev is not None and abs(ez - prev) < 0.0005:
                    self._wedge_stall = int(getattr(
                        self, "_wedge_stall", 0)) + 1
                else:
                    self._wedge_stall = 0
                self._wedge_prev = ez
                if self._wedge_stall >= 6:
                    self._wedge_rel = 1
            else:
                self._wedge_stall = 0
                self._wedge_prev = None
            if getattr(self, "_wedge_rel", 0):
                a[3] = -1.0
                a[2] = 0.05
                if w > 0.082:
                    self._wedge_rel = 0
        return orig_step(self, a.astype(np.float32))

    env_cls.step = _step
    env_cls.reset = _reset
    env_cls._transport_residual_servo = True
