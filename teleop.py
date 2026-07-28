"""Drag-to-move teleoperation for the OpenArm bimanual model.

Adds a draggable mocap target per hand and servos each arm to it with
damped-least-squares IK, so you steer the arms with the mouse instead of
writing ctrl arrays. Targets are injected via MjSpec, so this works on any
scene in v2/ without editing the MJCF.

Controls (mouse):
  double-click a target sphere, then
    Ctrl + right-drag   -> move that hand
    Ctrl + left-drag    -> rotate it

Controls (keyboard, if Ctrl+right-drag is awkward on your pointer):
  1 / 2        choose which hand the arrow keys jog (left / right)
  up / down    move target along +x / -x
  left/right   move target along +y / -y
  pgup / pgdn  move target along +z / -z
  o / p        toggle left / right gripper
  r            reset targets back to the current hand poses
  space        pause physics
"""

import argparse
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

import openarm_mujoco.v2 as openarm_mujoco

ARMS = ("left", "right")
DAMPING = 1e-4  # DLS regularisation
GAIN = 0.6  # fraction of the DLS step taken per tick
MAX_DQ = 0.03  # rad per IK iteration, per joint
IK_ITERS = 1  # IK iterations per physics tick (1 kHz already over-iterates)
JOG_STEP = 0.01  # metres per keyboard jog press
POS_W, ROT_W = 1.0, 0.35  # weight position over orientation


def build(scene: str):
    """Compile the scene with one mocap target per hand."""
    spec = mujoco.MjSpec.from_file(scene)
    for arm, rgba in zip(ARMS, ([0.2, 0.5, 1.0, 0.45], [1.0, 0.35, 0.15, 0.45])):
        body = spec.worldbody.add_body(name=f"target_{arm}", mocap=True)
        # The EE site sits exactly on ee_base_link's origin, i.e. inside the
        # gripper mesh, so a small marker is invisible. A translucent sphere
        # wider than the hand reads as a halo and gives a big click target.
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.055, 0, 0],
            rgba=rgba,
            contype=0,
            conaffinity=0,
            group=1,
        )
        # a small frame marker so orientation is visible while dragging
        body.add_site(
            name=f"target_{arm}_site", size=[0.004, 0.004, 0.004], rgba=rgba
        )
    return spec.compile()


class ArmIK:
    """Damped-least-squares IK for one arm, writing to position-actuator ctrl."""

    def __init__(self, model, data, arm):
        self.m, self.d, self.arm = model, data, arm
        self.site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_ee_control_point"
        )
        if self.site < 0:
            raise ValueError(f"no {arm}_ee_control_point site in this scene")
        self.mocap = model.body_mocapid[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"target_{arm}")
        ]
        jids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{arm}_joint{i}")
            for i in range(1, 8)
        ]
        self.dofs = np.array([model.jnt_dofadr[j] for j in jids], dtype=np.intp)
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids], dtype=np.intp)
        self.lo, self.hi = model.jnt_range[jids].T
        self.act = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}_joint{i}_ctrl")
                for i in range(1, 8)
            ],
            dtype=np.intp,
        )
        self.grip_act = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}_finger1_ctrl"
        )
        # 0 = jaws closed (9 mm), |0.785| = fully open (148 mm). The sign is
        # mirrored between arms: left is [0, +0.785], right is [-0.785, 0].
        lo, hi = model.actuator_ctrlrange[self.grip_act]
        self.grip_closed = 0.0
        self.grip_open = hi if abs(hi) > abs(lo) else lo
        self.closed = False

        self.q_cmd = data.qpos[self.qadr].copy()
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def sync_target_to_hand(self):
        """Park the mocap target on the current hand pose (no jump on start)."""
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.d.site_xmat[self.site])
        self.d.mocap_pos[self.mocap] = self.d.site_xpos[self.site]
        self.d.mocap_quat[self.mocap] = quat

    def toggle_gripper(self):
        self.closed = not self.closed

    def step(self, ghost):
        """Advance the IK solution and write it to ctrl.

        The IK runs on `ghost`, a kinematics-only MjData holding the commanded
        configuration. Solving against the *physical* state instead would
        linearise the Jacobian about a different configuration than the one
        being integrated, which diverges; it would also couple tracking to the
        position actuators' settling time.
        """
        d, m = self.d, self.m
        ghost.qpos[self.qadr] = self.q_cmd

        for _ in range(IK_ITERS):
            mujoco.mj_kinematics(m, ghost)
            mujoco.mj_comPos(m, ghost)

            err = np.empty(6)
            err[:3] = (d.mocap_pos[self.mocap] - ghost.site_xpos[self.site]) * POS_W
            cur = np.zeros(4)
            mujoco.mju_mat2Quat(cur, ghost.site_xmat[self.site])
            mujoco.mju_subQuat(err[3:], d.mocap_quat[self.mocap], cur)
            err[3:] *= ROT_W
            if np.linalg.norm(err) < 1e-6:
                break

            mujoco.mj_jacSite(m, ghost, self._jacp, self._jacr, self.site)
            jac = np.vstack([self._jacp[:, self.dofs], self._jacr[:, self.dofs]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + DAMPING * np.eye(6), err)
            self.q_cmd = np.clip(
                self.q_cmd + np.clip(GAIN * dq, -MAX_DQ, MAX_DQ), self.lo, self.hi
            )
            ghost.qpos[self.qadr] = self.q_cmd

        d.ctrl[self.act] = self.q_cmd
        d.ctrl[self.grip_act] = self.grip_closed if self.closed else self.grip_open


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("xml", nargs="?", default=openarm_mujoco.openarm_pedestal_xml())
    ap.add_argument("--keyframe", "-k", default="home")
    args = ap.parse_args()

    model = build(args.xml)
    data = mujoco.MjData(model)

    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model, data, kid)
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        if jid >= 0:
            data.ctrl[i] = data.qpos[model.jnt_qposadr[jid]]
    mujoco.mj_forward(model, data)

    arms = [ArmIK(model, data, a) for a in ARMS]
    for a in arms:
        a.sync_target_to_hand()
    ghost = mujoco.MjData(model)
    ghost.qpos[:] = data.qpos

    paused = [False]
    active = [0]  # which hand the keyboard jog drives

    # GLFW keycodes
    KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265
    KEY_PGUP, KEY_PGDN = 266, 267
    JOG = {
        KEY_UP: (+JOG_STEP, 0, 0),
        KEY_DOWN: (-JOG_STEP, 0, 0),
        KEY_LEFT: (0, +JOG_STEP, 0),
        KEY_RIGHT: (0, -JOG_STEP, 0),
        KEY_PGUP: (0, 0, +JOG_STEP),
        KEY_PGDN: (0, 0, -JOG_STEP),
    }

    def on_key(code):
        ch = chr(code).lower() if 32 <= code < 127 else ""
        if ch == "o":
            arms[0].toggle_gripper()
        elif ch == "p":
            arms[1].toggle_gripper()
        elif ch == "r":
            for a in arms:
                a.sync_target_to_hand()
        elif ch in ("1", "2"):
            active[0] = int(ch) - 1
            print(f"jogging {ARMS[active[0]]} hand")
        elif code == 32:
            paused[0] = not paused[0]
        elif code in JOG:
            data.mocap_pos[arms[active[0]].mocap] += JOG[code]

    print(__doc__)
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as v:
        v.cam.lookat[:] = model.stat.center
        v.cam.distance = model.stat.extent * 1.2
        while v.is_running():
            t0 = time.time()
            # launch_passive only RECORDS Ctrl+drag into v.perturb; unlike the
            # managed viewer it never applies it, so without this call the
            # target cubes select but never move.
            mujoco.mjv_applyPerturbPose(model, data, v.perturb, 0)
            if not paused[0]:
                for a in arms:
                    a.step(ghost)
                mujoco.mj_step(model, data)
            v.sync()
            time.sleep(max(0, model.opt.timestep - (time.time() - t0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
