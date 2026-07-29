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

Controls (phones, with --phone):
  Two phones, one per hand; each page is tinted to match its target sphere.
  Translation and rotation are separate gestures, so neither disturbs the other:
    drag on the pad     -> move the hand (one finger x/y, two fingers z)
    ROTATE toggle       -> tilt turns the hand, relative to the attitude at the
                           tap; dragging stays live, so you can do both at once
    Hand / World toggle -> whether "forward" follows the gripper or the room
    HOLD TO GRIP        -> squeeze the jaws, release to open
  Each target carries a long arrow along its approach axis and a short one on
  the hand's up axis, so orientation and roll are both visible while driving.
"""

import argparse
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

import openarm_mujoco.v2 as openarm_mujoco

ARMS = ("left", "right")
# Per-arm target colours. The phone pages tint themselves to match, so it is
# obvious at a glance which handset drives which hand.
TARGET_COLORS = ([0.2, 0.5, 1.0, 0.45], [1.0, 0.35, 0.15, 0.45])
DAMPING = 1e-4  # baseline DLS regularisation
SIGMA_EPS = 0.08  # smallest singular value below which damping ramps up
LAMBDA_MAX = 0.05  # extra damping at a full singularity
ERR_CLAMP = 0.03  # m of position error fed to the solver per tick
ROT_CLAMP = 0.20  # rad of orientation error fed to the solver per tick
NULL_GAIN = 0.01  # pull of the redundant DoF back to the rest posture
GAIN = 0.10  # per-tick loop gain; this is kp*dt, so ~0.1 at 1 kHz.
# Larger values (0.6 was the first guess) drive dq into the MAX_DQ clip every
# tick and the arm bang-bangs -- that reads as IK shakiness.
MAX_DQ = 0.03  # rad per IK iteration, per joint
IK_ITERS = 1  # IK iterations per physics tick (1 kHz already over-iterates)
JOG_STEP = 0.01  # metres per keyboard jog press
FRAME_HZ = 60  # render rate; physics steps in a burst between frames
POS_W, ROT_W = 1.0, 0.35  # weight position over orientation


def build(scene: str):
    """Compile the scene with one mocap target per hand."""
    spec = mujoco.MjSpec.from_file(scene)
    for arm, rgba in zip(ARMS, TARGET_COLORS):
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
        self.q_rest = self.q_cmd.copy()
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
            # mju_subQuat returns the error in the CURRENT BODY frame
            # (qb*quat(res) = qa), but mj_jacSite's rotational block is in the
            # world frame. Rotate it out; at the home pose the two differ by
            # 90 degrees, so feeding it raw drives rotation sideways and the
            # hand oscillates instead of converging.
            rot_err = np.zeros(3)
            mujoco.mju_subQuat(rot_err, d.mocap_quat[self.mocap], cur)
            err[3:] = ghost.site_xmat[self.site].reshape(3, 3) @ rot_err
            err[3:] *= ROT_W
            if np.linalg.norm(err) < 1e-6:
                break

            # Cap how far the solver is asked to travel in one tick. A fast
            # drag would otherwise request a metre-scale step and saturate the
            # per-joint clip, which reads as thrashing.
            pn = np.linalg.norm(err[:3])
            if pn > ERR_CLAMP:
                err[:3] *= ERR_CLAMP / pn
            rn = np.linalg.norm(err[3:])
            if rn > ROT_CLAMP:
                err[3:] *= ROT_CLAMP / rn

            mujoco.mj_jacSite(m, ghost, self._jacp, self._jacr, self.site)
            jac = np.vstack([self._jacp[:, self.dofs], self._jacr[:, self.dofs]])

            # Adaptive damping. Fixed small lambda amplifies near-singular
            # directions without bound; this arm's smallest singular value
            # falls from ~0.13 to ~0 as it extends, so lambda has to grow as
            # the configuration degenerates.
            sigma_min = np.linalg.svd(jac, compute_uv=False)[-1]
            lam = DAMPING
            if sigma_min < SIGMA_EPS:
                lam += LAMBDA_MAX * (1.0 - (sigma_min / SIGMA_EPS) ** 2)

            dq = jac.T @ np.linalg.solve(jac @ jac.T + lam * np.eye(6), err)

            # Null-space term: bleed the redundant 7th DoF back toward the
            # rest posture so it does not wander into limits over time.
            null = np.eye(7) - jac.T @ np.linalg.solve(
                jac @ jac.T + lam * np.eye(6), jac
            )
            dq += null @ (NULL_GAIN * (self.q_rest - self.q_cmd))

            self.q_cmd = np.clip(
                self.q_cmd + np.clip(GAIN * dq, -MAX_DQ, MAX_DQ), self.lo, self.hi
            )
            ghost.qpos[self.qadr] = self.q_cmd

        d.ctrl[self.act] = self.q_cmd
        d.ctrl[self.grip_act] = self.grip_closed if self.closed else self.grip_open


# --- phone teleop ---------------------------------------------------------
# One phone per hand, driving the same mocap targets the mouse and keyboard
# drive. Motion is relative to the pose held when the clutch engaged, so the
# hand never snaps to the phone's absolute attitude and gyro yaw drift is
# reset on every press instead of accumulating.
PHONE_MAX_REACH = 0.9  # m the target may stray from where it started
PHONE_MAX_STEP = 0.15  # m of drag accepted in one frame before assuming a reset
# At 20 Hz, 0.15 m per frame is 3 m/s of thumb travel -- far beyond a real
# drag, so anything larger means the phone's running total restarted (page
# reload, new tab) rather than that the operator moved.
PHONE_ROT_SMOOTH = 0.08  # per-tick blend toward the phone's attitude
PHONE_ROT_DEADBAND = 3.0  # deg of wrist wobble ignored while ROTATE is on
# With drag and rotate live at once, the small tilt your hand makes while
# thumbing the trackpad would otherwise turn the gripper. The deadband is
# subtracted continuously rather than gated, so crossing it does not jump;
# set it to 0 for a strict 1:1 mapping.
PHONE_POS_SMOOTH = 0.05  # per-tick fraction of pending drag applied
# Both smoothing constants are per physics tick. At 1 kHz they give roughly a
# 12-20 ms time constant, enough to hide the 20 Hz staircase of arriving
# frames without adding lag you can feel.


def orientation_quat(pitch: float, roll: float, yaw: float) -> np.ndarray:
    """Convert DeviceOrientation angles in degrees to a quaternion.

    Uses the intrinsic Z-X'-Y'' order from the W3C DeviceOrientation spec,
    where beta is X (pitch), gamma is Y (roll) and alpha is Z (yaw).
    """
    half = np.radians([pitch, roll, yaw]) / 2.0
    (cx, cy, cz), (sx, sy, sz) = np.cos(half), np.sin(half)
    return np.array(
        [
            cx * cy * cz - sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz + sx * sy * cz,
        ]
    )


def hand_basis(quat: np.ndarray) -> np.ndarray:
    """Return the hand's (forward, left, up) axes as matrix columns.

    Forward is the approach axis, which for this model's ee_control_point site
    is local **-z**: the fingers lie along -z, so +z points back into the wrist.
    Left and up are then chosen to keep up as close to world up as possible, so
    at the home pose this basis coincides with the world axes and switching
    frames there changes nothing.
    """
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    forward = -mat.reshape(3, 3)[:, 2]

    reference = np.array([0.0, 0.0, 1.0])
    left = np.cross(reference, forward)
    if np.linalg.norm(left) < 1e-6:
        # Pointing straight up or down leaves the heading undefined; fall back
        # to the world x axis so the basis stays well conditioned.
        left = np.cross(np.array([1.0, 0.0, 0.0]), forward)
    left /= np.linalg.norm(left)

    return np.column_stack([forward, left, np.cross(forward, left)])


def direction_frame(direction: np.ndarray) -> np.ndarray:
    """Build a rotation whose local +z is `direction`, for mjv arrow geoms."""
    direction = direction / np.linalg.norm(direction)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, direction)) > 0.99:
        helper = np.array([1.0, 0.0, 0.0])
    first = np.cross(helper, direction)
    first /= np.linalg.norm(first)
    return np.column_stack([first, np.cross(direction, first), direction])


def shrink_quat(rotation: np.ndarray, deadband_rad: float) -> np.ndarray:
    """Reduce a rotation's angle by `deadband_rad`, keeping its axis.

    Subtracting rather than gating keeps the mapping continuous: a wobble just
    past the deadband produces a tiny rotation instead of snapping to the full
    angle, which is what a plain threshold would do.
    """
    if deadband_rad <= 0.0:
        return rotation

    # Take the short arc, so a sign-flipped quaternion is not read as ~360 deg.
    signed = rotation if rotation[0] >= 0.0 else -rotation
    angle = 2.0 * np.arccos(np.clip(signed[0], -1.0, 1.0))
    if angle <= deadband_rad:
        return np.array([1.0, 0.0, 0.0, 0.0])

    axis = signed[1:]
    if (norm := np.linalg.norm(axis)) < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])

    reduced = np.zeros(4)
    mujoco.mju_axisAngle2Quat(reduced, axis / norm, angle - deadband_rad)
    return reduced


def blend_quat(current: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    """Blend toward `target` along the short arc, renormalising the result.

    Negating an antipodal target first keeps the blend on the short way round;
    without it a sign flip in the phone's attitude spins the hand the long way.
    """
    if np.dot(current, target) < 0.0:
        target = -target
    blended = (1.0 - alpha) * current + alpha * target
    return blended / np.linalg.norm(blended)


class PhoneDriver:
    """Steer one arm's mocap target from that arm's phone.

    Translation and rotation are deliberately separate: dragging on the phone
    moves the hand and never turns it, holding ROTATE turns the hand and never
    moves it. Neither path integrates acceleration, so a phone left untouched
    leaves the target exactly where it was.
    """

    def __init__(self, arm_ik, data, controller=None):
        self.ik = arm_ik
        self.d = data
        self.controller = controller
        self.home = data.mocap_pos[arm_ik.mocap].copy()
        self.last_offset = None  # cumulative drag already consumed
        self.epoch = None  # page load the running total belongs to
        self.pending = np.zeros(3)  # drag not yet fed to the target
        self.ref = None  # phone attitude latched when ROTATE engaged
        self.smoothed = data.mocap_quat[arm_ik.mocap].copy()
        self.prev_hand = data.site_xpos[arm_ik.site].copy()
        self.speed = 0.0

    def apply(self, reading, dt):
        """Advance this arm's target from the latest phone frame."""
        if reading is None:
            self.last_offset = None
            self.ref = None
            return

        # --- translation ------------------------------------------------
        # The phone sends a running total, so re-reading an unchanged frame
        # contributes nothing; only genuine movement produces a delta.
        offset = np.array([reading["tx"], reading["ty"], reading["tz"]])
        # A reloaded page restarts its running total at zero, which would read
        # as one large drag back to the origin. Re-baseline on a new epoch
        # rather than lurching; the magnitude guard then only has to catch
        # corrupt frames, which no threshold could distinguish from a reload.
        if self.epoch != reading["epoch"] or self.last_offset is None:
            self.epoch = reading["epoch"]
            self.last_offset = offset
        delta = offset - self.last_offset
        self.last_offset = offset
        if np.linalg.norm(delta) <= PHONE_MAX_STEP:
            # Rotate into world immediately, so `pending` is always world-frame
            # and toggling frames mid-drag cannot mix conventions.
            if reading["frame"] == "hand":
                delta = hand_basis(self.d.mocap_quat[self.ik.mocap]) @ delta
            self.pending += delta

        step = self.pending * PHONE_POS_SMOOTH
        self.pending -= step
        moved = self.d.mocap_pos[self.ik.mocap] + step

        # Keep a runaway target inside reach instead of dragging the arm to
        # its limits and leaving the IK stuck against them.
        reach = moved - self.home
        if (norm := np.linalg.norm(reach)) > PHONE_MAX_REACH:
            moved = self.home + reach * (PHONE_MAX_REACH / norm)
            self.pending[:] = 0.0  # stop banking drag we will never apply
        self.d.mocap_pos[self.ik.mocap] = moved

        # --- rotation, only while ROTATE is held ------------------------
        if reading["rotate"]:
            now = orientation_quat(
                reading["pitch"], reading["roll"], reading["yaw"]
            )
            
            if self.ref is None:
                # Latch, so the hand never snaps to the phone's absolute
                # attitude and yaw drift resets on every press.
                self.ref = now
                self.hand = self.d.mocap_quat[self.ik.mocap].copy()
                self.smoothed = self.hand.copy()
            else:
                inverse, delta, target = np.zeros(4), np.zeros(4), np.zeros(4)
                mujoco.mju_negQuat(inverse, self.ref)
                mujoco.mju_mulQuat(delta, inverse, now)
                mujoco.mju_mulQuat(
                    target,
                    self.hand,
                    shrink_quat(delta, np.radians(PHONE_ROT_DEADBAND)),
                )
                self.smoothed = blend_quat(
                    self.smoothed, target, PHONE_ROT_SMOOTH
                )
                self.d.mocap_quat[self.ik.mocap] = self.smoothed

            # print the rotation of the sphere in the world frame, so the operator can see how the hand is turning
            print(f"{self.ik.arm} hand: pitch={reading['pitch']:.1f} roll={reading['roll']:.1f} yaw={reading['yaw']:.1f}")
        else:
            self.ref = None

        self.ik.closed = bool(reading["grip"])
        self._report(dt)

    def _report(self, dt):
        """Publish hand speed and reach back to the phone."""
        if self.controller is None:
            return
        hand = self.d.site_xpos[self.ik.site]
        instant = float(np.linalg.norm(hand - self.prev_hand) / dt)
        self.prev_hand = hand.copy()
        # Heavily smoothed: a per-tick difference at 1 kHz is mostly noise.
        self.speed += 0.002 * (instant - self.speed)
        self.controller.telemetry = {
            "speed": round(self.speed, 3),
            "reach": round(
                float(np.linalg.norm(self.d.mocap_pos[self.ik.mocap] - self.home)), 3
            ),
            "limit": PHONE_MAX_REACH,
        }


def draw_target_frames(viewer, arms):
    """Draw an arrow per target showing where that hand is pointing.

    Model geoms cannot be arrows -- mjGEOM_ARROW is visualisation-only -- so
    these go into the viewer's user scene each frame instead of into the MJCF.
    The long arrow is the approach axis, i.e. where the gripper actually points
    (local -z for this model's site). The short one marks the hand's "up", which
    is what makes roll readable rather than ambiguous.
    """
    scene = viewer.user_scn
    scene.ngeom = 0

    for arm_ik, colour in zip(arms, TARGET_COLORS):
        basis = hand_basis(arm_ik.d.mocap_quat[arm_ik.mocap])
        arrows = (
            (basis[:, 0], [0.010, 0.010, 0.16], 1.00),  # forward / approach
            (basis[:, 2], [0.006, 0.006, 0.07], 0.55),  # hand up, shows roll
        )
        for direction, size, shade in arrows:
            if scene.ngeom >= scene.maxgeom:
                return
            rgba = np.array([*(np.array(colour[:3]) * shade), 0.9])
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.array(size),
                arm_ik.d.mocap_pos[arm_ik.mocap].copy(),
                # mjv arrows point along the geom's local z.
                direction_frame(direction).flatten(),
                rgba.astype(np.float32),
            )
            scene.ngeom += 1


def serve_phones(host: str, port: int):
    """Start the phone web server on a background thread.

    Returns the shared OrientationState the sockets write into. The viewer
    stays on the main thread, which is where MuJoCo wants it.
    """
    import threading

    import uvicorn

    from openarm_mujoco.web.app import app, resolve_public_url, tailnet_url

    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_level="warning")
    )
    threading.Thread(target=server.run, daemon=True).start()

    url = tailnet_url() or f"http://{host}:{port}"
    print(f"\nphone control: open {url} and scan one QR per arm")
    print(f"  left  -> {url}/register?arm=left")
    print(f"  right -> {url}/register?arm=right\n")
    del resolve_public_url  # imported only to fail fast if the app is broken
    return app.state.orientation


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("xml", nargs="?", default=openarm_mujoco.openarm_cell_xml())
    ap.add_argument("--keyframe", "-k", default="home")
    ap.add_argument(
        "--phone",
        action="store_true",
        help="Serve the phone control pages and drive each hand from a phone.",
    )
    ap.add_argument("--phone-host", default="127.0.0.1", help="Web server bind address.")
    ap.add_argument("--phone-port", type=int, default=8000, help="Web server port.")
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

    orientation = serve_phones(args.phone_host, args.phone_port) if args.phone else None
    drivers = (
        {
            arm: PhoneDriver(ik, data, orientation.arms[arm])
            for arm, ik in zip(ARMS, arms)
        }
        if orientation
        else {}
    )

    print(__doc__)
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as v:
        v.cam.lookat[:] = model.stat.center
        v.cam.distance = model.stat.extent * 1.2
        # v.sync() costs ~5.7 ms against a 0.19 ms physics step, so syncing
        # once per step pins the loop at ~164 Hz and the sim crawls at 0.16x
        # realtime with several ms of frame jitter -- which looks like shake.
        # Render at FRAME_HZ and advance the physics in a burst between frames.
        steps_per_frame = max(1, round((1.0 / FRAME_HZ) / model.opt.timestep))
        frame_dt = steps_per_frame * model.opt.timestep
        while v.is_running():
            t0 = time.time()
            # launch_passive only RECORDS Ctrl+drag into v.perturb; unlike the
            # managed viewer it never applies it, so without this call the
            # target spheres select but never move.
            mujoco.mjv_applyPerturbPose(model, data, v.perturb, 0)
            if not paused[0]:
                for _ in range(steps_per_frame):
                    for arm, driver in drivers.items():
                        driver.apply(orientation.arms[arm].reading, model.opt.timestep)
                    for a in arms:
                        a.step(ghost)
                    mujoco.mj_step(model, data)
            draw_target_frames(v, arms)
            v.sync()
            time.sleep(max(0, frame_dt - (time.time() - t0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
