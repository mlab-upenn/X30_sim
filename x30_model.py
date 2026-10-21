"""Build a simulation-ready MuJoCo model of the DEEP Robotics X30.

The MJCF shipped by DR (assets/X30_mjcf) is a welded visual model: no free joint,
no actuators, floor at z=-1 and cylinder collision geoms. This patches it in memory
(the source file is left untouched) so it can stand and walk.
"""

from pathlib import Path

import mujoco
import numpy as np

# Copied from rl_training/deep_robotics_model/X30/mjcf (unmodified).
X30_XML = Path(__file__).resolve().parent / "assets" / "X30_mjcf" / "X30.xml"

# SDK / policy joint order: FL, FR, HL, HR x HipX, HipY, Knee.
LEGS = ("FL", "FR", "HL", "HR")
JOINT_NAMES = [f"{leg}_{j}_joint" for leg in LEGS for j in ("HipX", "HipY", "Knee")]
FOOT_BODIES = [f"{leg}_FOOT" for leg in LEGS]

# sdk_config.toml caps on the robot: [hipx, hipy, knee] N*m.
TORQUE_LIMIT = np.array([84.0, 84.0, 160.0] * 4)

# The SDK demo's standing pose (0, -42, 78 deg); ~0.50 m body height, matching motion.toml.
STAND_POSE = np.deg2rad([0.0, -42.0, 78.0] * 4)
# The SDK demo's crouched pose (0, -70, 150 deg), used as the lying-down start.
CROUCH_POSE = np.deg2rad([0.0, -70.0, 150.0] * 4)


def build_model(dt: float = 0.001, torque_limit: np.ndarray = TORQUE_LIMIT) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(X30_XML))
    spec.option.timestep = dt
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080  # offscreen video

    floor = spec.geom("floor")
    floor.pos = [0, 0, 0]
    floor.friction = [1.0, 0.01, 0.01]

    torso = spec.body("TORSO")
    torso.pos = [0, 0, 0.6]
    torso.add_freejoint(name="root")
    torso.add_site(name="imu", pos=[0, 0, 0])

    # Cylinder collisions -> capsules (more stable contacts, MJX-compatible later).
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER and geom.contype != 0:
            geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE

    for name, limit in zip(JOINT_NAMES, torque_limit):
        joint = spec.joint(name)
        joint.armature = 0.01  # reflected rotor inertia; unknown for X30, a small stabilising value
        joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_FALSE
        act = spec.add_actuator(name=name.replace("_joint", "_motor"), target=name)
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [-limit, limit]

    spec.add_sensor(name="gyro", type=mujoco.mjtSensor.mjSENS_GYRO,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    spec.add_sensor(name="acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    return spec.compile()


class X30Sim:
    """Thin wrapper: SDK-ordered joint state, PD+feed-forward torque law, IMU readout."""

    def __init__(self, dt: float = 0.001, torque_limit: np.ndarray = TORQUE_LIMIT):
        self.model = build_model(dt, torque_limit)
        self.data = mujoco.MjData(self.model)
        self.torque_limit = np.asarray(torque_limit, dtype=np.float64)
        self.qadr = np.array([self.model.joint(n).qposadr[0] for n in JOINT_NAMES])
        self.vadr = np.array([self.model.joint(n).dofadr[0] for n in JOINT_NAMES])
        self.act_ids = np.array([self.model.actuator(n.replace("_joint", "_motor")).id for n in JOINT_NAMES])
        self.foot_ids = np.array([self.model.body(b).id for b in FOOT_BODIES])
        self.torso_id = self.model.body("TORSO").id
        self.tau = np.zeros(12)

    def reset(self, joint_pos: np.ndarray = STAND_POSE, base_height: float | None = None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = joint_pos
        self.data.qpos[3:7] = [1, 0, 0, 0]
        if base_height is None:
            # Drop the body so the lowest foot just touches the floor.
            self.data.qpos[2] = 1.0
            mujoco.mj_kinematics(self.model, self.data)
            foot_r = self.model.geom_size[self.model.body_geomadr[self.foot_ids[0]], 0]
            base_height = 1.0 - (self.data.xpos[self.foot_ids, 2].min() - foot_r) + 0.002
        self.data.qpos[2] = base_height
        mujoco.mj_forward(self.model, self.data)

    # --- state in SDK conventions -------------------------------------------------
    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    @property
    def dq(self) -> np.ndarray:
        return self.data.qvel[self.vadr].copy()

    @property
    def quat(self) -> np.ndarray:  # w, x, y, z (world <- body)
        return self.data.qpos[3:7].copy()

    @property
    def gyro(self) -> np.ndarray:  # body frame, rad/s
        return self.data.sensor("gyro").data.copy()

    @property
    def acc(self) -> np.ndarray:  # body frame specific force, m/s^2
        return self.data.sensor("acc").data.copy()

    def rpy(self) -> np.ndarray:
        w, x, y, z = self.quat
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([roll, pitch, yaw])

    def projected_gravity(self) -> np.ndarray:
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, self.quat)
        return rot.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])

    def base_lin_vel_body(self) -> np.ndarray:
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, self.quat)
        return rot.reshape(3, 3).T @ self.data.qvel[0:3]

    def foot_contact_forces(self) -> np.ndarray:
        """World-frame contact force on each foot, SDK order (12,)."""
        out = np.zeros((4, 3))
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b1 = self.model.geom_bodyid[con.geom1]
            b2 = self.model.geom_bodyid[con.geom2]
            for leg, fid in enumerate(self.foot_ids):
                if fid in (b1, b2):
                    mujoco.mj_contactForce(self.model, self.data, i, f6)
                    frame = con.frame.reshape(3, 3)
                    f = frame.T @ f6[:3]
                    out[leg] += f if fid == b2 else -f
        return out.reshape(-1)

    # --- actuation -----------------------------------------------------------------
    def apply_joint_cmd(self, pos, vel, kp, kd, tau_ff):
        """The X30 joint law: tau = kp*(pos - q) + kd*(vel - dq) + tau_ff, clipped to limits."""
        tau = kp * (pos - self.q) + kd * (vel - self.dq) + tau_ff
        self.tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        self.data.ctrl[self.act_ids] = self.tau

    def step(self, n: int = 1):
        mujoco.mj_step(self.model, self.data, nstep=n)
