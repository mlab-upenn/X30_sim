"""Run DEEP Robotics' on-robot X30 policies (TorchScript, policies/*.pt).

What the .pt files themselves reveal (see README.md):
  * input  = 6 stacked observation frames [current, then 5 history frames]; output = 12 actions.
  * frame layouts, joint order FL,FR,HL,HR x HipX,HipY,Knee (same as the MotionSDK):
      45: ang_vel(3) proj_gravity(3) cmd(3) dof_pos(12) dof_vel(12) last_action(12)
      52: ang_vel(3) proj_gravity(3) cmd(3) sin_phase cos_phase gait_cmd(5) dof_pos dof_vel last_action
          (reflect_proprioception in the export adds 0.5 to gait_cmd[1:4] when mirroring, i.e.
          walk-these-ways [freq, phase, offset, bound, duration])
What they do not reveal (obs scales, default pose, action scale, PD gains, policy rate,
history order) lives in PolicyConfig and was calibrated in sim (calibrate.py).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from x30_model import STAND_POSE

POLICY_DIR = Path(__file__).resolve().parent / "policies"
NUM_FRAMES = 6


@dataclass
class PolicyConfig:
    name: str = "silent_policy"
    default_pose: np.ndarray = field(default_factory=lambda: STAND_POSE.copy())
    # Defaults = best calibrate.py result for silent_policy (2026-10-21). Sharp optima:
    # action_scale, dof_vel_scale, policy_dt, cmd_scale, history order. Flat: kp/kd, default pose.
    action_scale: float = 0.25
    kp: float = 200.0
    kd: float = 4.0
    policy_dt: float = 0.01
    ang_vel_scale: float = 1.0
    dof_vel_scale: float = 0.05
    cmd_scale: tuple = (1.0, 1.0, 1.0)
    # Order of the 5 history frames after the current one. h0 = current, h1 = previous, ...
    #   "old_to_new": h5..h1   "old_to_new_incl": h4..h0   "new_to_old": h1..h5   "new_to_old_incl": h0..h4
    history_mode: str = "old_to_new"
    clip_action: float = 100.0
    # 52-dim (gait-clock) policies only. gait_cmd follows walk-these-ways:
    # [frequency Hz, phase, offset, bound, duration]; trot = phase 0.5, duration 0.5.
    gait_cmd: tuple = (2.0, 0.5, 0.0, 0.0, 0.5)
    clock_uses_gait_freq: bool = True  # clock advances at gait_cmd[0]; else at clock_freq
    clock_freq: float = 2.0


# Per-policy overrides of the PolicyConfig defaults, from calibrate.py-style sweeps in sim.
# trot_policy_reflect uses the classic legged_gym scales and 50 Hz, unlike silent_policy.
# Its gait_cmd inputs had no visible effect (pace/bound/pronk values still trot), so it is
# trot-only and gait_cmd just needs to be the trot default.
PRESETS = {
    "silent_policy": {},
    "trot_policy_reflect": dict(policy_dt=0.02, cmd_scale=(2.0, 2.0, 0.25), ang_vel_scale=0.25,
                                dof_vel_scale=0.1, gait_cmd=(2.0, 0.5, 0.0, 0.0, 0.5)),
}


def policy_path(name: str) -> Path:
    path = POLICY_DIR / f"{name}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; the policies aren't in git, see policies/README.md")
    return path


def frame_dim_of(name: str) -> int:
    net = torch.jit.load(str(policy_path(name)), map_location="cpu")
    return dict(net.named_parameters())["cenet_encoder.0.weight"].shape[1] // (NUM_FRAMES - 1)


def config_for(name: str, **overrides) -> PolicyConfig:
    preset = PRESETS.get(name)
    if preset is None:
        # Uncalibrated: borrow the calibrated policy with the same observation layout.
        donor = "trot_policy_reflect" if frame_dim_of(name) == 52 else "silent_policy"
        print(f"[policy] no calibrated preset for {name}; using {donor} settings")
        preset = PRESETS[donor]
    return PolicyConfig(name=name, **{**preset, **overrides})


class X30Policy:
    """Builds observations from SDK-style state and returns joint position targets."""

    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg
        self.net = torch.jit.load(str(policy_path(cfg.name)), map_location="cpu").eval()
        if not hasattr(self.net, "actor_body"):
            raise NotImplementedError(f"{cfg.name} has no actor (velocity estimator only); it can't drive the robot")
        in_dim = dict(self.net.named_parameters())["cenet_encoder.0.weight"].shape[1]
        self.frame_dim = in_dim // (NUM_FRAMES - 1)
        if self.frame_dim not in (45, 52):
            raise NotImplementedError(f"{cfg.name}: frame size {self.frame_dim}; only 45/52-dim policies are wired up")
        args = [a.name for a in self.net.forward.schema.arguments][1:]
        if "latent_lidar" in args:
            raise NotImplementedError(f"{cfg.name} needs a lidar latent from an encoder we don't have")
        # *_reflect policies take a mirror flag; 0 = no mirroring.
        self.extra_args = [torch.zeros(1)] if "reflect" in args else []
        self.reset()

    def reset(self):
        self.last_action = np.zeros(12)
        self.history = None  # (NUM_FRAMES, frame_dim), row 0 = newest
        self.gait_phase = 0.0

    def frame(self, gyro, proj_gravity, cmd, q, dq) -> np.ndarray:
        c = self.cfg
        clock = []
        if self.frame_dim == 52:
            freq = c.gait_cmd[0] if c.clock_uses_gait_freq else c.clock_freq
            self.gait_phase = (self.gait_phase + freq * c.policy_dt) % 1.0
            angle = 2 * np.pi * self.gait_phase
            clock = [np.sin(angle), np.cos(angle), *c.gait_cmd]
        return np.concatenate([
            gyro * c.ang_vel_scale,
            proj_gravity,
            np.asarray(cmd) * np.asarray(c.cmd_scale),
            clock,
            q - c.default_pose,
            dq * c.dof_vel_scale,
            self.last_action,
        ])

    @torch.no_grad()
    def act(self, gyro, proj_gravity, cmd, q, dq) -> np.ndarray:
        """One policy step. Returns joint position targets (SDK order, rad)."""
        f = self.frame(gyro, proj_gravity, cmd, q, dq)
        if self.history is None:
            self.history = np.tile(f, (NUM_FRAMES, 1))
        else:
            self.history = np.roll(self.history, 1, axis=0)
            self.history[0] = f
        h = self.history
        rows = {"old_to_new": [5, 4, 3, 2, 1], "old_to_new_incl": [4, 3, 2, 1, 0],
                "new_to_old": [1, 2, 3, 4, 5], "new_to_old_incl": [0, 1, 2, 3, 4]}[self.cfg.history_mode]
        stacked = np.concatenate([h[:1], h[rows]])
        obs = torch.as_tensor(stacked.reshape(1, -1), dtype=torch.float32)
        action = self.net(obs, *self.extra_args)[0].numpy().astype(np.float64)
        action = np.clip(action, -self.cfg.clip_action, self.cfg.clip_action)
        self.last_action = action
        return self.cfg.default_pose + self.cfg.action_scale * action
