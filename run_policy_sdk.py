"""Drive the X30 through DR's MotionSDK (pybind module) with an on-robot policy.

Same flow as DR's motion_sdk_example.py: acquire control, tuck legs, stand up, then hand over
to the policy (100 Hz, 1 kHz command stream). Talks to sim_bridge.py by default.

    .venv/bin/python run_policy_sdk.py                        # sim on 127.0.0.1
    .venv/bin/python run_policy_sdk.py --policy slope_policy

Keys: w/s vx +-0.1, a/d vy +-0.1, q/e yaw +-0.1, space = zero command, x = release control and exit.

This script never targets a real robot unless --real is passed; on hardware follow DR's safety
notes (robot hoisted, people >= 5 m away, low torque_limit in sdk_config.toml).
"""

import argparse
import ipaddress
import select
import sys
import termios
import tty
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import deeprobotics_x30_motion_sdk_py as dr  # noqa: E402  (built by sdk/CMakeLists.txt)

from x30_model import CROUCH_POSE, STAND_POSE  # noqa: E402
from x30_policy import X30Policy, config_for  # noqa: E402

TUCK_TIME = 1.0  # DR demo: PreStandUp
STAND_TIME = 1.5  # DR demo: StandUp
STAND_KP, STAND_KD = 300.0, 4.0
CMD_STEP = 0.1
CMD_MAX = np.array([2.0, 1.0, 1.5])


def gravity_from_rpy_deg(roll, pitch, yaw):
    r, p, _ = np.deg2rad([roll, pitch, yaw])
    # World gravity (0, 0, -1) expressed in the body frame for ZYX Euler angles.
    return np.array([np.sin(p), -np.sin(r) * np.cos(p), -np.cos(r) * np.cos(p)])


def cubic(q0, q1, t, total):
    s = np.clip(t / total, 0.0, 1.0)
    pos = q0 + (q1 - q0) * (3 * s**2 - 2 * s**3)
    vel = (q1 - q0) * (6 * s - 6 * s**2) / total if t < total else np.zeros_like(q0)
    return pos, vel


def fill_cmd(cmd, pos, vel, kp, kd, tor=0.0):
    pos, vel = np.broadcast_to(pos, 12), np.broadcast_to(vel, 12)
    for i in range(12):
        j = cmd.joint_cmd[i]
        j.pos, j.vel, j.tor, j.kp, j.kd = float(pos[i]), float(vel[i]), float(tor), float(kp), float(kd)


class Keyboard:
    def __enter__(self):
        self.tty = sys.stdin.isatty()
        if self.tty:
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if self.tty:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll(self):
        if self.tty and select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="127.0.0.1", help="motion host (sim bridge, or 192.168.1.103 with --real)")
    ap.add_argument("--port", type=int, default=43893)
    ap.add_argument("--recv-port", type=int, default=43897)
    ap.add_argument("--policy", default="silent_policy")
    ap.add_argument("--cmd", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("VX", "VY", "WZ"),
                    help="initial velocity command once the policy is running")
    ap.add_argument("--duration", type=float, default=None, help="exit after this many seconds of policy control")
    ap.add_argument("--schedule", default=None,
                    help='timed commands after stand-up, "secs:vx,vy,wz ..." e.g. "3:1,0,0 3:1,0,0.5 2:0,0,0"; '
                         "exits when it ends")
    ap.add_argument("--real", action="store_true", help="allow a non-loopback --ip (real robot)")
    args = ap.parse_args()
    if not ipaddress.ip_address(args.ip).is_loopback and not args.real:
        sys.exit(f"--ip {args.ip} is not the local sim; pass --real if you really mean a robot")

    schedule, end = [], 0.0
    for item in (args.schedule or "").split():
        secs, c = item.split(":")
        end += float(secs)
        schedule.append((end, np.array([float(v) for v in c.split(",")])))

    policy = X30Policy(config_for(args.policy))
    cfg = policy.cfg
    decim = int(round(cfg.policy_dt / 0.001))

    timer = dr.TimeTool()
    sender = dr.SendToRobot(args.ip, args.port)
    receiver = dr.ParseCommand(args.recv_port)
    receiver.start_work()
    timer.time_init(1)

    print(f"[runner] waiting for robot state on :{args.recv_port} ...")
    for _ in range(3000):
        timer.time_interrupt()
        if receiver.get_receive_state():
            break
    else:
        sys.exit("[runner] no data from the robot/sim; is sim_bridge.py running?")

    print("[runner] acquiring control (DR's robot_state_init waits 7 s) ...")
    sender.robot_state_init()

    cmd = dr.RobotCmdSDK()
    vel_cmd = np.zeros(3)
    policy_t0 = None
    phase, phase_t0, q_start, tick = "tuck", 0.0, None, 0
    target = None

    print("[runner] keys: w/s a/d q/e, space=stop, x=exit")
    with Keyboard() as kb:
        start = timer.get_start_time()
        while True:
            if timer.time_interrupt():
                continue
            now = timer.get_now_time(start)
            if not receiver.get_receive_state():
                print("[runner] lost robot data, exiting")
                break
            data = receiver.get_data()
            q = np.array([j.pos for j in data.joint_data])
            dq = np.array([j.vel for j in data.joint_data])
            if q_start is None:
                q_start, phase_t0 = q, now

            key = kb.poll()
            if key == "x":
                break
            if key:
                delta = {"w": (0, 1), "s": (0, -1), "a": (1, 1), "d": (1, -1), "q": (2, 1), "e": (2, -1)}.get(key)
                if delta:
                    vel_cmd[delta[0]] += delta[1] * CMD_STEP
                elif key == " ":
                    vel_cmd[:] = 0.0
                vel_cmd = np.clip(vel_cmd, -CMD_MAX, CMD_MAX)
                print(f"\r[runner] cmd vx={vel_cmd[0]:+.1f} vy={vel_cmd[1]:+.1f} wz={vel_cmd[2]:+.1f}   ", end="", flush=True)

            t = now - phase_t0
            if phase == "tuck":
                pos, vel = cubic(q_start, CROUCH_POSE, t, TUCK_TIME)
                fill_cmd(cmd, pos, vel, STAND_KP, STAND_KD)
                if t >= TUCK_TIME:
                    phase, phase_t0, q_start = "stand", now, q
            elif phase == "stand":
                pos, vel = cubic(q_start, STAND_POSE, t, STAND_TIME)
                fill_cmd(cmd, pos, vel, STAND_KP, STAND_KD)
                if t >= STAND_TIME + 0.5:
                    print(f"[runner] standing; {args.policy} in control, cmd={args.cmd}")
                    phase, policy_t0, policy_tick0 = "policy", now, tick
                    vel_cmd = np.clip(np.array(args.cmd), -CMD_MAX, CMD_MAX)
            else:
                if args.duration is not None and now - policy_t0 >= args.duration:
                    break
                if schedule:
                    elapsed, seg = now - policy_t0, None
                    for end, c in schedule:
                        if elapsed < end:
                            seg = c
                            break
                    if seg is None:
                        break
                    if not np.array_equal(seg, vel_cmd):
                        vel_cmd = seg
                        print(f"[runner] t={elapsed:4.1f}s cmd vx={seg[0]:+.1f} vy={seg[1]:+.1f} wz={seg[2]:+.1f}")
                if target is None or (tick - policy_tick0) % decim == 0:
                    imu = data.imu
                    gyro = np.array([imu.omega_x, imu.omega_y, imu.omega_z])
                    grav = gravity_from_rpy_deg(imu.roll, imu.pitch, imu.yaw)
                    target = policy.act(gyro, grav, vel_cmd, q, dq)
                fill_cmd(cmd, target, 0.0, cfg.kp, cfg.kd)
            sender.set_send(cmd)
            tick += 1

    print("\n[runner] releasing control (damping, 5 s) ...")
    sender.control_get(1)


if __name__ == "__main__":
    main()
