"""A simulated X30 motion host: MuJoCo X30 that speaks the MotionSDK UDP protocol.

Any MotionSDK program (C++ or the pybind module) works against it unchanged by pointing it at
127.0.0.1 instead of 192.168.1.103:

    recv  :43893  0x0114 take control | 0x0113 release | 0x0111 + RobotCmdSDK (12 x pos,vel,tor,kp,kd)
    send  -> :43897 every 1 ms: 0x0906 + RobotDataSDK (tick, IMU rpy[deg]/gyro/acc, joints, foot forces)

Like the robot, joints run tau = kp*(pos-q) + kd*(vel-dq) + tor (clipped to the sdk_config.toml
torque limits) while the SDK holds control, and fall back to damping when control is released
or no command arrives for > 1 s. Before an SDK takes control the robot lies in damping.

    .venv/bin/python sim_bridge.py                # viewer, real time
    .venv/bin/python sim_bridge.py --headless
    .venv/bin/python sim_bridge.py --headless --record x30.mp4 --duration 40
"""

import argparse
import os
import socket
import struct
import sys
import time

if "--headless" in sys.argv and "--record" in sys.argv:
    # Offscreen GPU rendering; a stale DISPLAY makes MuJoCo's EGL setup fail.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.pop("DISPLAY", None)

import mujoco
import numpy as np

from x30_model import CROUCH_POSE, TORQUE_LIMIT, X30Sim

CMD_JOINT = 0x0111
CMD_RELEASE = 0x0113
CMD_TAKE = 0x0114
STATE_CODE = 0x0906
MESS_VALUES = 1

HEADER = struct.Struct("<III")  # code, value|size, type:8 | count:24 (little-endian bitfield)
JOINT_CMD = struct.Struct("<60f")  # RobotCmdSDK, 240 B
ROBOT_DATA = struct.Struct("<Ii9f48f12f")  # RobotDataSDK, 284 B
assert JOINT_CMD.size == 240 and ROBOT_DATA.size == 284

CMD_TIMEOUT = 1.0
DAMPING_KD = 12.5  # what SendToRobot::control_get(UNABLE) sends before releasing


class SimMotionHost:
    def __init__(self, args):
        self.sim = X30Sim(dt=0.001, torque_limit=np.array(args.torque_limit * 4))
        self.sim.reset(CROUCH_POSE)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.rx.bind((args.bind, args.port))
        except OSError as e:
            raise SystemExit(f"[bridge] can't bind {args.bind}:{args.port} ({e.strerror}); is another bridge running?")
        self.rx.setblocking(False)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.targets = [(t.rsplit(":", 1)[0], int(t.rsplit(":", 1)[1])) for t in args.state_target]
        self.sdk_control = False
        self.cmd = None  # (12, 5) pos, vel, tor, kp, kd
        self.last_cmd_time = -np.inf
        self.tick = 0
        self.count = 0
        print(f"[bridge] listening on {args.bind}:{args.port}, state -> {self.targets}")

    # --- protocol ------------------------------------------------------------------
    def poll_commands(self, now):
        while True:
            try:
                msg, _ = self.rx.recvfrom(4096)
            except BlockingIOError:
                return
            if len(msg) < HEADER.size:
                continue
            code, value, word = HEADER.unpack_from(msg)
            if code == CMD_TAKE:
                if not self.sdk_control:
                    print("[bridge] 0x0114: SDK took control")
                self.sdk_control = True
                self.last_cmd_time = now
            elif code == CMD_RELEASE:
                print("[bridge] 0x0113: SDK released control -> damping")
                self.sdk_control, self.cmd = False, None
            elif code == CMD_JOINT and (word & 0xFF) == MESS_VALUES and value == JOINT_CMD.size:
                cmd = np.array(JOINT_CMD.unpack_from(msg, HEADER.size), dtype=np.float64).reshape(12, 5)
                if not np.all(np.isfinite(cmd)):
                    print("[bridge] non-finite joint command dropped")
                    continue
                self.cmd = cmd
                self.last_cmd_time = now

    def send_state(self):
        s = self.sim
        imu = np.concatenate([np.rad2deg(s.rpy()), s.gyro, s.acc])
        joints = np.stack([s.q, s.dq, s.tau, np.full(12, 30.0)], axis=1).reshape(-1)
        payload = ROBOT_DATA.pack(self.tick, int(self.tick), *imu, *joints, *s.foot_contact_forces())
        header = HEADER.pack(STATE_CODE, ROBOT_DATA.size, MESS_VALUES | ((self.count & 0xFFFFFF) << 8))
        self.count += 1
        for t in self.targets:
            try:
                self.tx.sendto(header + payload, t)
            except OSError:
                pass

    # --- control -------------------------------------------------------------------
    def actuate(self, now):
        active = self.sdk_control and self.cmd is not None and now - self.last_cmd_time <= CMD_TIMEOUT
        if self.sdk_control and self.cmd is not None and not active:
            print(f"[bridge] no command for > {CMD_TIMEOUT:.0f} s -> damping (re-send 0x0114 to resume)")
            self.sdk_control, self.cmd = False, None
        if active:
            c = self.cmd
            self.sim.apply_joint_cmd(c[:, 0], c[:, 1], c[:, 3], c[:, 4], c[:, 2])
        else:
            self.sim.apply_joint_cmd(0.0, 0.0, 0.0, DAMPING_KD, 0.0)

    def step(self, now):
        self.poll_commands(now)
        self.actuate(now)
        self.sim.step()
        self.tick += 1
        self.send_state()
        if self.tick % 2000 == 0:
            x, y, z = self.sim.data.qpos[:3]
            v = self.sim.base_lin_vel_body()
            mode = "sdk" if self.sdk_control and self.cmd is not None else "damping"
            print(f"[bridge] t={self.tick / 1000:6.1f}s {mode:7s} base=({x:+.2f},{y:+.2f},{z:.2f}) "
                  f"v_body=({v[0]:+.2f},{v[1]:+.2f}) yaw={np.rad2deg(self.sim.rpy()[2]):+.0f}deg")


class Recorder:
    """Offscreen MP4 capture from a camera that follows the robot."""

    def __init__(self, sim, path, fps, size=(720, 1280)):
        import imageio.v2 as imageio

        self.sim, self.path, self.period = sim, path, 1.0 / fps
        self.renderer = mujoco.Renderer(sim.model, *size)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = sim.torso_id
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 3.0, -18.0, 135.0
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=8)
        self.next_t = 0.0
        self.frames = 0
        print(f"[bridge] recording {size[1]}x{size[0]} @ {fps} fps -> {path}")

    def maybe_capture(self, sim_time):
        if sim_time < self.next_t:
            return
        self.next_t += self.period
        self.renderer.update_scene(self.sim.data, camera=self.cam)
        self.writer.append_data(self.renderer.render())
        self.frames += 1

    def close(self):
        self.writer.close()
        print(f"[bridge] wrote {self.frames} frames ({self.frames * self.period:.1f} s) to {self.path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=43893, help="motion-host command port (robot: 43893)")
    ap.add_argument("--state-target", action="append", default=None,
                    help="ip:port to stream RobotDataSDK to (repeatable; default 127.0.0.1:43897)")
    ap.add_argument("--torque-limit", type=float, nargs=3, default=list(TORQUE_LIMIT[:3]),
                    metavar=("HIPX", "HIPY", "KNEE"), help="sdk_config.toml torque_limit")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record", metavar="MP4", help="write a video from a camera tracking the robot")
    ap.add_argument("--record-fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=None, help="stop after this many sim seconds")
    args = ap.parse_args()
    args.state_target = args.state_target or ["127.0.0.1:43897"]
    sys.stdout.reconfigure(line_buffering=True)

    host = SimMotionHost(args)
    dt = host.sim.model.opt.timestep
    viewer = None
    if not args.headless:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(host.sim.model, host.sim.data)
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 2.5, -20, 135
        viewer.cam.trackbodyid = host.sim.torso_id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    recorder = Recorder(host.sim, args.record, args.record_fps) if args.record else None

    t0 = time.perf_counter()
    next_sync = 0.0
    lag_warned = False
    try:
        while viewer is None or viewer.is_running():
            sim_time = host.tick * dt
            if args.duration is not None and sim_time >= args.duration:
                break
            host.step(t0 + sim_time)
            if recorder is not None:
                recorder.maybe_capture(sim_time)
            if viewer is not None and sim_time >= next_sync:
                viewer.sync()
                next_sync = sim_time + 1 / 60
            ahead = t0 + host.tick * dt - time.perf_counter()
            if ahead > 0.0005:
                time.sleep(ahead - 0.0003)
            elif ahead < -0.1:
                if not lag_warned:
                    print("[bridge] running slower than real time; re-anchoring clock")
                    lag_warned = True
                t0 = time.perf_counter() - host.tick * dt
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
        if recorder is not None:
            recorder.close()


if __name__ == "__main__":
    main()
