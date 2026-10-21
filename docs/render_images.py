"""Render the README images: docs/x30_hero.png and docs/x30_strip.png.

Runs trot_policy_reflect at 1 m/s directly in MuJoCo (no SDK) and grabs frames with a
diagonal leg pair in the air.  Usage, from the repo root:  .venv/bin/python docs/render_images.py
"""
import os, sys
os.environ["MUJOCO_GL"] = "egl"; os.environ.pop("DISPLAY", None)
import numpy as np, mujoco, imageio.v2 as iio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from x30_model import X30Sim, STAND_POSE
from x30_policy import X30Policy, config_for
out = sys.argv[1] if len(sys.argv) > 1 else "docs/x30"
cfg = config_for("trot_policy_reflect")
sim = X30Sim(); sim.reset(STAND_POSE); pol = X30Policy(cfg)
dec = int(round(cfg.policy_dt / 0.001))
r = mujoco.Renderer(sim.model, 720, 1280)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING; cam.trackbodyid = sim.torso_id
cam.distance, cam.elevation, cam.azimuth = 2.4, -12.0, 120.0
opt = mujoco.MjvOption()
frames = []
for k in range(int(5.0 / cfg.policy_dt)):
    tgt = pol.act(sim.gyro, sim.projected_gravity(), np.array([1.0, 0, 0]), sim.q, sim.dq)
    for _ in range(dec):
        sim.apply_joint_cmd(tgt, 0, cfg.kp, cfg.kd, 0); sim.step()
    if k * cfg.policy_dt >= 4.0:
        fz = sim.foot_contact_forces().reshape(4, 3)[:, 2]
        r.update_scene(sim.data, camera=cam, scene_option=opt)
        frames.append((k, fz > 5, r.render()))
# candidates: frames with exactly one diagonal pair in the air
cands = [(k, c, img) for k, c, img in frames if list(c) in ([True, False, False, True], [False, True, True, False])]
print(len(frames), "frames,", len(cands), "with one diagonal pair airborne")
pick = cands[len(cands) // 2] if cands else frames[len(frames) // 2]
iio.imwrite(out + "_hero.png", pick[2])
step = max(1, len(frames) // 8)
strip = [f[2][120:660, 240:1040] for f in frames[::step][:4]]
iio.imwrite(out + "_strip.png", np.concatenate(strip, axis=1)[::2, ::2])
print("picked step", pick[0], "contacts", pick[1])
sys.stdout.flush()
os._exit(0)  # skip MuJoCo EGL teardown errors at interpreter exit
