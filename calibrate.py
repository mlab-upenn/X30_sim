"""Sweep the policy settings the .pt files don't record, scoring each combo in MuJoCo.

Each episode runs SCHEDULE (stand, forward slow/fast, sideways, turn both ways).
Score rewards staying upright, holding height and tracking each command.

    .venv/bin/python calibrate.py --policy silent_policy --workers 16
"""

import argparse
import itertools
import json
from dataclasses import asdict, replace
from multiprocessing import Pool

import numpy as np
import torch

from x30_model import STAND_POSE, X30Sim
from x30_policy import PolicyConfig, X30Policy, config_for

GRID = {
    "action_scale": [0.25],
    "gains": [(100.0, 2.0), (200.0, 4.0), (300.0, 6.0)],
    "policy_dt": [0.02, 0.01],
    "ang_vel_scale": [0.25, 1.0],
    "dof_vel_scale": [0.05],
    "cmd_scale": [(2.0, 2.0, 0.25), (1.0, 1.0, 1.0), (2.0, 2.0, 1.0), (1.0, 1.0, 0.25)],
    "history_mode": ["old_to_new", "old_to_new_incl", "new_to_old", "new_to_old_incl"],
}
# (duration s, [vx, vy, wz]); tracking is scored over the last 2 s of each moving segment.
SCHEDULE = [
    (2.0, [0.0, 0.0, 0.0]),
    (3.0, [0.5, 0.0, 0.0]),
    (3.0, [1.0, 0.0, 0.0]),
    (3.0, [0.0, 0.3, 0.0]),
    (3.0, [0.0, 0.0, 0.6]),
    (3.0, [0.0, 0.0, -0.6]),
]


def run_episode(cfg: PolicyConfig, schedule=SCHEDULE):
    torch.set_num_threads(1)
    sim = X30Sim()
    sim.reset(STAND_POSE)
    pol = X30Policy(cfg)
    decim = int(round(cfg.policy_dt / sim.model.opt.timestep))
    total = sum(d for d, _ in schedule)
    errs, heights, t, fell = [], [], 0.0, False
    for dur, cmd in schedule:
        cmd = np.array(cmd)
        meas = []
        for k in range(int(round(dur / cfg.policy_dt))):
            target = pol.act(sim.gyro, sim.projected_gravity(), cmd, sim.q, sim.dq)
            for _ in range(decim):
                sim.apply_joint_cmd(target, 0.0, cfg.kp, cfg.kd, 0.0)
                sim.step()
            t += cfg.policy_dt
            heights.append(sim.data.qpos[2])
            if k * cfg.policy_dt >= dur - 2.0:
                v = sim.base_lin_vel_body()
                meas.append([v[0], v[1], sim.gyro[2]])
            if sim.projected_gravity()[2] > -0.7 or sim.data.qpos[2] < 0.25:
                fell = True
                break
        if fell:
            break
        errs.append(np.abs(np.mean(meas, axis=0) - cmd).sum())
    alive = t / total
    track = float(np.mean(errs)) if errs else 2.0
    h = float(np.mean(heights))
    score = 3.0 * alive - 2.0 * track - 2.0 * abs(h - 0.48)
    return {"score": score, "alive": alive, "track_err": track, "height": h,
            "seg_err": [round(float(e), 2) for e in errs]}


def make_cfg(policy, combo):
    keys = list(GRID)
    d = dict(zip(keys, combo))
    kp, kd = d.pop("gains")
    return replace(config_for(policy), kp=kp, kd=kd, **d)


def _job(args):
    policy, combo = args
    cfg = make_cfg(policy, combo)
    return combo, run_episode(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="silent_policy")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    combos = list(itertools.product(*GRID.values()))
    print(f"{len(combos)} combos for {args.policy}")
    with Pool(args.workers) as pool:
        results = pool.map(_job, [(args.policy, c) for c in combos], chunksize=1)
    results.sort(key=lambda r: -r[1]["score"])
    keys = list(GRID)
    for combo, r in results[: args.top]:
        print(f"{r['score']:6.2f} alive={r['alive']:.2f} err={r['track_err']:.2f} h={r['height']:.2f} "
              f"seg={r['seg_err']} | "
              + " ".join(f"{k}={v}" for k, v in zip(keys, combo)))
    if args.out:
        best = make_cfg(args.policy, results[0][0])
        d = asdict(best)
        d["default_pose"] = list(map(float, best.default_pose))
        with open(args.out, "w") as f:
            json.dump({"config": d, "result": results[0][1]}, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
