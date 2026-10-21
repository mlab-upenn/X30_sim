# X30 sim: MotionSDK + on-robot policies in MuJoCo

![X30 trotting in MuJoCo under DR's trot_policy_reflect](docs/x30_hero.png)

<p align="center"><img src="docs/x30_strip.png" alt="Four frames through one trot stride at 1 m/s"></p>
<p align="center"><sub>DEEP Robotics' on-robot <code>trot_policy_reflect</code> trotting at 1 m/s in MuJoCo, one stride.</sub></p>

Test the DEEP Robotics X30 MotionSDK and the policies copied off the robot against a simulated X30,
before going near hardware. Everything needed is in this directory:

```
x30_sim/
├── third_party/x30_motion_sdk/   DR MotionSDK V2.0, unmodified, MIT (pybind11 submodule not vendored)
├── policies/*.pt                 DR's TorchScript policies, copied off the robot (not in git; see policies/README.md)
├── assets/X30_mjcf/              DR's X30 MJCF + meshes, unmodified, BSD-3 (from DeepRoboticsLab/deep_robotics_model)
├── x30_model.py  x30_policy.py  calibrate.py  sim_bridge.py  run_policy_sdk.py  run_sim.sh
├── sdk/CMakeLists.txt            out-of-tree SDK build → sdk/build/, lib/ (gitignored)
├── .venv/                        uv venv (gitignored)
└── videos/                       recordings (gitignored)
```

```
run_policy_sdk.py ──(SDK pybind, UDP)──► sim_bridge.py ──► MuJoCo X30
  policy @100 Hz        0x0111 joint cmds :43893        tau = kp(q*-q) + kd(dq*-dq) + tau_ff
  cmd stream @1 kHz  ◄── 0x0906 RobotDataSDK :43897 ──  1 kHz, real time
```

`sim_bridge.py` speaks the SDK's wire protocol, so any SDK program works against it by using
`127.0.0.1` instead of `192.168.1.103`.

| File | What it is |
| --- | --- |
| `x30_model.py` | Loads `assets/X30_mjcf/X30.xml` and patches it in memory: free joint, motors, IMU, floor at z=0, capsule collisions |
| `sim_bridge.py` | Fake motion host: 0x0114/0x0113 control handover, 0x0111 joint commands, 1 s no-command → damping, state stream |
| `x30_policy.py` | Observation builder + TorchScript runner for DR's policies |
| `calibrate.py` | Offline sweep (no UDP) for the settings the `.pt` files don't record |
| `run_policy_sdk.py` | MotionSDK client: take control → tuck → stand (DR demo poses) → policy, with keyboard commands |
| `sdk/CMakeLists.txt` | Out-of-tree build of the SDK lib and pybind module (→ `lib/`); DR's C++ example only with `-DBUILD_DR_EXAMPLE=ON` |

## Setup

```bash
cd x30_sim
uv venv -p 3.11 .venv
uv pip install -p .venv/bin/python mujoco==3.15.0 numpy==1.26.4 glfw pybind11 imageio imageio-ffmpeg
uv pip install -p .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
cmake -S sdk -B sdk/build -Dpybind11_DIR=$(.venv/bin/python -m pybind11 --cmakedir) \
      -DPython_EXECUTABLE=$PWD/.venv/bin/python
cmake --build sdk/build -j
./sdk/build/print_sizes   # EthCommand=12 RobotCmdSDK=240 RobotDataSDK=284
```

## Run

```bash
./run_sim.sh                          # tmux: bridge + viewer | runner (keys: w/s a/d q/e, space, x)
POLICY=trot_policy_reflect ./run_sim.sh
# or by hand, two terminals:
.venv/bin/python sim_bridge.py [--headless]
.venv/bin/python run_policy_sdk.py --cmd 1.0 0 0 --duration 10
```

The runner refuses a non-loopback `--ip` unless `--real` is given.

**Don't run DR's own examples against the sim.** `example/main.cpp` uses `SendToRobot()`, which
defaults to `192.168.1.103`, and `python/motion_sdk_example.py` hard-codes the same address. Both
talk to a real X30 if one is on your network. That's why `sdk/CMakeLists.txt` only builds the C++
example with `-DBUILD_DR_EXAMPLE=ON`.

Record a video (offscreen GPU rendering, no display needed) with a scripted command schedule:

```bash
.venv/bin/python sim_bridge.py --headless --record videos/demo.mp4 --duration 37 &
.venv/bin/python run_policy_sdk.py --schedule "2:0,0,0 4:1,0,0 4:1,0,0.5 3:1.5,0,0 3:0.7,0,0 4:0,0,0.6 2:0,0,0"
```

## Supported policies

Two of DR's on-robot policies are calibrated and run in the sim: `silent_policy` and
`trot_policy_reflect`. Both are TorchScript exports from DR's rsl_rl fork (`actor_critic_decoder`,
a DreamWaQ-style CENet estimator). Each takes 6 stacked frames (the current frame, then 5 history
frames) and outputs 12 joint actions. Joint order is FL, FR, HL, HR × HipX, HipY, Knee, the same as
the SDK.

| Policy | Frame | Layout |
| --- | --- | --- |
| `silent_policy` | 45 | ang_vel 3, proj_gravity 3, cmd 3, dof_pos−default 12, dof_vel 12, last_action 12 |
| `trot_policy_reflect` | 52 | ang_vel, gravity, cmd, sin_phase, cos_phase, gait_cmd 5, dof_pos, dof_vel, last_action (names from the export's `reflect_proprioception`) |

`trot_policy_reflect` also takes a `reflect` flag that mirrors left and right; the sim passes 0.

### Calibrated settings (silent_policy, `calibrate.py`)

| Setting | Value | Confidence |
| --- | --- | --- |
| action → target | `default + 0.25·a` | sharp optimum |
| default pose | SDK stand (0, −42°, 78°) | flat; nearby poses score the same |
| dof_vel scale | 0.05 | sharp |
| ang_vel scale / cmd scale | 1.0 / (1, 1, 1) | clear winner |
| policy rate | 100 Hz | clear winner over 50 Hz |
| history order | oldest → newest after the current frame | sharp vs newest-first; can't tell whether it includes the current frame |
| kp / kd | 200 / 4 | flat between 150 and 300 |

Measured in sim, starting from standing:
- **Forward:** tracks 0.6–2.0 m/s.
- **0.2 and 0.5 m/s forward:** it stays standing. 0.4 m/s walks, so this probably isn't a simple deadband.
- **Turning:** tracks ±0.6 rad/s.
- **Backward:** only weakly (−1.0 → −0.4 m/s).
- **Sideways:** about half the commanded speed.

That could be how the silent gait behaves or a gap in the sim dynamics: rotor inertia, motor model
and friction are unknown. Check against robot logs before trusting it.

### trot_policy_reflect (52-dim, preset in `x30_policy.PRESETS`)

This is a separate training run with different settings from `silent_policy`. It uses the classic
legged_gym scales:

| Setting | Value |
| --- | --- |
| cmd scale | × (2, 2, 0.25) |
| ang_vel scale | × 0.25 |
| dof_vel scale | × 0.1 |
| policy rate | 50 Hz |
| history order | oldest → newest |
| gait clock | `sin/cos(2π·φ)`, φ advancing at 2 Hz |
| gait_cmd | (2, 0.5, 0, 0, 0.5) |
| `reflect` input | 0 |

Diagonal feet land together about 95% of the time. Pace, bound or pronk gait_cmd values still
produce a trot, so the policy is trot-only.

Measured in sim:
- **Forward:** 0.5 → 0.55, 1.0 → 1.0 and 1.5 → 1.4 m/s.
- **Backward:** −0.5 → −0.35 m/s.
- **Sideways:** 0.4 → 0.27 m/s.
- **Turning:** 0.8 → 0.62 rad/s.
- **Standing:** creeps forward at about 0.08 m/s.
- **Spin after a run:** survives turning in place right after a 1.5 m/s run, which flips `silent_policy`.

`POLICY=trot_policy_reflect ./run_sim.sh`

## Not done yet

- **SDK on current firmware:** the API docs for current firmware don't mention this SDK, so it
  hasn't been checked on the robot.
