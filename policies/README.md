# Policies (not in git)

The `.pt` files are DEEP Robotics' on-robot policies, copied off the X30's motion host. They
have no license, so this public repo doesn't include them. To use the sim, copy them here:

```bash
scp 'ysc@192.168.1.103:<path to the .pt files on the robot>/*.pt' policies/
```

Expected files (16): backup_policy, follow_policy, low_energy_policy, policy_x30_H_v3.0/3.3/3.4/3.6,
poscmd_policy, recover_policy, run_policy_reflect, silent_policy, slope_policy, stand_recover_policy,
trot_policy_reflect, trot_slope_nav_policy, trot_velest. Only silent_policy and trot_policy_reflect
are calibrated (see ../README.md).
