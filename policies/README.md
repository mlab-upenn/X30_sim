# Policies (not in git)

The `.pt` files are DEEP Robotics' on-robot policies, copied off the X30's motion host. They
have no license, so this public repo doesn't include them. To use the sim, copy them here:

```bash
scp 'ysc@192.168.1.103:<path to the .pt files on the robot>/*.pt' policies/
```

The sim uses `silent_policy.pt` and `trot_policy_reflect.pt` (see ../README.md).
