# Hierarchical Waypoint RL

This note explains how the high-level waypoint policy works in
`source/RL_UR5/RL_UR5/tasks/direct/rl_ur5/huber_obj_hierarchical_gray_depth.py`.

## Big Picture

The hierarchical setup replaces direct joint-action PPO with a two-level control structure:

```text
gray+depth image + robot state
        |
        v
high-level PPO policy
        |
        v
3D end-effector waypoint
        |
        v
damped differential IK executor
        |
        v
smooth UR5 joint targets
```

The high-level policy decides **where the end effector should go next**. The IK executor decides **how the joints should move to get there**.

This keeps the visual policy from having to learn perception, obstacle avoidance, inverse kinematics, and smooth motor control all at once.

## What the High-Level Policy Outputs

The high-level task is `UR5-Hierarchical-Depth-PPO`.

Its action space is 3D:

```python
action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))
```

So the PPO policy outputs:

```text
[a_x, a_y, a_z]
```

Each action component is in `[-1, 1]`. These are **not absolute coordinates**. They are learned Cartesian residuals used to bend a default waypoint.

## How a Waypoint Is Chosen

The environment builds the waypoint from three pieces:

```text
waypoint = current_end_effector_position
         + nominal_step_toward_final_target
         + learned_policy_residual
```

### 1. Start From the Current End-Effector Position

The current end-effector position is computed in the robot base frame:

```python
ee_pos_b, _ = self._get_ee_pose_b()
```

All high-level waypoints are expressed in this base-frame Cartesian space.

### 2. Add a Nominal Step Toward the Final Target

The environment computes the direction from the current end effector to the sampled final target:

```python
target_delta = self._target_poses[:, :3] - ee_pos_b
target_distance = torch.norm(target_delta, dim=-1, keepdim=True)
direction = target_delta / torch.clamp(target_distance, min=1e-6)
```

Then it creates a short default step toward that target:

```python
nominal_step = direction * torch.clamp(
    target_distance, max=self.cfg.waypoint_nominal_step
)
```

The default `waypoint_nominal_step` is:

```python
waypoint_nominal_step = 0.08
```

So before learning does anything clever, the waypoint is roughly an 8 cm step toward the final target.

### 3. Add the Learned Residual

The PPO action is scaled and added to the nominal step:

```python
waypoint_pos = ee_pos_b + nominal_step + actions * residual_scale
```

The residual scale is:

```python
waypoint_residual_scale = (0.08, 0.08, 0.06)
```

That means the policy can adjust the waypoint by approximately:

```text
x: +/- 8 cm
y: +/- 8 cm
z: +/- 6 cm
```

This is intentionally bounded. The policy does not freely command arbitrary workspace points. It learns to modify a reasonable default step.

## Workspace Clamping

After the waypoint is computed, it is clamped to a reachable workspace:

```python
waypoint_bounds = {
    "x": (0.35, 0.90),
    "y": (-0.60, 0.65),
    "z": (-0.25, 0.35),
}
```

This prevents the policy from asking the IK controller to track impossible or unsafe Cartesian points.

## Waypoint Holding

The selected waypoint is held for several RL steps:

```python
waypoint_hold_steps = 4
```

This reduces jitter because the high-level policy does not command a brand-new waypoint every single physics step.

The flow is:

```text
choose waypoint
hold it for 4 policy steps
track it with IK
choose the next waypoint
```

## How the Waypoint Is Executed

The high-level environment uses Isaac Lab's `DifferentialIKController`:

```python
DifferentialIKControllerCfg(
    command_type="pose",
    use_relative_mode=False,
    ik_method="dls",
    ik_params={"lambda_val": self.cfg.ik_lambda},
)
```

The default damping value is:

```python
ik_lambda = 0.05
```

The IK controller receives the selected waypoint pose:

```python
self._diff_ik.set_command(self._waypoint_pose_b)
joint_pos_des = self._diff_ik.compute(...)
```

Then the environment applies safety and smoothing:

```python
joint_pos_des = self._clamp_joint_targets(joint_pos_des)
smoothed_target = self._robot_dof_targets + self.cfg.ik_target_smoothing * (
    joint_pos_des - self._robot_dof_targets
)
self._robot_dof_targets = self._limit_joint_velocity(smoothed_target, joint_pos)
```

So the final command sent to the UR5 is a smoothed, velocity-limited joint target.

## What the Policy Observes

The high-level policy receives:

```text
image: gray+depth camera observation
state: proprioception and task state
```

The image shape is:

```text
120 x 160 x 2
```

The two channels are:

```text
channel 0: grayscale
channel 1: normalized depth
```

The state vector includes:

```text
normalized joint positions
joint velocities
final target pose
current end-effector position
current waypoint position
previous high-level action
```

The human arm pose is **not** included in the policy state. The high-level policy must infer the arm and its free-space constraints from the gray+depth image. This keeps the method end-to-end vision-based for obstacle awareness.

The policy network in
`source/RL_UR5/RL_UR5/tasks/direct/rl_ur5/agents/PPO_skrl_hierarchical_gray_depth.yaml`
uses:

```text
CNN encoder for the image
MLP encoder for the state
fusion MLP
3D Gaussian policy output
```

## How It Learns Good Waypoints

The PPO policy learns from reward, not from supervised waypoint labels.

It discovers useful waypoint residuals by trying actions and receiving reward based on the resulting motion.

The high-level reward encourages:

- reducing final target position error
- making progress toward the final target
- choosing waypoints that IK can track
- keeping the robot smooth
- avoiding the human arm only when it is actually close
- avoiding table collision
- reaching the target accurately and stably

The simulator still uses ground-truth arm geometry to compute reward and success terms during training. That information is not exposed as an observation to the policy, and it is not needed at deployment time.

The key reward pieces are:

```text
position tracking reward
progress reward
waypoint feasibility reward
orientation reward
joint velocity penalty
action-rate penalty
arm safety penalty
table collision penalty
success reward
```

The progress reward is especially important:

```python
progress = self._last_target_distance - position_error
```

If a waypoint helps the end effector get closer to the final target, the policy gets positive reward. If it moves away or wastes motion, the reward is worse.

The waypoint feasibility reward is also important:

```python
waypoint_error = torch.norm(ee_pos_b - self._waypoint_pose_b[:, :3], dim=-1)
waypoint_reward = self.cfg.reward_waypoint_feasibility_weight * (
    1.0 - torch.tanh(waypoint_error / 0.08)
)
```

This encourages the high-level policy to choose waypoints that the low-level IK executor can actually follow.

## Why This Helps Compared With Direct Joint PPO

In the older direct-control setup, the visual PPO policy had to learn:

```text
where to go
how to avoid the arm
how to solve inverse kinematics
how to produce smooth joint commands
how to stop accurately at the target
```

That is a lot for one policy.

The hierarchical setup makes the high-level policy learn:

```text
where should the end effector go next?
```

The IK executor handles:

```text
how should the joints move to reach that waypoint?
```

This should reduce jitter, improve target accuracy, and reduce the tendency to avoid the entire workspace just because the human arm is present.

## Training Commands

Train the high-level visual waypoint policy:

```bash
python scripts/skrl/train.py \
    --task=UR5-Hierarchical-Depth-PPO \
    --num_envs 32 \
    --enable_cameras \
    --headless
```

Quick smoke test:

```bash
python scripts/skrl/train.py \
    --task=UR5-Hierarchical-Depth-PPO \
    --num_envs 2 \
    --enable_cameras \
    --headless \
    --max_iterations 1
```

The separate low-level learning task is:

```bash
python scripts/skrl/train.py \
    --task=UR5-Waypoint-LowLevel-PPO \
    --num_envs 64 \
    --headless
```

The high-level task currently runs without requiring a trained low-level checkpoint because it uses DLS IK internally.
