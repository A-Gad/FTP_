# Why PPO Fails on the UR5e Reaching Task: Diagnosis and Fixes

A review of `ppo.py`, `train_ppo.py`, `evaluate_policy.py` and `ur5e_mujoco_env.py`. The code runs, but several design choices make the problem much harder for PPO than it needs to be, and one real bug invalidates the incremental-action experiments. Issues are listed in rough order of impact.

## 1. With absolute joint targets and repeat=200, the task is a one-step bandit

Each decision sets absolute joint targets and holds them for 200 sim steps. The arm mostly settles, so the next end-effector pose depends almost entirely on the action and barely on the previous state. The task is therefore "map (target, obstacles) to a joint configuration", which is learned inverse kinematics.

Despite that, PPO uses `gamma=0.99` and `gae_lambda=0.95`. Advantages then include rewards from future steps that the current action did not cause, which is pure noise.

**Fix:** try `gamma≈0` (or ≤0.5) and `gae_lambda=0`, so the advantage is just `r - V(s)`. This one change should help a lot.

## 2. The value loss is probably drowning the actor's gradient

Per-step reward is roughly 0 to 9, so discounted returns are in the hundreds. That makes `value_loss` huge. `clip_grad_norm_(self.policy.parameters(), 0.5)` clips one global norm across actor and critic, so the large value gradient shrinks the actor's gradient to almost nothing.

**Fix:** normalize rewards or returns, clip actor and critic separately, or lower gamma as in issue 1.

## 3. Incremental mode was never tested fairly (a real bug)

`train_ppo.py` always passes `initial_action=env.initial_qpos`. In incremental mode the action bounds are ±1, so values like -π/2 get clamped to ±(1-1e-6) and `atanh` gives about ±7.25. The actor starts saturated at ±1 on four joints, where the tanh gradient is about zero. The policy is stuck from step 0, so "incremental didn't help" is not a valid conclusion.

**Fix:** pass `initial_action` only when `incremental_actions` is false (or use zeros).

## 4. Termination on success conflicts with the reward

The reward `-log(pe² + 1e-4)` is positive and large near the goal (about 6 per step at the threshold). Success terminates the episode, so reaching the goal cuts off all future reward. The optimal policy is to hover just outside the threshold. You are not at that stage yet, but you will hit it as soon as the policy gets close.

**Fix:** add a success bonus larger than what hovering would earn, stop terminating on success, or use a bounded reward that is negative everywhere.

## 5. Truncation bootstrapping is wrong

In `RolloutBuffer.finish`, a time-limit truncation bootstraps from `values[index+1]`, which is the value of the first state of the next episode. It should use the value of the final observation of the truncated episode.

**Fix:** store that final-observation value when truncating.

## 6. The exploration noise is enormous

`action_std=0.6` is in pre-tanh space, and the output is scaled by π. That is roughly 1 to 2 rad of random noise per joint, per decision. With a precision target of about 2.7 cm per axis point and 0.1 rad orientation, that is far too noisy. The deterministic eval policy also jumps 0.72 rad between consecutive steps, so it is jittery as well.

**Fix:** try an initial std of about 0.1–0.2.

## 7. Far too few samples

76,800 decisions is about 300 updates of 16 gradient steps each, around 4,800 gradient steps in total. Learning IK to centimeter precision from a half-space of targets typically needs millions of samples. The final axis error of 1.09 (the sum of three distances, so about 36 cm each) says the policy is essentially outputting an average pose. It is undertrained, not subtly mis-tuned.

## 8. Observation and task issues

- **Obstacle positions aren't observed.** The policy only sees distances from the current link segments to each obstacle. For absolute-target actions it can't know whether its next configuration collides. Add obstacle positions.
- **EE quaternion sign flips.** `mju_mat2Quat` can flip the sign of the quaternion between steps, so the observation jumps discontinuously. Use the rotation matrix or a 6D representation.
- **Reachability with the fixed orientation is unproven.** Targets span a large half-space (radius up to 0.75, z from 0.05) but always demand the same orientation `(0, π/2, π/2)`. Many of those poses may be infeasible. "Targets are reachable" needs to be checked against the full sampling distribution, with both position and orientation. A quick Jacobian-based IK sweep over 1,000 sampled targets would give the feasible fraction.
- **Success is strict.** Combining a summed axis error ≤ 0.08 with orientation ≤ 0.1 rad is a hard threshold. Use a curriculum: position-only first, then tighten.

## Recommended next steps

1. Set gamma≈0, lambda=0, and normalize returns.
2. Lower the initial std to about 0.15 and fix the truncation bootstrap.
3. Run an IK reachability check on the sampling distribution.
4. Train position-only on a small target region for about 1M steps, then widen the region and add orientation.
5. Re-test incremental mode with the init bug fixed. It restores real sequential structure, and then normal gamma makes sense.

To check how bandit-like the problem really is, the `ur5e.xml` actuator settings (kp, timestep) are needed to confirm that 200 substeps lets the arm settle.
