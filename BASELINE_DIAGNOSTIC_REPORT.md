# Plain PPO Baseline Diagnostic Report

## Result

The first 25,600-step baseline did not learn the task:

- Evaluation mean return: `-686.37`.
- Evaluation success rate: `0%`.
- Completed-episode return windows did not improve consistently.
- PPO losses and gradients remained finite.
- The checkpoint loaded successfully.

This is a training failure, not a numerical crash.

The corrected repeat-matched evaluation of the paper-style checkpoint is
different from the earlier mismatched evaluation:

- Repeat `100`, 20 deterministic episodes: mean return `255.28`.
- Success rate: `0%`.
- Final axis error: `1.41`.
- Final orientation error: `0.44` radians.
- Minimum obstacle distance: approximately `0.16` m.
- Mean action change: `0.121`.

The higher return is a real improvement over repeat `1`, but the policy still
does not satisfy the combined pose-success condition. The evaluator now reads
the environment configuration from new checkpoints and supports
`--action-repeat` for older checkpoints.

## Environment audit

The MuJoCo model uses:

- MuJoCo timestep: `0.002` seconds.
- Six absolute joint-position actuator commands.
- Actuator control range: approximately `[-pi, pi]` radians.
- Current default action repeat: `1`.
- Paper-equivalent action repeat: `100` physics steps, or `0.2` seconds per
  agent action.

A 1,000-step benchmark measured approximately 4,265 environment calls per
second with action repeat `1`.

The environment now exposes `action_repeat` explicitly. The default remains
`1` to preserve the previous contract. With `action_repeat=100`, one call to
`env.step()` advances 100 MuJoCo physics steps, but `step_count` increases by
one. This matches the paper’s distinction between repeated simulation steps
and agent decisions.

The reward is returned once after the repeated action, rather than summed over
internal physics steps, matching the reference environment’s reward timing.

## Pose and reward ranges

For a fixed zero action over a seeded rollout, the observed ranges were
approximately:

| Quantity | Minimum | Mean | Maximum |
|---|---:|---:|---:|
| Reward, repeat 1 | -1.29 | -1.11 | -0.85 |
| Axis error, repeat 1 | 1.49 | 2.39 | 2.56 |
| Orientation error, repeat 1 | 1.58 | 2.20 | 3.14 |
| Minimum obstacle distance | 0.137 | 0.152 | 0.181 |

With repeat `100`, the same zero-action probe produced larger axis error on
average and similar reward scale. This indicates that the paper-compatible
repeat changes the control problem materially; it is not just a performance
setting.

## Changes made

### Explicit action repeat

`UR5eMujocoEnv` now accepts `action_repeat`. Internal physics advances use
that value, while episode length counts agent actions. This makes both the
previous MuJoCo behavior and the paper-style behavior testable.

### Persistent episode logging

`train_ppo.py` now carries a partial episode return across rollout boundaries
instead of resetting it at every PPO update. It records:

- Number of completed episodes.
- Mean return over the latest completed episodes.

This removes the misleading pattern where most updates reported `nan` simply
because a rollout ended in the middle of an episode.

### Controlled repeat experiment

A short five-update run with repeat `100` completed with finite PPO metrics.
Its returns were approximately:

```text
-36.3, -40.4, -39.1, -41.0, -45.2
```

This is too short to establish learning, but it confirms that the
paper-equivalent timing runs end-to-end.

## Interpretation

The baseline should not yet be expanded to AE or PF. The main open training
questions are:

1. Whether the PPO baseline should use repeat `1` for the MuJoCo reproduction
   or repeat `100` for released-code parity.
2. Whether the broad initial Gaussian action distribution is appropriate for
   absolute joint-position actions.
3. Whether observation normalization is needed for stable learning.
4. Whether the 25-D pose task and orientation tolerance are too difficult for
   the current one-environment smoke configuration.

## Next controlled experiment

Run two equal-budget baselines with identical seeds and PPO hyperparameters:

1. `action_repeat=1`.
2. `action_repeat=100`.

For each, save:

- Training metrics.
- Deterministic evaluation returns.
- Success rate.
- Axis and orientation errors.
- Action smoothness.

Then select the main reproduction contract based on the paper/released-code
parity requirement and the measured learning behavior. Only after a plain
baseline demonstrates learning should AE or PF be introduced.

The repeat comparison is now complete. The next baseline experiment should
keep repeat `100`, add observation normalization as a controlled change, and
measure whether final axis and orientation errors decrease rather than relying
on reward alone.

## Observation-normalization experiment

The repeat-100 run was repeated with a running mean/variance normalizer for
the 25-D observation. The statistics are stored in the checkpoint and reused
unchanged during evaluation.

Training used the same seed, 25,600 environment decisions, PPO settings, and
action repeat as the unnormalized baseline. The normalized checkpoint produced
these results over 20 deterministic episodes:

| Metric | Unnormalized | Normalized |
|---|---:|---:|
| Mean return | 255.28 | -119.89 |
| Success rate | 0% | 0% |
| Final axis error | 1.41 | 1.54 |
| Final orientation error (rad) | 0.44 | 1.54 |
| Minimum obstacle distance (m) | 0.16 | 0.15 |
| Mean action change | 0.121 | 0.113 |

Normalization is therefore not a useful change for this baseline at the
current training budget. It remains available through
`--normalize-observations` for future controlled experiments, but the main
track should use the unnormalized observation contract while the action
distribution and success criterion are investigated.

## Old-code action-standard-deviation experiment

The old PPO configuration starts its Gaussian action standard deviation at
`0.6`, while the current implementation previously started at `1.0`. The
current PPO now exposes this as `--action-std` and stores it in the checkpoint
configuration.

Using repeat `100`, seed `0`, 25,600 environment decisions, and otherwise
unchanged PPO settings, the `action_std=0.6` checkpoint produced:

| Metric | Unnormalized std=1.0 | std=0.6 |
|---|---:|---:|
| Mean return | 255.28 | 259.59 |
| Success rate | 0% | 0% |
| Final axis error | 1.41 | 1.37 |
| Final orientation error (rad) | 0.44 | 0.50 |
| Minimum obstacle distance (m) | 0.16 | 0.11 |
| Mean action change | 0.121 | 0.0046 |
| Mean action jerk proxy | not recorded | 0.0043 |

The lower exploration scale produces much smoother deterministic actions and a
small axis-error improvement, but it does not solve the combined pose
criterion. Keep `action_std=0.6` as the preferred deployment/smoothness
baseline. The next controlled change should be the old episode horizon and
update-batch scale: use 100 agent steps per episode and a 4096-step rollout
with 20 PPO epochs, while keeping the corrected MuJoCo success criterion.

## Old PPO training-scale experiment

The old rollout scale was tested with 100 agent steps per episode, 4,096
transitions per PPO update, 20 update epochs, repeat `100`, and
`action_std=0.6`. Six updates produced 24,576 environment decisions and 245
completed episodes.

Over 20 deterministic evaluation episodes:

| Metric | std=0.6, 500-step horizon | Old training scale |
|---|---:|---:|
| Mean return | 259.59 | -54.12 |
| Success rate | 0% | 0% |
| Final axis error | 1.37 | 1.94 |
| Final orientation error (rad) | 0.50 | 1.49 |
| Minimum obstacle distance (m) | 0.11 | 0.14 |
| Mean action change | 0.0046 | 0.0045 |

The old batch and horizon settings are not beneficial in the current MuJoCo
environment. The next experiment should isolate the task difficulty by
training and evaluating a position-only success criterion while retaining the
25-D observation and original reward for diagnosis. If position-only reaching
learns, orientation control—not PPO timing—is the limiting factor. Do not add
AE or PF until that diagnostic is complete.

## Position-only success diagnostic

The environment now supports `--position-only-success`. This changes only the
termination criterion: the 25-D observation, reward, action repeat, and PPO
settings remain unchanged. The position-only checkpoint was trained with
repeat `100`, action standard deviation `0.6`, seed `0`, and 25,600 decisions.

The result was identical to the corresponding full-pose run:

| Metric | Full-pose success | Position-only success |
|---|---:|---:|
| Mean return | 259.59 | 259.59 |
| Success rate | 0% | 0% |
| Final axis error | 1.37 | 1.37 |
| Final orientation error (rad) | 0.50 | 0.50 |
| Mean action change | 0.0046 | 0.0046 |

No episode reached the position tolerance (`0.08`), so relaxing orientation
did not change the learning trajectory. Orientation is therefore not the
primary blocker; positional reaching itself is failing. The next step should
be a short kinematic/control sanity experiment: verify that a known joint
configuration can reach sampled targets and that actions move the end
effector in the expected direction before changing PPO or adding AE/PF.

## Kinematic and actuator sanity check

The MuJoCo checks found:

- Each of joints 1–5 moves the tool when its absolute command is increased by
  `0.2` rad; measured tool displacements were 0.0177–0.1121 m.
- The end-effector position Jacobian has rank 3 at home, so local
  position control is possible.
- Bounded position-only IK solved all 20 sampled targets, with mean residual
  `2.6e-9` m and maximum residual `1.6e-8` m.

The check also found that the XML position actuators were clipped to ±π while
five corresponding joints were declared with ±2π ranges. The actuator ranges
were corrected to match the joint ranges in `ur5e.xml`.

The corrected model still shows a control-timescale problem for large
absolute commands. For one solved target, holding the IK command produced:

| Action repeat | Position distance after command |
|---:|---:|
| 100 | 0.366 m |
| 500 | 0.089 m |
| 1000 | 0.066 m |

Thus repeat `100` does not settle a large absolute joint jump in the current
MuJoCo dynamics. Existing checkpoints were trained with the pre-correction
actuator bounds and should not be used as final evidence. Before another long
PPO run, test smaller/incremental joint commands or retune the position
actuator gains while measuring command settling time.

## Incremental-action PPO experiment

The environment now provides an opt-in incremental action mode. The policy
outputs six values in `[-1, 1]`; each value changes the current absolute joint
command by at most `0.2` rad. The corrected actuator bounds and repeat `100`
were used, with action standard deviation `0.6`, seed `0`, and 25,600
environment decisions.

Over 20 deterministic evaluation episodes:

| Metric | Absolute-action std=0.6 | Incremental-action |
|---|---:|---:|
| Mean return | 259.59 | -238.25 |
| Success rate | 0% | 0% |
| Final axis error | 1.37 | 1.59 |
| Final orientation error (rad) | 0.50 | 1.99 |
| Minimum obstacle distance (m) | 0.11 | 0.12 |
| Mean action change | 0.0046 | 0.0028 |

Incremental commands improve command smoothness but did not produce useful
PPO learning at this budget. Keep the mode as a diagnostic option, not the
main training contract. The main baseline must now be retrained after the
actuator-range correction; comparisons with the older checkpoints are
provisional because those checkpoints used the clipped XML.

## Corrected-actuator absolute PPO retraining

The main absolute-action baseline was retrained after expanding the actuator
control ranges to match the XML joint ranges. It used repeat `100`,
action_std `0.6`, the 500-step horizon, seed `0`, and 25,600 decisions.

Evaluation over 20 deterministic episodes produced:

- Mean return: `52.45`
- Success rate: `0%`
- Final axis error: `1.54`
- Final orientation error: `0.97` radians
- Mean action change: `0.646`
- Mean action jerk proxy: `0.925`

This is not a successful baseline. It also exposed an action-contract issue:
the PPO action space is currently derived directly from actuator ranges, so
five action dimensions now span ±2π. The old code exposed actions in ±π even
though the robot joints could have wider physical ranges. The next controlled
change should preserve the corrected actuator limits but restore an explicit
policy command range of ±π, then retrain. This separates physical actuator
limits from the policy’s exploration/action contract and makes the comparison
with `FTP4RM` meaningful.

## Quaternion continuity correction and CUDA retraining

The observation path was corrected so the end-effector quaternion is
canonicalized in `_get_obs`, while pose-error calculations retain the raw
quaternion representation. A 5,000-transition verification across ten seeded
episodes found zero adjacent observation quaternion sign flips. Environment
validation also passed.

The corrected environment was retrained on the NVIDIA CUDA device using
repeat `200`, `gamma=0`, `gae_lambda=0`, action standard deviation `0.15`,
home-pose initialization, 25,600 decisions, and the existing 500-step
horizon. The run completed on CUDA, but evaluation over 10 deterministic
episodes still produced:

- Mean return: `681.11`
- Success rate: `0%`
- Final axis error: `0.946`
- Final orientation error: `0.263` radians
- Mean action change: `0.00124`
- Mean action jerk proxy: `0.000935`

The quaternion discontinuity was therefore a real observation defect and is
resolved, but it was not the sole cause of the remaining zero-success result.
The current baseline is not yet solved; curriculum learning, reward/threshold
analysis, and/or a richer observation contract remain necessary before
proceeding to AE/PF.

## Permanent curriculum sampler implementation

The temporary curriculum experiments have been converted into an optional
environment/training feature. `UR5eMujocoEnv` now supports a radius schedule
around the end-effector position computed from the configured home joint pose:

- `curriculum_start_radius`
- `curriculum_end_radius`
- `curriculum_steps`
- `full_workspace_probability`

During training, curriculum targets are sampled from an expanding spherical
neighborhood, while the full-workspace probability can preserve exposure to
the deployment distribution. The schedule and sampler settings are stored in
checkpoints and restored by the evaluator. Defaults preserve the original
random target behavior.

A CUDA smoke run using start/end radii `0.08/0.55 m`, 100 curriculum episodes,
20% full-workspace sampling, repeat `2`, and 128 decisions completed
successfully. Environment validation, Python compilation, checkpoint loading,
evaluation, and whitespace checks passed. This validates the implementation
contract; it is not evidence of full-workspace PPO success yet.

## Full CUDA curriculum training

The permanent curriculum was exercised with the strongest current PPO
configuration: repeat `200`, `gamma=0`, `gae_lambda=0`, action standard
deviation `0.15`, 100 updates, 25,600 decisions, a radius schedule from
`0.08` to `0.55 m` over 300 episodes, and 20% full-workspace sampling.

The run completed on the NVIDIA RTX 3060. Held-out evaluation over 20
deterministic full-workspace episodes produced:

- Mean return: `1839.36`
- Success rate: `0%`
- Final axis error: `0.285`
- Final orientation error: `0.126` radians
- Mean minimum obstacle distance: `0.193 m`
- Mean action change: `0.000618`

This is the best current pose-error result and a substantial improvement over
the previous CUDA baseline, but it still does not cross the strict success
thresholds (`0.08` axis error and `0.1` radians orientation error). The system
remains in the plain-PPO/curriculum stage; AE/PF should remain blocked until a
held-out success rate is demonstrated.

## Physics/control patch experiments

The XML inertia and gain/damping suggestions were tested in isolated model
copies before changing the shared model:

- Wrist-3 inertia correction destabilized the existing position controller and
  increased static sag; it was not adopted.
- Differentiated gains/damping were stable but slower and had larger final
  joint error; they were not adopted.
- A combined inertia/gain/gravity patch achieved stable simulation but produced
  `0%` held-out success and large pose errors; it was not adopted.

An optional torque-limited gravity/bias compensation mode was implemented in
the environment. It applies the current `qfrc_bias` through `qfrc_applied`,
clamped to `jnt_actfrcrange`, between `mj_step1` and `mj_step2`. This avoids
the unrestricted force bypass and removes the measured static home sag from
`10.3 mm` to effectively zero.

The implementation also supports normalized actions mapped to the XML
actuator ranges and optional bounded progress shaping. A normalized-action
run did not improve the best pose error, and small incremental actions were
worse.

## Gravity-compensated curriculum results

The strongest new candidate used gravity compensation, repeat `200`, action
standard deviation `0.15`, `gamma=0`, `gae_lambda=0`, the `0.08` to `0.55 m`
curriculum, and 20% full-workspace sampling. A 120-update CUDA run produced
`15%` success on an initial 20-episode evaluation, with axis error `0.206` and
orientation error `0.113` radians. On a separate 100-episode held-out set it
produced `4%` success, mean axis error `0.633`, and mean orientation error
`0.174` radians. The improvement is real but not yet reliable.

Increasing full-workspace sampling to 50% degraded the short evaluation to
`0%` success, so the 20% mixture remains preferred. The system has therefore
advanced from zero-success PPO to an unstable low-success candidate, but it
has not reached a reliable deployment checkpoint.

## PPO-preserving acceleration

The training path now includes `PPO.act_batch` and a vectorized trainer in
`train_ppo_vectorized.py`. It supports synchronous or asynchronous
multi-environment rollout collection, batches policy inference on the selected
device, and preserves the existing actor/critic PPO update and checkpoint
format. An eight-environment CUDA smoke run completed 6,144 environment
steps, and an asynchronous four-environment smoke run also completed.

This addresses the main GPU-utilization issue in the original trainer:
single-observation inference and CPU/GPU transfers occurred once per MuJoCo
step. The MuJoCo simulation remains CPU-bound, so GPU utilization may still
be bursty, but batched inference and asynchronous environment workers provide
the correct path for increasing throughput. The vectorized trainer should be
used for the next long candidate with 8--16 environments and total rollout
minibatches of at least 512 transitions.

The first long vectorized candidate used 8 asynchronous environments, 60
updates, 64 steps per environment per update (30,720 decisions total), repeat
`200`, gravity compensation, action standard deviation `0.15`, `gamma=0`,
`gae_lambda=0`, and the existing 20% full-workspace curriculum mixture. On
100 fixed held-out episodes it achieved:

- Success rate: `0%`
- Mean axis error: `0.501`
- Mean orientation error: `0.294` radians
- Mean return: `771.80`

This is worse than the best single-environment gravity-compensated candidate
(`4%` on its independent 100-episode set). Vectorization is therefore a
throughput improvement and a valid PPO architecture path, but it should not
be treated as a learning improvement without additional observation or
rollout-quality changes.
