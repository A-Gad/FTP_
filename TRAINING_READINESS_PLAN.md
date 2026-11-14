# UR5e Training Readiness and Milestone Plan

**Re-baselined:** November 13, 2026  
**Milestone 3 target:** November 18, 2026  
**Experiment freeze:** November 28, 2026  
**Final milestone target:** December 2, 2026

This plan combines the supplied Wang & Kasaei reproduction plan with the
paper’s state, reward, pose-error, action, and evaluation definitions. It is
specific to the current repository and is an execution plan only; it does not
implement the remaining work.

## 1. Current repository baseline

The MuJoCo environment already has:

- A six-joint UR5e model with six position actuators.
- A mocap target and three mocap obstacle bodies.
- A visible target orientation marker.
- A bent-arm default home pose:
  `[0, -pi/2, pi/2, -pi/2, -pi/2, 0]` radians.
- Seeded target and obstacle sampling.
- Robot qpos, qvel, and actuator reset.
- A 25-dimensional observation.
- Position and quaternion orientation error features.
- Paper-inspired reward and Gymnasium termination/truncation semantics.
- Reusable validation scripts for mocap objects and the environment.

The repository does **not** yet contain:

- A training implementation.
- PPO rollout storage, updates, or checkpointing.
- Vectorized training.
- AE or PF variants.
- A parity harness against the authors’ environment.
- A planner, waypoint follower, or perception pipeline.
- ROS 2/Gazebo integration.
- Experiment configuration, plots, or reproducible result aggregation.

The current environment also requires one explicit pre-training audit:

- Confirm whether one `env.step()` represents the paper’s `actionRepeat=100`
  simulation steps. The paper/released code uses absolute joint-position
  actions held for repeated simulation steps; the current environment advances
  only one `mj_step()` per call.

## 2. Non-negotiable reproduction decisions

These decisions should remain fixed unless a deliberate extension is labeled
separately:

1. **State:** use the paper’s 25-dimensional state:
   six joint positions, end-effector pose, target pose, two pose-error values,
   and three obstacle distances.
2. **Target orientation:** preserve the active reference orientation
   `(roll, pitch, yaw) = (0, pi/2, pi/2)` unless a separate orientation
   randomization experiment is announced.
3. **Actions:** treat actions as absolute desired joint positions, matching the
   released implementation and XML position actuators.
4. **Reward:** preserve the paper-inspired pose-error and obstacle-penalty
   formula for the main reproduction. Any acceleration or action-change
   penalty is a separate extension, not part of the comparable baseline.
5. **Success:** report position/axis error and orientation error separately,
   even if the environment combines them for a termination threshold.
6. **Evaluation:** use deterministic mean actions for deployment and measure
   smoothness rather than assuming AE or PF improves it.
7. **Seeds:** use three seeds for the final variants where time permits; use
   two only as an explicitly documented fallback.

## 3. Gates and deliverables

### Gate 0 — Environment contract and physics audit

**Deadline:** November 13

Before training, freeze and test the environment contract:

- Confirm action meaning and action range.
- Decide the MuJoCo simulation timestep and action-repeat equivalent.
- Confirm target and obstacle workspace bounds against Appendix A.4.
- Confirm the default home pose is collision-free and physically stable.
- Confirm `reset(seed)` deterministically reproduces all layout state.
- Confirm `terminated` and `truncated` are never conflated.
- Confirm the first observation after reset is finite and shape `(25,)`.
- Run Gymnasium’s environment checker and resolve failures, documenting
  warnings that are intentionally deferred.

**Deliverable:** an environment contract note and a passing smoke test.

**Blocker rule:** no PPO comparison run begins until this gate passes.

### Gate 1 — PPO learning sanity check

**Deadline:** November 14

Implement a plain PPO baseline before AE or PF:

- Actor with a six-dimensional Gaussian policy.
- Critic with a scalar value head.
- Explicit rollout buffer storing observations, actions, log-probabilities,
  rewards, values, termination flags, and truncation flags.
- Generalized advantage estimation with correct episode boundaries.
- Clipped PPO policy objective and value objective.
- Entropy term and gradient clipping.
- Checkpointing containing model, optimizer, normalization, configuration, seed,
  and training step.
- Deterministic evaluation using the actor mean action.

Start with one environment and a short run. Verify:

- Losses are finite.
- Advantages are finite.
- Policy parameters change.
- Rewards are not constant because of a data-flow bug.
- Reset occurs exactly once after termination/truncation.

Then run six to eight CPU environments with one Torch thread per worker.

**Deliverable:** plain-PPO learning curve and a reproducible baseline command.

**Gate criterion:** learning must show a meaningful upward trend and must not
be explained by an invalid reward, stale observation, or incorrect reset.

### Gate 2 — Reference parity and Milestone 3 minimum

**Deadline:** November 16

Build a parity harness using fixed inputs:

- Joint configuration.
- Target pose.
- Obstacle positions.
- One action.

Compare between the MuJoCo implementation and the authors’ environment where
the comparison is meaningful:

- 25-dimensional state components.
- Pose-error values.
- Reward terms.
- Success decision.

Record differences caused by physics engines separately from differences caused
by state/reward definitions. Do not claim exact parity where geometry or
contact behavior differs.

In parallel, implement the minimum planner path:

- Known obstacle positions as input.
- 3D voxel grid.
- A* from start to goal.
- B-spline smoothing.
- Waypoint sampling.
- Static-scene waypoint follower.

**Deliverable:** parity table, plain-PPO/reference plot, planner figure, and
static waypoint demo.

### Gate 3 — Milestone 3 video and submission package

**Deadline:** November 17 buffer; submit November 18

Prepare a short deterministic video containing:

1. Environment and pipeline overview.
2. Obstacle map, A* path, and smoothed B-spline.
3. Robot reaching randomized targets.
4. Policy tracking static-scene waypoints.
5. Initial reward and success-rate results with seed and budget stated.

Freeze the Milestone 3 evidence. Do not wait for PF or ROS 2 to make this
submission complete.

## 4. PPO implementation sequence

### 4.1 Plain PPO

Use the paper’s six-action setting and the current absolute-position action
semantics. Define and record:

- Number of environments.
- Rollout horizon.
- Discount factor.
- GAE lambda.
- Clip range.
- Learning rate and schedule.
- Batch size and epochs.
- Entropy coefficient.
- Value-loss coefficient.
- Gradient norm limit.
- Observation/reward normalization policy.

Normalization must be consistent between training and evaluation and saved in
the checkpoint.

### 4.2 Action ensemble (AE)

Implement AE only after plain PPO is learning:

- Draw multiple policy actions.
- Average them into the executed action.
- Use the Weibull schedule from Algorithm 1 to choose the sample count.
- Use `alpha=7` and `beta=20` initially, with clipping and integer rounding.
- Make the log-probability treatment explicit.

The paper does not fully specify the PPO ratio for an averaged action. Resolve
this by reading the authors’ `choose_action` implementation. If still
ambiguous, run and report the chosen interpretation as a reproduction
assumption; do not silently mix the probability of one sample with the
probability of the averaged action.

Measure:

- Mean absolute action change.
- Per-joint action standard deviation.
- Approximate jerk from successive command differences.
- Success and reward against plain PPO.

### 4.3 Policy feedback (PF)

Implement PF after AE and plain PPO are independently validated:

- Locate the authors’ `update` implementation.
- Determine how continuous policy density becomes a scalar in `[eta, 1]`.
- Use PF only in critic return/discount computation.
- Do not add PF as a reward term.
- Record the chosen `eta` and density normalization.

Run a small ablation first. If the behavior cannot be matched or becomes
numerically unstable, keep the reference result and clearly label the own
implementation as unresolved rather than changing the reward to compensate.

### 4.4 Four-variant run matrix

Run:

1. Plain PPO.
2. PPO + AE.
3. PPO + PF.
4. PPO + AE + PF.

Use the same:

- Environment contract.
- Step budget.
- Evaluation episodes.
- Seed list.
- Checkpoint and logging format.

## 5. Planner and controller integration

### Static planner

Implement in this order:

1. Occupancy/voxel grid from known MuJoCo obstacle positions.
2. Collision-aware A*.
3. B-spline smoothing with collision validation after smoothing.
4. Waypoint resampling at a defined spatial interval.
5. Waypoint advancement when end-effector distance is below tolerance.
6. Target-orientation handling for the final waypoint.
7. Policy input adapter that preserves the 25-D state contract.

The planner should remain independent of PPO. It supplies waypoint targets;
the controller supplies joint-position commands.

### Dynamic obstacle demonstration

Only after the static demo is stable:

- Move or insert one obstacle mid-episode.
- Detect map invalidation.
- Replan.
- Show the old path becoming invalid and the new path being followed.

## 6. Perception and video track

Defer live perception until the known-obstacle planner works.

Then add:

- MuJoCo RGB-D rendering with `mujoco.Renderer`.
- Headless EGL rendering for reproducible video.
- Segmentation/masking experiment.
- Height map.
- Voxel map update.
- Planner query.

Keep an offline known-obstacle figure as the fallback if FastSAM or the
perception stack is unstable.

## 7. ROS 2 and Gazebo sim-to-sim

Start the de-risking check immediately after Gate 1, but do not block
Milestone 3 on it:

- Verify one interpreter can import `torch`, `mujoco`, and `rclpy`, or document
  the split-environment solution.
- Verify ROS 2 Lyrical packages, ros2_control, the joint-group position
  controller, UR5e description, and Gazebo Jetty integration.
- Move one Gazebo joint through the controller before loading PPO.

The final node should:

- Load actor weights only.
- Use deterministic mean actions.
- Receive target/obstacle/waypoint data.
- Build the same 25-D state, including FK and pose errors.
- Publish six joint-position commands.
- Match the training action period, including the action-repeat decision.

If Lyrical/Jetty packages are unavailable, use the documented Jazzy/Harmonic
fallback and label it clearly.

## 8. Evaluation and reporting

For every PPO variant, record:

- Seed.
- Environment count.
- Total environment steps.
- Mean and standard deviation of return.
- Success rate versus error threshold.
- Position/axis error.
- Orientation error.
- Collision rate.
- Mean absolute action change.
- Jerk proxy.
- Planning time.
- Inference time.

Produce:

- Smoothed training curves with unsmoothed data retained.
- Success-vs-threshold curves from approximately 5 mm to 100 mm.
- Variant comparison table.
- Planner path and timing figure.
- Smoothness table.
- Sim-to-sim results and known limitations.

Never compare curves with different budgets or different reward definitions
without labeling the difference.

## 9. Revised calendar

### November 13–14

- Pass Gate 0.
- Implement and smoke-test plain PPO.
- Start six-to-eight-environment throughput measurement.
- De-risk ROS 2 imports/packages in parallel.

### November 15–16

- Run plain PPO baseline.
- Build parity harness.
- Implement known-obstacle grid, A*, smoothing, and waypoint follower.
- Prepare Milestone 3 figures.

### November 17–18

- Record deterministic static-scene video.
- Submit Milestone 3 package.
- Keep only a small buffer for fixes; do not expand scope.

### November 19–22

- Read and reproduce authors’ AE and PF details.
- Implement AE first, validate, then PF.
- Start three-seed runs of the four variants.

### November 23–25

- Complete multi-seed experiments.
- Build success, smoothness, and planning-time plots.
- Integrate static planner/controller demo.

### November 26–28

- Add dynamic obstacle replanning.
- Add perception loop if stable.
- Execute ROS 2/Gazebo sim-to-sim.
- Freeze experiments on November 28.

### November 29–December 1

- Generate final plots only from frozen logs.
- Write report.
- Assemble final video.
- Re-run reproducibility commands from a clean environment.

### December 2

- Final submission and archive of code, checkpoints, logs, figures, and video.

## 10. Cut order if schedule slips

Cut scope in this order:

1. OMPL comparison against RRT* and PRM.
2. Live perception; retain offline perception figures.
3. Three seeds down to two, with the limitation stated.
4. Own PF implementation; retain authors’ PF reference results.
5. Dynamic ROS 2 obstacle behavior.

Do not cut:

- Environment validation.
- Plain PPO baseline.
- At least one reproducible planner demo.
- Success and smoothness evaluation.
- Reproducible logs/checkpoints.

## 11. Definition of training-ready

Training is ready only when all of the following are true:

- `reset(seed)` is deterministic.
- The robot starts from the documented home pose.
- Target and obstacles are valid and randomized.
- The 25-D observation is finite and stable.
- Action semantics and action repeat are documented.
- Reward terms are independently inspectable.
- Success and truncation are separately tested.
- A short PPO run updates finite parameters and produces nonconstant returns.
- Checkpoints restore policy, optimizer, normalization, and configuration.
- Evaluation uses deterministic mean actions.
- A single command can reproduce a smoke run and save logs.

The first execution priority is therefore Gate 0 followed by the plain PPO
sanity check. AE, PF, planner integration, perception, and ROS 2 should not be
started as substitutes for a verified baseline.
