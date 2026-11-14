import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class UR5eMujocoEnv(gym.Env):
    """MuJoCo UR5e reaching environment with movable target and obstacles."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    DEFAULT_HOME_QPOS = np.array(
        [0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0],
        dtype=np.float64,
    )

    def __init__(
        self,
        xml_file="ur5e.xml",
        render_mode=None,
        initial_qpos=None,
        max_episode_steps=500,
        action_repeat=1,
        target_rpy=None,
        max_sampling_attempts=1000,
        position_tolerance=0.08,
        orientation_tolerance=0.1,
        obstacle_clearance=0.08,
        position_only_success=False,
        incremental_actions=False,
        max_joint_delta=0.2,
        policy_action_bound=np.pi,
        success_bonus=50.0,
        curriculum_start_radius=None,
        curriculum_end_radius=None,
        curriculum_steps=0,
        full_workspace_probability=0.0,
        normalized_actions=False,
        progress_reward_scale=0.0,
        gravity_compensation=False,
        position_reward_weight=0.5,
        orientation_reward_weight=0.25,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)

        if self.model.nq != 6 or self.model.nv != 6 or self.model.nu != 6:
            raise ValueError(
                "UR5eMujocoEnv requires a six-joint, six-actuator model "
                f"(nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu})"
            )
        if max_episode_steps <= 0 or max_sampling_attempts <= 0:
            raise ValueError("episode and sampling limits must be positive")
        if action_repeat <= 0:
            raise ValueError("action_repeat must be positive")
        if position_tolerance <= 0 or orientation_tolerance <= 0:
            raise ValueError("pose tolerances must be positive")
        if obstacle_clearance <= 0:
            raise ValueError("obstacle_clearance must be positive")

        if initial_qpos is None:
            initial_qpos = self.DEFAULT_HOME_QPOS
        initial_qpos = np.asarray(initial_qpos, dtype=np.float64)
        if initial_qpos.shape != (self.model.nq,):
            raise ValueError(
                f"initial_qpos must have shape ({self.model.nq},), "
                f"got {initial_qpos.shape}"
            )
        self.initial_qpos = initial_qpos.copy()

        if target_rpy is None:
            target_rpy = (0.0, np.pi / 2, np.pi / 2)
        target_rpy = np.asarray(target_rpy, dtype=np.float64)
        if target_rpy.shape != (3,):
            raise ValueError(f"target_rpy must have shape (3,), got {target_rpy.shape}")
        self.target_rpy = target_rpy.copy()
        self.max_episode_steps = int(max_episode_steps)
        self.action_repeat = int(action_repeat)
        self.max_sampling_attempts = int(max_sampling_attempts)
        self.position_tolerance = float(position_tolerance)
        self.orientation_tolerance = float(orientation_tolerance)
        self.obstacle_clearance = float(obstacle_clearance)
        self.position_only_success = bool(position_only_success)
        if max_joint_delta <= 0:
            raise ValueError("max_joint_delta must be positive")
        self.incremental_actions = bool(incremental_actions)
        self.max_joint_delta = float(max_joint_delta)
        if policy_action_bound <= 0:
            raise ValueError("policy_action_bound must be positive")
        self.policy_action_bound = float(policy_action_bound)
        self.success_bonus = float(success_bonus)
        if curriculum_steps < 0 or not 0.0 <= full_workspace_probability <= 1.0:
            raise ValueError("curriculum_steps and full_workspace_probability are invalid")
        if (curriculum_start_radius is None) != (curriculum_end_radius is None):
            raise ValueError("curriculum radius bounds must be provided together")
        if curriculum_start_radius is not None and (
            curriculum_start_radius <= 0
            or curriculum_end_radius < curriculum_start_radius
        ):
            raise ValueError("curriculum radius bounds are invalid")
        self.curriculum_start_radius = (
            None if curriculum_start_radius is None else float(curriculum_start_radius)
        )
        self.curriculum_end_radius = (
            None if curriculum_end_radius is None else float(curriculum_end_radius)
        )
        self.curriculum_steps = int(curriculum_steps)
        self.full_workspace_probability = float(full_workspace_probability)
        self.normalized_actions = bool(normalized_actions)
        self.progress_reward_scale = float(progress_reward_scale)
        self.gravity_compensation = bool(gravity_compensation)
        if position_reward_weight < 0 or orientation_reward_weight < 0:
            raise ValueError("reward weights must be non-negative")
        self.position_reward_weight = float(position_reward_weight)
        self.orientation_reward_weight = float(orientation_reward_weight)

        self.target_body_id = self._named_id(mujoco.mjtObj.mjOBJ_BODY, "target_body")
        self.target_mocap_id = self.model.body_mocapid[self.target_body_id]
        self.target_site_id = self._named_id(mujoco.mjtObj.mjOBJ_SITE, "target")
        self.obstacle_body_ids = np.array(
            [self._named_id(mujoco.mjtObj.mjOBJ_BODY, f"obs_{i}_body") for i in range(1, 4)],
            dtype=np.int32,
        )
        self.obstacle_mocap_ids = self.model.body_mocapid[self.obstacle_body_ids].copy()
        self.obstacle_geom_ids = np.array(
            [self._named_id(mujoco.mjtObj.mjOBJ_GEOM, f"obs_{i}") for i in range(1, 4)],
            dtype=np.int32,
        )
        self.ee_site_id = self._named_id(mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.ee_body_id = self._named_id(mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
        self.link_body_ids = np.arange(
            self._named_id(mujoco.mjtObj.mjOBJ_BODY, "shoulder_link"),
            self.ee_body_id + 1,
            dtype=np.int32,
        )
        if self.target_mocap_id < 0 or np.any(self.obstacle_mocap_ids < 0):
            raise ValueError("target and obstacles must be attached to mocap bodies")

        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.target_position = np.zeros(3, dtype=np.float64)
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.obstacle_positions = np.zeros((3, 3), dtype=np.float64)
        self._previous_ee_quaternion = None
        self._episode_count = 0

        if self.incremental_actions or self.normalized_actions:
            action_low = np.full(self.model.nu, -1.0, dtype=np.float32)
            action_high = np.full(self.model.nu, 1.0, dtype=np.float32)
        else:
            action_low = np.full(self.model.nu, -self.policy_action_bound, dtype=np.float32)
            action_high = np.full(self.model.nu, self.policy_action_bound, dtype=np.float32)
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32
        )
        self.viewer = None
        self._previous_pose_error = None

        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._home_ee_position = self.data.site_xpos[self.ee_site_id].copy()
        self._refresh_object_state()

    def initial_policy_action(self):
        if self.incremental_actions:
            return np.zeros(self.model.nu, dtype=np.float32)
        if self.normalized_actions:
            low = self.model.actuator_ctrlrange[:, 0]
            high = self.model.actuator_ctrlrange[:, 1]
            return np.clip(
                2.0 * (self.initial_qpos - low) / (high - low) - 1.0,
                -1.0,
                1.0,
            ).astype(np.float32)
        return np.clip(self.initial_qpos, self.action_space.low, self.action_space.high).astype(np.float32)

    def _named_id(self, object_type, name):
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return object_id

    def _refresh_object_state(self):
        self.target_position = self.data.site_xpos[self.target_site_id].copy()
        self.target_quat = self.data.mocap_quat[self.target_mocap_id].copy()
        self.obstacle_positions = np.array(
            [self.data.geom_xpos[geom_id] for geom_id in self.obstacle_geom_ids],
            dtype=np.float64,
        )

    def _target_quaternion(self):
        quaternion = np.zeros(4, dtype=np.float64)
        mujoco.mju_euler2Quat(quaternion, self.target_rpy, "xyz")
        return quaternion

    def _sample_target_position(self):
        use_curriculum = self.curriculum_start_radius is not None
        use_full_workspace = (
            self.full_workspace_probability > 0.0
            and self.np_random.random() < self.full_workspace_probability
        )
        if use_curriculum and not use_full_workspace:
            progress = min(1.0, self._episode_count / max(1, self.curriculum_steps))
            radius = self.curriculum_start_radius + progress * (
                self.curriculum_end_radius - self.curriculum_start_radius
            )
            for _ in range(self.max_sampling_attempts):
                direction = self.np_random.normal(size=3)
                direction /= np.linalg.norm(direction)
                candidate = self._home_ee_position + direction * (
                    radius * self.np_random.random() ** (1.0 / 3.0)
                )
                radial_sq = candidate[0] ** 2 + candidate[1] ** 2
                if (
                    0.09 <= radial_sq
                    and np.dot(candidate, candidate) <= 0.5625
                    and 0.05 <= candidate[2] <= 0.75
                ):
                    return candidate.astype(np.float64)
        for _ in range(self.max_sampling_attempts):
            candidate = np.array(
                [
                    self.np_random.uniform(-1.0, 1.0),
                    self.np_random.uniform(0.0, 1.0),
                    self.np_random.uniform(0.05, 0.75),
                ],
                dtype=np.float64,
            )
            radial_sq = candidate[0] ** 2 + candidate[1] ** 2
            if 0.09 <= radial_sq and np.dot(candidate, candidate) <= 0.5625:
                return candidate
        raise RuntimeError("could not sample a valid target position")

    def _sample_obstacle_position(self, target, previous):
        for _ in range(self.max_sampling_attempts):
            candidate = np.array(
                [
                    self.np_random.uniform(-1.0, 1.0),
                    self.np_random.uniform(0.0, 1.0),
                    self.np_random.uniform(0.05, 0.75),
                ],
                dtype=np.float64,
            )
            radial_sq = candidate[0] ** 2 + candidate[1] ** 2
            if np.dot(candidate, candidate) > 0.25 or radial_sq < 0.04:
                continue
            if np.linalg.norm(candidate - target) < self.obstacle_clearance:
                continue
            if any(
                np.linalg.norm(candidate - other) < self.obstacle_clearance
                for other in previous
            ):
                continue
            return candidate
        raise RuntimeError("could not sample a valid obstacle position")

    def _set_mocap_scene(self):
        self.data.mocap_pos[self.target_mocap_id] = self.target_position
        self.data.mocap_quat[self.target_mocap_id] = self.target_quat
        for mocap_id, position in zip(self.obstacle_mocap_ids, self.obstacle_positions):
            self.data.mocap_pos[mocap_id] = position
            self.data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]

    def _site_quaternion(self, continuous=False):
        quaternion = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, self.data.site_xmat[self.ee_site_id])
        if continuous and self._previous_ee_quaternion is not None:
            if np.dot(self._previous_ee_quaternion, quaternion) < 0.0:
                quaternion = -quaternion
        if continuous:
            self._previous_ee_quaternion = quaternion.copy()
        return quaternion

    @staticmethod
    def _axis_points(position, quaternion, distance=0.1):
        rotation = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, quaternion)
        return position + distance * rotation.reshape(3, 3)

    @staticmethod
    def _quaternion_error(first, second):
        dot = np.clip(abs(float(np.dot(first, second))), -1.0, 1.0)
        return 2.0 * np.arccos(dot)

    def _obstacle_distances(self):
        link_points = np.array(
            [self.data.xpos[body_id] for body_id in self.link_body_ids]
            + [self.data.site_xpos[self.ee_site_id]],
            dtype=np.float64,
        )
        distances = []
        for obstacle_id in self.obstacle_geom_ids:
            point = self.data.geom_xpos[obstacle_id]
            segment_distances = []
            for start, end in zip(link_points[:-1], link_points[1:]):
                direction = end - start
                denominator = np.dot(direction, direction)
                parameter = 0.0 if denominator == 0.0 else np.dot(point - start, direction) / denominator
                parameter = np.clip(parameter, 0.0, 1.0)
                segment_distances.append(np.linalg.norm(point - (start + parameter * direction)))
            distances.append(min(segment_distances))
        return np.asarray(distances, dtype=np.float64)

    def _pose_errors(self):
        ee_position = self.data.site_xpos[self.ee_site_id].copy()
        ee_quaternion = self._site_quaternion()
        target_points = self._axis_points(self.target_position, self.target_quat)
        ee_points = self._axis_points(ee_position, ee_quaternion)
        axis_error = float(np.linalg.norm(ee_points - target_points, axis=1).sum())
        orientation_error = self._quaternion_error(ee_quaternion, self.target_quat)
        return axis_error, orientation_error

    def _get_obs(self):
        joint_positions = self.data.qpos.copy()
        ee_position = self.data.site_xpos[self.ee_site_id].copy()
        ee_quaternion = self._site_quaternion(continuous=True)
        axis_error, orientation_error = self._pose_errors()
        observation = np.concatenate(
            [
                joint_positions,
                ee_position,
                ee_quaternion,
                self.target_position,
                self.target_quat,
                [axis_error, orientation_error],
                self._obstacle_distances(),
            ]
        )
        if observation.shape != self.observation_space.shape:
            raise RuntimeError(f"unexpected observation shape: {observation.shape}")
        return observation.astype(np.float32)

    def _reward(self, observation):
        axis_error, orientation_error = observation[20:22]
        pose_error = (
            self.position_reward_weight * axis_error
            + self.orientation_reward_weight * orientation_error
        )
        obstacle_penalty = np.maximum(
            0.0, 1.0 - observation[22:] / self.obstacle_clearance
        ).sum()
        reward = float(
            -0.001 * pose_error**2
            - np.log(pose_error**2 + 0.0001)
            - 0.1 * obstacle_penalty
        )
        if self.progress_reward_scale and self._previous_pose_error is not None:
            reward += self.progress_reward_scale * np.clip(
                self._previous_pose_error - pose_error, -1.0, 1.0
            )
        self._previous_pose_error = pose_error
        return reward

    def _is_success(self, observation):
        axis_error, orientation_error = observation[20:22]
        return bool(
            axis_error <= self.position_tolerance
            and (
                self.position_only_success
                or orientation_error <= self.orientation_tolerance
            )
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(f"action must have shape {self.action_space.shape}, got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("action must contain only finite values")
        if self.incremental_actions:
            delta = action * self.max_joint_delta
            target = self.data.ctrl + delta
            self.data.ctrl[:] = np.clip(
                target,
                self.model.actuator_ctrlrange[:, 0],
                self.model.actuator_ctrlrange[:, 1],
            )
        else:
            if self.normalized_actions:
                low = self.model.actuator_ctrlrange[:, 0]
                high = self.model.actuator_ctrlrange[:, 1]
                self.data.ctrl[:] = low + 0.5 * (np.clip(action, -1.0, 1.0) + 1.0) * (high - low)
            else:
                self.data.ctrl[:] = np.clip(action, self.action_space.low, self.action_space.high)
            self.data.ctrl[:] = np.clip(
                self.data.ctrl,
                self.model.actuator_ctrlrange[:, 0],
                self.model.actuator_ctrlrange[:, 1],
            )
        reward = 0.0
        for _ in range(self.action_repeat):
            if self.gravity_compensation:
                mujoco.mj_step1(self.model, self.data)
                torque_limits = self.model.jnt_actfrcrange[:, 1]
                self.data.qfrc_applied[:] = np.clip(
                    self.data.qfrc_bias, -torque_limits, torque_limits
                )
                mujoco.mj_step2(self.model, self.data)
            else:
                self.data.qfrc_applied[:] = 0.0
                mujoco.mj_step(self.model, self.data)
            observation = self._get_obs()
            reward = self._reward(observation)
            self.terminated = self._is_success(observation)
            if self.terminated:
                reward += self.success_bonus
                break
        self.step_count += 1
        self.truncated = self.step_count >= self.max_episode_steps and not self.terminated

        info = {
            "is_success": self.terminated,
            "axis_error": float(observation[20]),
            "orientation_error": float(observation[21]),
            "obstacle_distances": observation[22:].copy(),
            "action_repeat": self.action_repeat,
            "incremental_actions": self.incremental_actions,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, self.terminated, self.truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self._previous_ee_quaternion = None
        self._previous_pose_error = None

        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.initial_qpos
        self._episode_count += 1
        self.target_position = self._sample_target_position()
        self.target_quat = self._target_quaternion()
        sampled_obstacles = []
        for _ in range(3):
            sampled_obstacles.append(
                self._sample_obstacle_position(self.target_position, sampled_obstacles)
            )
        self.obstacle_positions = np.asarray(sampled_obstacles, dtype=np.float64)
        self._set_mocap_scene()
        mujoco.mj_forward(self.model, self.data)
        self._refresh_object_state()

        observation = self._get_obs()
        info = {
            "target_position": self.target_position.copy(),
            "target_quaternion": self.target_quat.copy(),
            "obstacle_positions": self.obstacle_positions.copy(),
        }
        if self.render_mode == "human":
            self.render()
        return observation, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer

                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
