import argparse
from pathlib import Path

import numpy as np
import torch

from ppo import PPO
from observation_normalizer import ObservationNormalizer
from ur5e_mujoco_env import UR5eMujocoEnv


def main(checkpoint_path, episodes, seed, device, action_repeat=None):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    environment_config = checkpoint.get("environment", {})
    env = UR5eMujocoEnv(
        action_repeat=(
            action_repeat
            if action_repeat is not None
            else environment_config.get("action_repeat", 1)
        ),
        max_episode_steps=environment_config.get("max_episode_steps", 500),
        position_only_success=environment_config.get("position_only_success", False),
        incremental_actions=environment_config.get("incremental_actions", False),
        max_joint_delta=environment_config.get("max_joint_delta", 0.2),
        policy_action_bound=environment_config.get("policy_action_bound", np.pi),
        curriculum_start_radius=environment_config.get("curriculum_start_radius"),
        curriculum_end_radius=environment_config.get("curriculum_end_radius"),
        curriculum_steps=environment_config.get("curriculum_steps", 0),
        full_workspace_probability=environment_config.get("full_workspace_probability", 0.0),
        normalized_actions=environment_config.get("normalized_actions", False),
        progress_reward_scale=environment_config.get("progress_reward_scale", 0.0),
        gravity_compensation=environment_config.get("gravity_compensation", False),
        position_reward_weight=environment_config.get("position_reward_weight", 0.5),
        orientation_reward_weight=environment_config.get("orientation_reward_weight", 0.25),
    )
    normalizer_state = checkpoint.get("observation_normalizer")
    normalizer = (
        ObservationNormalizer.from_state_dict(normalizer_state)
        if normalizer_state is not None
        else None
    )
    saved_config = checkpoint["ppo"]["config"]
    agent = PPO(
        env.observation_space.shape[0],
        env.action_space.low,
        env.action_space.high,
        device=device,
    )
    agent.load_state_dict(checkpoint["ppo"])
    returns = []
    successes = 0
    axis_errors = []
    orientation_errors = []
    minimum_obstacle_distances = []
    action_changes = []
    jerk_proxies = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            if normalizer is not None:
                observation = normalizer.normalize(observation)
            episode_return = 0.0
            previous_action = None
            previous_delta = None
            for _ in range(env.max_episode_steps):
                action, _, _ = agent.act(observation, deterministic=True)
                if previous_action is not None:
                    delta = action - previous_action
                    action_changes.append(float(np.abs(delta).mean()))
                    if previous_delta is not None:
                        jerk_proxies.append(float(np.abs(delta - previous_delta).mean()))
                    previous_delta = delta
                previous_action = action.copy()
                observation, reward, terminated, truncated, info = env.step(action)
                if normalizer is not None and not (terminated or truncated):
                    observation = normalizer.normalize(observation)
                episode_return += reward
                if terminated or truncated:
                    successes += int(info["is_success"])
                    axis_errors.append(info["axis_error"])
                    orientation_errors.append(info["orientation_error"])
                    minimum_obstacle_distances.append(
                        float(np.min(info["obstacle_distances"]))
                    )
                    break
            returns.append(episode_return)
    finally:
        env.close()
    print(
        {
            "checkpoint": str(Path(checkpoint_path)),
            "config": saved_config,
            "mean_return": float(np.mean(returns)),
            "success_rate": successes / episodes,
            "mean_final_axis_error": float(np.mean(axis_errors)),
            "mean_final_orientation_error": float(np.mean(orientation_errors)),
            "mean_min_obstacle_distance": float(np.mean(minimum_obstacle_distances)),
            "mean_action_change": float(np.mean(action_changes)),
            "mean_action_jerk_proxy": float(np.mean(jerk_proxies)),
            "action_repeat": env.action_repeat,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a PPO checkpoint.")
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-repeat", type=int, default=None)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.action_repeat is not None and args.action_repeat <= 0:
        parser.error("--action-repeat must be positive")
    main(args.checkpoint, args.episodes, args.seed, args.device, args.action_repeat)
