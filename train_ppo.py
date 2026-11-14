import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ppo import PPO, PPOConfig, RolloutBuffer
from observation_normalizer import ObservationNormalizer
from ur5e_mujoco_env import UR5eMujocoEnv


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = UR5eMujocoEnv(
        max_episode_steps=args.max_episode_steps,
        action_repeat=args.action_repeat,
        position_only_success=args.position_only_success,
        incremental_actions=args.incremental_actions,
        max_joint_delta=args.max_joint_delta,
        policy_action_bound=args.policy_action_bound,
        curriculum_start_radius=args.curriculum_start_radius,
        curriculum_end_radius=args.curriculum_end_radius,
        curriculum_steps=args.curriculum_steps,
        full_workspace_probability=args.full_workspace_probability,
        normalized_actions=args.normalized_actions,
        progress_reward_scale=args.progress_reward_scale,
        gravity_compensation=args.gravity_compensation,
        position_reward_weight=args.position_reward_weight,
        orientation_reward_weight=args.orientation_reward_weight,
    )
    config = PPOConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        entropy_coef=args.entropy_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        action_std=args.action_std,
        initial_action=(
            tuple(np.zeros(env.model.nu, dtype=np.float64))
            if args.incremental_actions
            else             tuple(env.initial_policy_action().tolist())
        ),
    )
    agent = PPO(
        observation_dim=env.observation_space.shape[0],
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        config=config,
        device=args.device,
    )
    normalizer = (
        ObservationNormalizer(env.observation_space.shape)
        if args.normalize_observations
        else None
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    observation, _ = env.reset(seed=args.seed)
    if normalizer is not None:
        normalizer.update(observation)
        observation = normalizer.normalize(observation)
    metrics = []
    episode_return = 0.0
    completed_returns = []

    try:
        for update in range(1, args.updates + 1):
            buffer = RolloutBuffer(
                args.rollout_length,
                env.observation_space.shape,
                env.action_space.shape,
            )
            for _ in range(args.rollout_length):
                action, log_prob, value = agent.act(observation)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                # Compute the value of the final observation for truncated episodes
                # so bootstrapping uses the correct state.
                truncated_value = 0.0
                if truncated:
                    with torch.no_grad():
                        # Value the observation *before* reset, not after
                        obs_for_value = next_observation
                        if normalizer is not None:
                            obs_for_value = normalizer.normalize(obs_for_value)
                        truncated_value = agent.policy.value(
                            torch.as_tensor(obs_for_value, dtype=torch.float32, device=agent.device).unsqueeze(0)
                        ).item()
                buffer.add(
                    observation, action, reward, terminated, truncated, value, log_prob,
                    truncated_value=truncated_value,
                )
                episode_return += reward
                observation = next_observation
                if terminated or truncated:
                    completed_returns.append(episode_return)
                    episode_return = 0.0
                    observation, _ = env.reset()
                if normalizer is not None:
                    normalizer.update(observation)
                    observation = normalizer.normalize(observation)

            with torch.no_grad():
                last_value = agent.policy.value(
                    torch.as_tensor(observation, dtype=torch.float32, device=agent.device).unsqueeze(0)
                ).item()
            buffer.finish(last_value, config.gamma, config.gae_lambda)
            update_metrics = agent.update(buffer)
            update_metrics.update(
                {
                    "update": update,
                    "environment_steps": update * args.rollout_length,
                    "episodes_completed": len(completed_returns),
                    "mean_episode_return": float(np.mean(completed_returns[-10:]))
                    if completed_returns
                    else float("nan"),
                }
            )
            metrics.append(update_metrics)
            print(json.dumps(update_metrics))

        checkpoint = {
            "ppo": agent.state_dict(),
            "seed": args.seed,
            "environment_steps": args.updates * args.rollout_length,
            "environment": {
                "action_repeat": args.action_repeat,
                "max_episode_steps": args.max_episode_steps,
                "position_only_success": args.position_only_success,
                "incremental_actions": args.incremental_actions,
                "max_joint_delta": args.max_joint_delta,
                "policy_action_bound": args.policy_action_bound,
                "curriculum_start_radius": args.curriculum_start_radius,
                "curriculum_end_radius": args.curriculum_end_radius,
                "curriculum_steps": args.curriculum_steps,
                "full_workspace_probability": args.full_workspace_probability,
                "normalized_actions": args.normalized_actions,
                "progress_reward_scale": args.progress_reward_scale,
                "gravity_compensation": args.gravity_compensation,
                "position_reward_weight": args.position_reward_weight,
                "orientation_reward_weight": args.orientation_reward_weight,
            },
            "observation_normalizer": (
                normalizer.state_dict() if normalizer is not None else None
            ),
        }
        torch.save(checkpoint, output_dir / "ppo_checkpoint.pt")
        with (output_dir / "metrics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=metrics[0].keys())
            writer.writeheader()
            writer.writerows(metrics)
        (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Train one-environment plain PPO.")
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--rollout-length", type=int, default=256)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--action-std", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="runs/ppo_smoke")
    parser.add_argument("--normalize-observations", action="store_true")
    parser.add_argument("--position-only-success", action="store_true")
    parser.add_argument("--incremental-actions", action="store_true")
    parser.add_argument("--max-joint-delta", type=float, default=0.2)
    parser.add_argument("--policy-action-bound", type=float, default=np.pi)
    parser.add_argument("--curriculum-start-radius", type=float, default=None)
    parser.add_argument("--curriculum-end-radius", type=float, default=None)
    parser.add_argument("--curriculum-steps", type=int, default=0)
    parser.add_argument("--full-workspace-probability", type=float, default=0.0)
    parser.add_argument("--normalized-actions", action="store_true")
    parser.add_argument("--progress-reward-scale", type=float, default=0.0)
    parser.add_argument("--gravity-compensation", action="store_true")
    parser.add_argument("--position-reward-weight", type=float, default=0.5)
    parser.add_argument("--orientation-reward-weight", type=float, default=0.25)
    args = parser.parse_args()
    if args.updates <= 0 or args.rollout_length <= 0 or args.action_repeat <= 0:
        parser.error("--updates, --rollout-length, and --action-repeat must be positive")
    if args.action_std <= 0:
        parser.error("--action-std must be positive")
    if args.max_joint_delta <= 0:
        parser.error("--max-joint-delta must be positive")
    if args.policy_action_bound <= 0:
        parser.error("--policy-action-bound must be positive")
    if args.position_reward_weight < 0 or args.orientation_reward_weight < 0:
        parser.error("reward weights must be non-negative")
    if (args.curriculum_start_radius is None) != (args.curriculum_end_radius is None):
        parser.error("curriculum radius bounds must be provided together")
    if args.curriculum_start_radius is not None and (
        args.curriculum_start_radius <= 0
        or args.curriculum_end_radius < args.curriculum_start_radius
    ):
        parser.error("curriculum radius bounds are invalid")
    if args.curriculum_steps < 0 or not 0.0 <= args.full_workspace_probability <= 1.0:
        parser.error("invalid curriculum schedule parameters")
    if args.rollout_length < args.minibatch_size:
        parser.error("--rollout-length must be at least --minibatch-size")
    train(args)


if __name__ == "__main__":
    main()
