import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from ppo import PPO, PPOConfig, RolloutBuffer
from ur5e_mujoco_env import UR5eMujocoEnv


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    def make_env(index):
        def factory():
            return UR5eMujocoEnv(
                max_episode_steps=args.max_episode_steps,
                action_repeat=args.action_repeat,
                gravity_compensation=args.gravity_compensation,
                curriculum_start_radius=args.curriculum_start_radius,
                curriculum_end_radius=args.curriculum_end_radius,
                curriculum_steps=args.curriculum_steps,
                full_workspace_probability=args.full_workspace_probability,
            )
        return factory

    vector_cls = gym.vector.AsyncVectorEnv if args.async_envs else gym.vector.SyncVectorEnv
    envs = vector_cls([make_env(i) for i in range(args.num_envs)])
    observation_space = envs.single_observation_space
    action_space = envs.single_action_space
    config = PPOConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        entropy_coef=args.entropy_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        action_std=args.action_std,
        initial_action=tuple(UR5eMujocoEnv.DEFAULT_HOME_QPOS.tolist()),
    )
    agent = PPO(
        observation_space.shape[0],
        action_space.low,
        action_space.high,
        config=config,
        device=args.device,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    observations, _ = envs.reset(seed=args.seed)
    metrics = []

    try:
        for update in range(1, args.updates + 1):
            obs_rollout = np.zeros(
                (args.rollout_length, args.num_envs, *observations.shape[1:]),
                dtype=np.float32,
            )
            action_rollout = np.zeros(
                (args.rollout_length, args.num_envs, *envs.single_action_space.shape),
                dtype=np.float32,
            )
            reward_rollout = np.zeros((args.rollout_length, args.num_envs), dtype=np.float32)
            done_rollout = np.zeros((args.rollout_length, args.num_envs), dtype=np.bool_)
            value_rollout = np.zeros_like(reward_rollout)
            log_prob_rollout = np.zeros_like(reward_rollout)

            for step in range(args.rollout_length):
                actions, log_probs, values = agent.act_batch(observations)
                next_observations, rewards, terminated, truncated, _ = envs.step(actions)
                obs_rollout[step] = observations
                action_rollout[step] = actions
                reward_rollout[step] = rewards
                done_rollout[step] = np.logical_or(terminated, truncated)
                value_rollout[step] = values
                log_prob_rollout[step] = log_probs
                observations = next_observations

            with torch.no_grad():
                last_values = agent.policy.value(
                    torch.as_tensor(observations, dtype=torch.float32, device=agent.device)
                ).cpu().numpy()
            advantages = np.zeros_like(reward_rollout)
            returns = np.zeros_like(reward_rollout)
            gae = np.zeros(args.num_envs, dtype=np.float32)
            next_values = last_values
            for step in reversed(range(args.rollout_length)):
                not_done = 1.0 - done_rollout[step].astype(np.float32)
                delta = reward_rollout[step] + args.gamma * next_values * not_done - value_rollout[step]
                gae = delta + args.gamma * args.gae_lambda * not_done * gae
                advantages[step] = gae
                returns[step] = gae + value_rollout[step]
                next_values = value_rollout[step]

            buffer = RolloutBuffer(
                args.rollout_length * args.num_envs,
                observation_space.shape,
                action_space.shape,
            )
            buffer.observations[:] = obs_rollout.transpose(1, 0, 2).reshape(
                buffer.observations.shape
            )
            buffer.actions[:] = action_rollout.transpose(1, 0, 2).reshape(buffer.actions.shape)
            buffer.rewards[:] = reward_rollout.T.reshape(-1)
            buffer.values[:] = value_rollout.T.reshape(-1)
            buffer.log_probs[:] = log_prob_rollout.T.reshape(-1)
            buffer.advantages[:] = advantages.T.reshape(-1)
            buffer.returns[:] = returns.T.reshape(-1)
            buffer.position = buffer.size
            update_metrics = agent.update(buffer)
            update_metrics.update({"update": update, "environment_steps": update * args.rollout_length * args.num_envs})
            metrics.append(update_metrics)
            print(json.dumps(update_metrics))

        checkpoint = {
            "ppo": agent.state_dict(),
            "seed": args.seed,
            "environment_steps": args.updates * args.rollout_length * args.num_envs,
            "environment": {
                "action_repeat": args.action_repeat,
                "max_episode_steps": args.max_episode_steps,
                "gravity_compensation": args.gravity_compensation,
                "curriculum_start_radius": args.curriculum_start_radius,
                "curriculum_end_radius": args.curriculum_end_radius,
                "curriculum_steps": args.curriculum_steps,
                "full_workspace_probability": args.full_workspace_probability,
            },
            "observation_normalizer": None,
        }
        torch.save(checkpoint, output_dir / "ppo_checkpoint.pt")
        with (output_dir / "metrics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=metrics[0].keys())
            writer.writeheader()
            writer.writerows(metrics)
        (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    finally:
        envs.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--async-envs", action="store_true")
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--rollout-length", type=int, default=32)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--action-repeat", type=int, default=100)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--gae-lambda", type=float, default=0.0)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--action-std", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="runs/ppo_vectorized_smoke")
    parser.add_argument("--gravity-compensation", action="store_true")
    parser.add_argument("--curriculum-start-radius", type=float, default=0.08)
    parser.add_argument("--curriculum-end-radius", type=float, default=0.55)
    parser.add_argument("--curriculum-steps", type=int, default=300)
    parser.add_argument("--full-workspace-probability", type=float, default=0.2)
    args = parser.parse_args()
    if args.num_envs <= 0 or args.rollout_length <= 0:
        parser.error("--num-envs and --rollout-length must be positive")
    if args.rollout_length * args.num_envs < args.minibatch_size:
        parser.error("total rollout size must be at least --minibatch-size")
    train(args)


if __name__ == "__main__":
    main()
