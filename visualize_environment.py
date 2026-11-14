import argparse
import time

import numpy as np

from ur5e_mujoco_env import UR5eMujocoEnv


def main(episodes, duration, seed):
    env = UR5eMujocoEnv(render_mode="human")
    try:
        for episode in range(episodes):
            observation, info = env.reset(seed=seed + episode)
            print(
                f"Episode {episode + 1}: "
                f"target={info['target_position']}, "
                f"quaternion={info['target_quaternion']}"
            )

            deadline = time.monotonic() + duration
            while time.monotonic() < deadline:
                action = np.asarray(env.data.qpos, dtype=np.float32)
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
                time.sleep(1 / env.metadata["render_fps"])
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize the UR5e MuJoCo environment."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of randomized episodes to display (default: 1).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Maximum seconds to display each episode (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the first episode; later episodes increment it.",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.duration <= 0:
        parser.error("--duration must be positive")
    main(args.episodes, args.duration, args.seed)
