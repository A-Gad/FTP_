import numpy as np

from ur5e_mujoco_env import UR5eMujocoEnv


def main():
    env = UR5eMujocoEnv(max_episode_steps=3)
    try:
        observation, info = env.reset(seed=123)
        assert observation.shape == (25,)
        assert np.isfinite(observation).all()
        assert np.isclose(np.linalg.norm(info["target_quaternion"]), 1.0)
        assert np.allclose(env.data.qpos, env.initial_qpos)
        assert np.allclose(env.data.qvel, 0.0)
        assert np.allclose(env.data.ctrl, env.initial_qpos)

        first_observation = observation.copy()
        first_target = info["target_position"].copy()
        first_quaternion = info["target_quaternion"].copy()
        first_obstacles = info["obstacle_positions"].copy()

        observation, repeated_info = env.reset(seed=123)
        assert np.allclose(observation, first_observation)
        assert np.allclose(repeated_info["target_position"], first_target)
        assert np.allclose(repeated_info["target_quaternion"], first_quaternion)
        assert np.allclose(repeated_info["obstacle_positions"], first_obstacles)

        for _ in range(3):
            observation, _, terminated, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
        assert observation.shape == (25,)
        assert not terminated
        assert truncated

        env.step(np.ones(6, dtype=np.float32))
        env.reset(seed=123)
        assert np.allclose(env.data.qpos, env.initial_qpos)
        assert np.allclose(env.data.qvel, 0.0)
        assert env.step_count == 0
        assert not env.terminated
        assert not env.truncated
    finally:
        env.close()

    print("Environment validation passed.")


if __name__ == "__main__":
    main()
