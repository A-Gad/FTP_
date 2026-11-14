"""Bounded IK-teacher feasibility test; does not modify shared training code."""
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ur5e_mujoco_env import UR5eMujocoEnv


def pose_error(env, target_position, target_quat):
    position = env.data.site_xpos[env.ee_site_id].copy()
    rotation = Rotation.from_matrix(env.data.site_xmat[env.ee_site_id].reshape(3, 3))
    target_rotation = Rotation.from_quat(
        [target_quat[1], target_quat[2], target_quat[3], target_quat[0]]
    )
    return np.r_[target_position - position, (target_rotation * rotation.inv()).as_rotvec()]


def solve_ik(env, target_position, target_quat, q0, iterations=250):
    q = np.array(q0, dtype=np.float64, copy=True)
    lower, upper = env.model.jnt_range[:6, 0], env.model.jnt_range[:6, 1]
    # Match the existing PPO action bound; otherwise a valid IK solution can
    # be silently clipped by UR5eMujocoEnv.step().
    lower = np.maximum(lower, -env.policy_action_bound)
    upper = np.minimum(upper, env.policy_action_bound)
    for _ in range(iterations):
        env.data.qpos[:6] = q
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        error = pose_error(env, target_position, target_quat)
        if np.linalg.norm(error[:3]) < 0.003 and np.linalg.norm(error[3:]) < 0.03:
            break
        jac_pos = np.zeros((3, env.model.nv))
        jac_rot = np.zeros((3, env.model.nv))
        mujoco.mj_jacSite(env.model, env.data, jac_pos, jac_rot, env.ee_site_id)
        jac = np.vstack((jac_pos[:, :6], jac_rot[:, :6]))
        damping = 0.03
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), error)
        q = np.clip(q + 0.65 * dq, lower + 1e-4, upper - 1e-4)
    env.data.qpos[:6] = q
    mujoco.mj_forward(env.model, env.data)
    return q, pose_error(env, target_position, target_quat)


def main():
    episodes = 8
    env = UR5eMujocoEnv(
        max_episode_steps=80,
        action_repeat=100,
        incremental_actions=True,
        max_joint_delta=0.2,
    )
    rng = np.random.default_rng(7)
    position_errors, orientation_errors, successes, tracking_steps = [], [], 0, []
    try:
        for episode in range(episodes):
            _, info = env.reset(seed=episode)
            q_target, residual = solve_ik(
                env, info["target_position"], info["target_quaternion"], env.initial_qpos
            )
            position_errors.append(float(np.linalg.norm(residual[:3])))
            orientation_errors.append(float(np.linalg.norm(residual[3:])))

            # Re-run from reset and use normalized joint-target deltas as the teacher action.
            env.reset(seed=episode)
            reached = False
            for step in range(80):
                action = np.clip(
                    (q_target - env.data.ctrl[:6]) / env.max_joint_delta, -1.0, 1.0
                ).astype(np.float32)
                _, _, terminated, truncated, step_info = env.step(action)
                if terminated:
                    successes += 1
                    reached = True
                    tracking_steps.append(step + 1)
                    break
                if truncated:
                    break
            if not reached:
                tracking_steps.append(80)
            print(
                f"episode={episode} ik_pos={position_errors[-1]:.4f}m "
                f"ik_rot={orientation_errors[-1]:.4f}rad "
                f"tracked_success={reached} steps={tracking_steps[-1]}"
            )
    finally:
        env.close()
    print(
        "SUMMARY "
        f"episodes={episodes} "
        f"mean_ik_pos={np.mean(position_errors):.4f}m "
        f"max_ik_pos={np.max(position_errors):.4f}m "
        f"mean_ik_rot={np.mean(orientation_errors):.4f}rad "
        f"teacher_success={successes}/{episodes} "
        f"mean_success_steps={np.mean([s for s, ok in zip(tracking_steps, [True]*len(tracking_steps)) if s < 80]) if successes else float('nan'):.1f}"
    )


if __name__ == "__main__":
    main()
