import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces

class UR5eMujocoEnv(gym.Env):
    """Custom Environment that follows gymnasium interface"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        xml_file="ur5e.xml",
        render_mode=None,
        initial_qpos=None,
        max_episode_steps=500,
    ):
        super().__init__()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        
        if self.model.nq != 6 or self.model.nv != 6 or self.model.nu != 6:
            raise ValueError(
                "UR5eMujocoEnv requires a six-joint, six-actuator MuJoCo model "
                f"(nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu})"
            )
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")

        if initial_qpos is None:
            initial_qpos = np.zeros(self.model.nq, dtype=np.float64)
        else:
            initial_qpos = np.asarray(initial_qpos, dtype=np.float64)
        if initial_qpos.shape != (self.model.nq,):
            raise ValueError(
                f"initial_qpos must have shape ({self.model.nq},), "
                f"got {initial_qpos.shape}"
            )

        self.initial_qpos = initial_qpos.copy()
        self.max_episode_steps = int(max_episode_steps)
        self.step_count = 0
        self.terminated = False
        self.truncated = False

        mujoco.mj_forward(self.model, self.data)
        self.target_position = self.data.site("target").xpos.copy()
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.obstacle_positions = np.array(
            [
                self.data.geom("obs_1").xpos,
                self.data.geom("obs_2").xpos,
                self.data.geom("obs_3").xpos,
            ],
            dtype=np.float64,
        )

        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)

        self.viewer = None

    def _get_obs(self):
        # Translate PyBullet's getJointState and getLinkState to MuJoCo:
        # self.data.qpos -> returns an array of current joint positions
        # self.data.qvel -> returns an array of current joint velocities
        # self.data.body("link_name").xpos -> returns Cartesian XYZ of a specific link
        
        # Example: Assemble your 25-D state array here just like in the old gym_env.py
        # obs = np.concatenate([joint_angles, tool_position, obstacle_distances...])
        
        # Placeholder for compilation:
        obs = np.zeros(25, dtype=np.float32)
        return obs

    def step(self, action):
        # 1. Apply the action to the robot's actuators
        self.data.ctrl[:] = action
        
        # 2. Step the physics simulation forward
        mujoco.mj_step(self.model, self.data)
        
        # 3. Calculate your custom reward (Eq. 18-19 from your paper)
        reward = 0.0 
        
        # 4. Check termination conditions (e.g., reached goal or hit collision threshold)
        terminated = False
        truncated = False # Use for max time-steps reached
        
        # 5. Get the new observation
        obs = self._get_obs()
        
        # 6. Info dictionary for debugging metrics (like 'distance_to_goal')
        info = {}
        
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Reset the simulation to its initial state
        mujoco.mj_resetData(self.model, self.data)
        
        # 2. Apply andom target/obstacle sampling (Appendix A.4)
        
        
        # 3. Forward the simulation once to update the kinematics with the new setup
        mujoco.mj_forward(self.model, self.data)
        
        obs = self._get_obs()
        info = {}
        
        if self.render_mode == "human":
            self.render()
            
        return obs, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                # Use the native mujoco viewer for gymnasium
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            
            # Sync the viewer with the current physics state
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None