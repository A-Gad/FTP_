import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces

class UR5eMujocoEnv(gym.Env):
    """Custom Environment that follows gymnasium interface"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, xml_file="ur5e.xml", render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # 1. Load the MuJoCo model and data from your newly created XML
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        
        # 2. Define the Action Space (6 continuous joint commands for UR5e)
        # Using -3.14 to 3.14 (pi) radians as the max action range
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32)
        
        # 3. Define the Observation Space (25-D continuous state to match your paper)
        # Assuming normalized bounds between -inf and inf for simplicity
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)

        # Optional: Initialize a viewer if rendering is requested
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
        
        # 2. Apply your random target/obstacle sampling here (Appendix A.4)
        # e.g., self.data.body("target_name").xpos = [new_x, new_y, new_z]
        
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