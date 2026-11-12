from ur5e_mujoco_env import UR5eMujocoEnv
import numpy as np
import time

env = UR5eMujocoEnv(render_mode="human")
obs, info = env.reset()

print("Custom MuJoCo UR5e Environment Initialized!")

# Define a fixed upright/ready pose (6 joint angles in radians)
fixed_action = np.array([1.57, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float32)

for _ in range(500):
    # Send the exact same position command every step
    obs, reward, terminated, truncated, info = env.step(fixed_action)
    time.sleep(0.02)

env.close()
print("Test completed successfully.")
import os
os._exit(0)