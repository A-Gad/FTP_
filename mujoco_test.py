import gymnasium as gym
import time

# Initialize a standard MuJoCo environment with human rendering
env = gym.make("Ant-v4", render_mode="human")
observation, info = env.reset()

print("MuJoCo Environment Initialized!")

for i in range(200):
    # Sample a random action from the continuous action space
    action = env.action_space.sample()
    
    # Step the simulation forward
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Render the environment visually
    env.render()
    time.sleep(0.02)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
print("Smoke test complete.")