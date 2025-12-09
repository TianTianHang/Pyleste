"""
Simple test script to verify the CelesteGymEnv changes work properly.
"""
import sys
sys.path.append(".")

from celeste_env import CelesteGymEnv
import numpy as np

def test_env_with_max_steps():
    print("Testing CelesteGymEnv with max_steps parameter...")
    
    # Test with a small max_steps to verify truncation
    env = CelesteGymEnv(max_steps=10)
    
    print(f"Environment created with max_steps={env.max_steps}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Reset the environment
    obs, info = env.reset()
    print(f"Reset successful. Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Run a few steps to test functionality
    total_reward = 0
    for step in range(15):  # Run more steps than max_steps to test truncation
        action = env.action_space.sample()  # Random action
        next_obs, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        
        print(f"Step {step+1}: Action={action}, Reward={reward:.2f}, "
              f"Player Y={info['player_y']:.2f}, Current Step={info['current_step']}, "
              f"Done={done}, Truncated={truncated}")
        
        if done or truncated:
            print(f"Episode ended at step {info['current_step']}: Done={done}, Truncated={truncated}")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    env.close()
    print("Test completed successfully!")


def test_env_with_default_max_steps():
    print("\nTesting CelesteGymEnv with default max_steps...")
    
    # Test with default max_steps
    env = CelesteGymEnv()  # Use default max_steps=1000
    
    print(f"Environment created with max_steps={env.max_steps}")
    
    # Reset and run a few steps
    obs, info = env.reset()
    
    for step in range(5):
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        
        print(f"Step {step+1}: Action={action}, Reward={reward:.2f}, "
              f"Player Y={info['player_y']:.2f}, Truncated={truncated}")
        
        if done or truncated:
            print(f"Episode ended: Done={done}, Truncated={truncated}")
            break
    
    env.close()
    print("Default max_steps test completed!")


if __name__ == "__main__":
    test_env_with_max_steps()
    test_env_with_default_max_steps()
    
    print("\nAll tests completed successfully!")