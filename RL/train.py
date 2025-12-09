import os
import sys
sys.path.append("")
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from celeste_env import CelesteGymEnv
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
class SymbolicMapCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]  # 3
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 推断展平后的维度
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
    def forward(self, x):
        return self.linear(self.cnn(x))

# 使用
policy_kwargs = dict(features_extractor_class=SymbolicMapCNN)

def train_agent_sb3(algorithm='PPO', total_timesteps=100000):
    """
    Train a reinforcement learning agent using Stable Baselines3
    """
    print(f"Training with {algorithm} algorithm...")
    
    # Create environment
    # For algorithms that need vectorized environments, we use make_vec_env
    env = make_vec_env(CelesteGymEnv, n_envs=4)
    
    # Choose algorithm
    if algorithm == 'PPO':
        model = PPO(
            "CnnPolicy", 
            env, 
            verbose=1,
            tensorboard_log="./celeste_tensorboard/",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
        )
    elif algorithm == 'DQN':
        # Note: DQN requires some adjustments for continuous observations
        model = DQN(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log="./celeste_tensorboard/",
            learning_rate=1e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            policy_kwargs=policy_kwargs
        )
    elif algorithm == 'A2C':
        model = A2C(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log="./celeste_tensorboard/",
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            vf_coef=0.25,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Define a callback for evaluation
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    
    # Train the agent
    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback
    )
    
    # Save the trained model
    model.save(f"./RL/celeste_{algorithm.lower()}_model")
    print(f"Model saved as ./RL/celeste_{algorithm.lower()}_model")
    
    return model

def test_agent(model_path, algorithm='PPO', episodes=5):
    """
    Test the trained agent
    """
    # Load environment
    env = CelesteGymEnv()
    
    # Load the trained model
    if algorithm == 'PPO':
        model = PPO.load(model_path)
    elif algorithm == 'DQN':
        model = DQN.load(model_path)
    elif algorithm == 'A2C':
        model = A2C.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    print(f"Testing trained {algorithm} agent...")
    
    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        step_count = 0
        
        print(f"\n--- Test Episode {episode + 1} ---")
        
        while True:
            # Predict action using the trained model
            action, _states = model.predict(obs, deterministic=True)
            
            # Take action in environment
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # Print player position occasionally
            if step_count % 50 == 0:
                print(f"Step {step_count}: Player at ({info['player_x']:.1f}, {info['player_y']:.1f}), "
                      f"action: {info['action']}, reward: {reward:.2f}")
            
            # Check if episode is done
            if done or truncated or step_count > 1000:
                print(f"Episode {episode + 1} finished: Total reward = {total_reward:.2f}, Steps = {step_count}")
                print(f"Final position: ({info['player_x']:.1f}, {info['player_y']:.1f})")
                break

def main():
    print("Celeste Reinforcement Learning with Stable Baselines3")
    print("Available algorithms: PPO (recommended), DQN, A2C")
    print("1. Train agent")
    print("2. Test agent")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        algorithm = input("Enter algorithm (PPO/DQN/A2C) [default PPO]: ") or "PPO"
        timesteps = int(input("Enter number of timesteps [default 100000]: ") or "100000")
        
        try:
            model = train_agent_sb3(algorithm=algorithm, total_timesteps=timesteps)
            print("Training completed!")
        except Exception as e:
            print(f"Error during training: {e}")
    
    elif choice == "2":
        algorithm = input("Enter algorithm that was used for training (PPO/DQN/A2C) [default PPO]: ") or "PPO"
        model_path = input("Enter path to trained model [default './RL/celeste_ppo_model']: ") or "./RL/celeste_ppo_model"
        episodes = int(input("Enter number of test episodes [default 5]: ") or "5")
        
        try:
            test_agent(model_path, algorithm=algorithm, episodes=episodes)
        except Exception as e:
            print(f"Error during testing: {e}")
    
    else:
        print("Invalid choice. Please run again and enter 1 or 2.")

if __name__ == "__main__":
    main()