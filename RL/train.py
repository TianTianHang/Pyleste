from time import sleep
import numpy as np
import torch
import random
from celeste_env import CelesteEnvironment
from dqn_agent import DQNAgent
import os

def train_agent(episodes=1000):
    """Train the DQN agent to play Celeste."""
    
    # Initialize environment and agent
    env = CelesteEnvironment()
    state_shape = env.observation_space
    n_actions = env.action_space
    
    agent = DQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        lr=1e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        memory_size=10000
    )
    
    # Load model if exists
    model_path = "./RL/celeste_dqn_model.pth"
    agent.load(model_path)
    
    # Training loop
    scores = []
    max_score = float('-inf')
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0
        
        while True:
            # Choose action
            action = agent.act(state)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience in memory
            agent.remember(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            total_reward += reward
            step_count += 1
            
            # Train the agent
            if len(agent.memory) > 32:  # Start training when memory has enough experiences
                agent.replay(batch_size=32)
            
            # Check if episode is done
            if done or step_count > 500:  # Cap at 500 steps to prevent very long episodes
                break
        
        scores.append(total_reward)
        
        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_network()
        
        # Save best model
        if total_reward > max_score:
            max_score = total_reward
            agent.save(model_path)
            print(f"New best model saved with score: {total_reward:.2f}")
        
        # Print progress
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        
        if episode % 50 == 0:
            print(f"Episode: {episode}, Score: {total_reward:.2f}, Avg Score: {avg_score:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Steps: {step_count}")
    
    print(f"Training completed! Best score: {max_score:.2f}")

def test_agent(episodes=5,render=False):
    """Test the trained agent."""
    
    # Initialize environment and agent
    env = CelesteEnvironment()
    state_shape = env.observation_space
    n_actions = env.action_space
    
    agent = DQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        lr=1e-4,
        gamma=0.99,
        epsilon=0.01,  # No exploration during testing
        epsilon_decay=1.0,
        epsilon_min=0.01,
        memory_size=10000
    )
    
    # Load trained model
    model_path = "./RL/celeste_dqn_model.pth"
    agent.load(model_path)
    
    print("Testing trained agent...")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0
        
        print(f"\n--- Test Episode {episode + 1} ---")
        
        while True:
            # Always choose best action during testing
            state_tensor = torch.FloatTensor(np.expand_dims(state, axis=0))
            q_values = agent.q_network(state_tensor)
            action = np.argmax(q_values.cpu().data.numpy())
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            if render:
                env.render()
                sleep(0.033)
            # Print player position occasionally
            if step_count % 20 == 0:
                print(f"Step {step_count}: Player at ({info['player_x']}, {info['player_y']}), action: {info['action']}")
            
            # Update state
            state = next_state
            total_reward += reward
            step_count += 1
            
            # Check if episode is done
            if done or step_count > 500:
                print(f"Episode {episode + 1} finished: Total reward = {total_reward:.2f}, Steps = {step_count}")
                print(f"Final position: ({info['player_x']}, {info['player_y']})")
                break

if __name__ == "__main__":
    print("Celeste DQN Training")
    print("1. Train agent")
    print("2. Test agent")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        episodes = int(input("Enter number of episodes (default 1000): ") or "1000")
        train_agent(episodes)
    elif choice == "2":
        episodes = int(input("Enter number of test episodes (default 5): ") or "5")
        render = input("Enter yes or no (default no): ") or "no"
        test_agent(episodes, render=='yes')
    else:
        print("Invalid choice. Please run again and enter 1 or 2.")