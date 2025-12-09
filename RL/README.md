# Celeste Reinforcement Learning with Stable Baselines3

This directory contains a reinforcement learning implementation for the Celeste game environment using Stable Baselines3 (SB3).

## Overview

The reinforcement learning agent is designed to play the Celeste platformer game with the goal of reaching the top of the screen (y < 0). The agent learns to navigate platforms, avoid spikes, and climb as high as possible using a combination of movement and dash abilities.

## Files

- `celeste_env.py`: Gym environment wrapper for the Celeste game that provides the SB3 interface
- `train.py`: Main training and testing script using Stable Baselines3
- `README.md`: This file

## Requirements

- Python 3.7+
- Stable Baselines3 (sb3)
- Gymnasium or Gym
- PyTorch
- NumPy

Install requirements:
```bash
pip install stable-baselines3 gymnasium torch numpy
```

## Usage

### Training the Agent

```bash
cd RL
python train.py
# Choose option 1 to train the agent
```

Available algorithms:
- PPO (Proximal Policy Optimization) - Recommended
- DQN (Deep Q-Network)
- A2C (Advantage Actor-Critic)

### Testing the Agent

```bash
cd RL
python train.py
# Choose option 2 to test the agent
```

## Architecture

### Environment

- **State Space**: 3x16x16 tensor representing:
  - Channel 0: Game map (terrain, spikes, etc.), normalized
  - Channel 1: Normalized player X position
  - Channel 2: Normalized player Y position

- **Action Space**: 10 discrete actions:
  - 0: No input
  - 1: Move right
  - 2: Move left
  - 3: Move right + jump
  - 4: Move left + jump
  - 5: Jump only
  - 6: Move right + dash
  - 7: Move left + dash
  - 8: Move right + up + dash
  - 9: Move left + up + dash

- **Reward Function**:
  - Positive reward for moving upward
  - Small penalty for moving downward
  - Large penalty for dying (falling off screen)
  - Bonus for reaching high positions
  - Small reward for staying alive

### Algorithms

The implementation supports multiple state-of-the-art RL algorithms from Stable Baselines3:

#### PPO (Proximal Policy Optimization)
- Most robust and recommended algorithm
- Good balance of sample efficiency and stability
- Uses actor-critic architecture

#### DQN (Deep Q-Network)
- Value-based approach
- Good for discrete action spaces
- Uses experience replay and target networks

#### A2C (Advantage Actor-Critic)
- Synchronous version of A3C
- Simpler than PPO but effective
- Good baseline algorithm

## Training Details

- Uses CNN policy networks to process visual input
- Implements proper Gym interface for SB3 compatibility
- Includes evaluation callbacks for monitoring progress
- Supports TensorBoard logging for visualization

## Results

The agent learns to:
- Navigate platforms efficiently
- Avoid dangerous spikes
- Use jumps and dashes strategically
- Climb as high as possible in the level

## Future Improvements

- Implement more complex level generation
- Add curriculum learning with increasing difficulty
- Multi-level training across different Celeste maps
- Hyperparameter optimization
- Advanced algorithms like SAC or TD3 for continuous control