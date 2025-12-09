# Celeste Reinforcement Learning

This directory contains a reinforcement learning implementation for the Celeste game environment using Deep Q-Network (DQN).

## Overview

The reinforcement learning agent is designed to play the Celeste platformer game with the goal of reaching the top of the screen (y < 0). The agent learns to navigate platforms, avoid spikes, and climb as high as possible using a combination of movement and dash abilities.

## Files

- `celeste_env.py`: Environment wrapper for the Celeste game that provides the RL interface
- `dqn_agent.py`: Implementation of the DQN agent with neural network architecture
- `train.py`: Main training and testing script
- `README.md`: This file

## Requirements

- Python 3.7+
- PyTorch
- NumPy

## Usage

### Training the Agent

```bash
cd RL
python train.py
# Choose option 1 to train the agent
```

### Testing the Agent

```bash
cd RL
python train.py
# Choose option 2 to test the agent
```

### Custom Training

You can also run training directly:

```bash
python train.py
```

## Architecture

### Environment

- **State Space**: 16x16x3 tensor representing:
  - Channel 0: Game map (terrain, spikes, etc.)
  - Channel 1: Normalized player X position
  - Channel 2: Normalized player Y position

- **Action Space**: 6 discrete actions:
  - 0: No input
  - 1: Move right
  - 2: Move left
  - 3: Move right + jump
  - 4: Move left + jump
  - 5: Jump only

- **Reward Function**:
  - Positive reward for moving upward
  - Negative reward for moving downward
  - Large penalty for dying (falling off screen)
  - Bonus for reaching high positions

### Neural Network

The DQN uses:
- Convolutional layers to process the game map
- Fully connected layers for decision making
- Experience replay for stable training
- Target network for stable Q-value estimation

## Training Details

- Epsilon-greedy exploration strategy (starts at 1.0, decays to 0.01)
- Adam optimizer with learning rate 1e-4
- Discount factor (gamma) of 0.99
- Batch size of 32 for experience replay
- Target network updates every 10 episodes

## Results

The agent learns to:
- Navigate platforms efficiently
- Avoid dangerous spikes
- Use jumps strategically
- Climb as high as possible in the level

## Future Improvements

- Implement PPO (Proximal Policy Optimization) for continuous control
- Add more complex level generation
- Implement curriculum learning with increasing difficulty
- Multi-level training across different Celeste maps