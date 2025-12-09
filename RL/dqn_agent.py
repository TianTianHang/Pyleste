import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os

class DQN(nn.Module):
    """Deep Q-Network model for Celeste."""
    
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        # Convolutional layers to process the game map
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate the size of the flattened features after convolutions
        conv_out_size = self._get_conv_out_size(input_shape)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_actions)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def _get_conv_out_size(self, shape):
        """Calculate the output size of the convolutional layers."""
        
        dummy_input = torch.zeros(1, *shape)
        conv_out = self.conv1(dummy_input)
        conv_out = self.conv2(conv_out)
        conv_out = self.conv3(conv_out)
        return int(np.prod(conv_out.size()))
    
    def forward(self, x):
        """Forward pass of the network."""
        # Ensure input is the right type and normalize
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        # # Input shape is [batch_size, 16, 16, 3], need to transpose to [batch_size, 3, 16, 16]
        # x = x.permute(0, 3, 1, 2) if len(x.shape) == 4 else x.permute(0, 3, 1, 2)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class DQNAgent:
    """DQN Agent for learning to play Celeste."""
    
    def __init__(self, state_shape, n_actions, lr=1e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=10000):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network = DQN(state_shape, n_actions).to(self.device)
        self.target_network = DQN(state_shape, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Update target network
        self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose an action using epsilon-greedy policy."""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.n_actions)
        
        state_tensor = torch.FloatTensor(np.expand_dims(state, axis=0)).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        """Train the model on a batch of experiences."""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        states_tensor = torch.FloatTensor(states).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        
        current_q_values = self.q_network(states_tensor).gather(1, torch.LongTensor(actions).to(self.device).unsqueeze(1))
        next_q_values = self.target_network(next_states_tensor).max(1)[0].detach()
        target_q_values = torch.FloatTensor(rewards).to(self.device) + (self.gamma * next_q_values * (1 - torch.FloatTensor(dones).to(self.device)))
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """Save the model to a file."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath):
        """Load the model from a file."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Model loaded from {filepath}")
        else:
            print(f"No model found at {filepath}, starting from scratch")