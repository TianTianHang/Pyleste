import numpy as np
from PICO8 import PICO8
from Carts.Celeste import Celeste
import CelesteUtils as utils
import gymnasium as gym
from gymnasium import spaces


class CelesteGymEnv(gym.Env):
    """
    Gym environment wrapper for Celeste game to work with Stable Baselines3.
    The goal is to make the player reach y < 0 (go up as much as possible).
    """
    
    def __init__(self):
        super(CelesteGymEnv, self).__init__()
        
        # Define action space as Discrete (10 different actions)
        # 0: no input, 1: right, 2: left, 3: right+jump, 4: left+jump, 5: jump
        # 6: right+dash, 7: left+dash, 8: right+up+dash, 9: left+up+dash
        self.action_space = spaces.Discrete(10)
        
        # Define observation space as 3-channel 16x16 image (map + player positions)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, 16, 16), dtype=np.float32
        )
        
        self.p8 = PICO8(Celeste)
        self.reset()
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.p8 = PICO8(Celeste)
        utils.load_room(self.p8, 0)
        utils.skip_player_spawn(self.p8)
        
        # Store initial player position for reward calculation
        self.initial_player_y = self.get_player_position()[1]
        self.previous_player_y = self.initial_player_y
        
        return self.get_state(), {}  # Return state and info dict
    
    def get_player_position(self):
        """Get current player position (x, y)."""
        player = self.p8.game.get_player()
        if player:
            return int(player.x / 8), int(player.y / 8)  # Convert to grid coordinates
        return 8, 15  # Default position if no player object exists
    
    def get_map_state(self):
        """Get the current map state as a 16x16 grid."""
        map_state = np.zeros((16, 16), dtype=np.uint8)
        
        for tx in range(16):
            for ty in range(16):
                tile = self.p8.mget(self.p8.game.room.x * 16 + tx, self.p8.game.room.y * 16 + ty)
                
                # Convert tile to category
                if self.p8.fget(tile, 0):  # Solid terrain
                    map_state[ty, tx] = 1
                elif tile == 17:  # Up spike
                    map_state[ty, tx] = 2
                elif tile == 27:  # Down spike
                    map_state[ty, tx] = 3
                elif tile == 43:  # Right spike
                    map_state[ty, tx] = 4
                elif tile == 59:  # Left spike
                    map_state[ty, tx] = 5
                elif tile == 18:  # Spring
                    map_state[ty, tx] = 6
                elif tile == 23:  # Crumble block
                    map_state[ty, tx] = 7
                else:
                    map_state[ty, tx] = 0  # Empty
        
        return map_state
    
    def get_state(self):
        """Get the current state of the environment."""
        map_state = self.get_map_state()
        player_x, player_y = self.get_player_position()
        
        # Create a 3-channel state: map, player x position, player y position
        state = np.zeros((3, 16, 16), dtype=np.float32)
        state[0, :, :] = map_state / 7.0  # Normalize map values
        state[1, :, :] = player_x / 15.0  # Normalize to [0, 1]
        state[2, :, :] = player_y / 15.0  # Normalize to [0, 1]
        
        return state
    
    def step(self, action):
        """Execute an action in the environment."""
        # Map action to inputs
        if action == 0:  # No input
            l, r, u, d, z, x = False, False, False, False, False, False
        elif action == 1:  # Right
            l, r, u, d, z, x = False, True, False, False, False, False
        elif action == 2:  # Left
            l, r, u, d, z, x = True, False, False, False, False, False
        elif action == 3:  # Right + Jump
            l, r, u, d, z, x = False, True, False, False, True, False
        elif action == 4:  # Left + Jump
            l, r, u, d, z, x = True, False, False, False, True, False
        elif action == 5:  # Jump
            l, r, u, d, z, x = False, False, False, False, True, False
        elif action == 6:  # Right + dash
            l, r, u, d, z, x = False, True, False, False, False, True
        elif action == 7:  # Left + dash
            l, r, u, d, z, x = True, False, False, False, False, True
        elif action == 8:  # Right + up + dash
            l, r, u, d, z, x = False, True, True, False, False, True
        elif action == 9:  # Left + up + dash
            l, r, u, d, z, x = True, False, True, False, False, True
        else:
            # Default to no input for invalid actions
            l, r, u, d, z, x = False, False, False, False, False, False
        
        # Set inputs for the game
        self.p8.set_inputs(l=l, r=r, u=u, d=d, z=z, x=x)
        
        # Step the game
        self.p8.step()
        
        # Get new state
        next_state = self.get_state()
        
        # Calculate reward
        current_player_y = self.get_player_position()[1]
        reward = self.calculate_reward(current_player_y)
        
        # Check if episode is done
        done = self.is_done()
        truncated = self.is_truncated()  # For Gym compatibility
        
        # Get player info for debugging
        info = {
            'player_x': self.get_player_position()[0],
            'player_y': current_player_y,
            'action': action
        }
        
        # Update previous position for next reward calculation
        self.previous_player_y = current_player_y
        
        return next_state, reward, done, truncated, info
    
    def calculate_reward(self, current_player_y):
        """Calculate reward based on player movement."""
        # Reward for moving upward (goal is to reach y < 0)
        height_reward = (self.previous_player_y - current_player_y) * 10  # Moving up is good
        
        # Penalty for moving down
        if current_player_y > self.previous_player_y:
            height_reward -= 2  # Reduce penalty to avoid stuck behavior
        
        # Large penalty for dying (falling off the screen)
        if current_player_y > 16:  # Player fell off the bottom
            height_reward -= 100
        
        # Bonus for reaching higher positions
        if current_player_y < 5:  # High up in the level
            height_reward += 50
        elif current_player_y < 10:
            height_reward += 20
        elif current_player_y < 15:
            height_reward += 5
        
        # Small bonus for staying alive
        height_reward += 0.1
        
        return height_reward
    
    def is_done(self):
        """Check if the episode is finished (terminal state)."""
        current_player_y = self.get_player_position()[1]
        player = self.p8.game.get_player()
        
        # Done conditions:
        # 1. Player reached top (y < 0) - level completed (this is good!)
        # 2. Player object is None (killed by spikes, etc.)
        return current_player_y < 0 or player is None
    
    def is_truncated(self):
        """Check if the episode is truncated (time limit)."""
        current_player_y = self.get_player_position()[1]
        # Truncate if player falls too far down
        return current_player_y > 16
    
    def render(self, mode='human'):
        """Render the current state of the game."""
        print(self.p8.game)