from collections import deque
import numpy as np
from PICO8 import PICO8
from Carts.Celeste import Celeste
import CelesteUtils as utils
import gymnasium as gym
from gymnasium import spaces

ACTIONS = {
    0: (False, False, False, False, False, False),  # no input
    1: (False, True,  False, False, False, False),  # right
    2: (True,  False, False, False, False, False),  # left
    3: (False, True,  False, False, True,  False),  # right + jump
    4: (True,  False, False, False, True,  False),  # left + jump
    5: (False, True,  False, False, False, True ),  # right + dash
    6: (True,  False, False, False, False, True ),  # left + dash
    7: (False, True,  True,  False, False, True ),  # right + up + dash
    8: (True,  False, True,  False, False, True ),  # left + up + dash
    9: (False,  False, True,  False, False, True ),  # up + dash
}

class CelesteGymEnv(gym.Env):
    """
    Gym environment for Celeste with Curriculum Learning (stage-wise goals).
    Each goal is a separate stage. Reaching a goal ends the episode successfully.
    Death or falling also ends the episode (failure).
    """

    def __init__(self, max_steps=64, done_threshold=1, custom_room=None, goal=None, render_mode=None):
        super(CelesteGymEnv, self).__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(10)
        self.custom_room = custom_room
        self.goal = goal 

        self.observation_space = spaces.Dict({
            "map": spaces.Box(low=0, high=1, shape=(16, 16), dtype=np.float32),
            "player_pos": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "velocity": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "on_ground": spaces.Discrete(2),
            "dashes_left": spaces.Discrete(3),
            "is_jumping": spaces.Discrete(2),
        })

        self.max_steps = max_steps
        self.done_threshold = done_threshold
        self.p8 = PICO8(Celeste)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.custom_room:
            utils.replace_room(self.p8, 0, self.custom_room)
        utils.load_room(self.p8, 0)


        utils.skip_player_spawn(self.p8)

        self.p8.set_inputs()
        self.current_step = 0

        pos = self.get_player_position()
        self.previous_player_x, self.previous_player_y = pos
        self.initial_player_x, self.initial_player_y = pos

        self.done_counter = 0
        self.previous_action = 0
        self.action_history = []
        self.visit_count = np.zeros((16, 16), dtype=int)

        # New: Track visitation for exploration bonus
        self.visited_tiles = set()

        # Set previous distance to current goal (if any)
        current_goal = self.goal
        if current_goal is not None:
            self.previous_distance = np.sqrt(
                (pos[0] - current_goal[0])**2 + (pos[1] - current_goal[1])**2
            )
        else:
            self.previous_distance = float('inf')

        return self.get_state(), {}

   

    def get_player_state(self):
        player = self.p8.game.get_player()
        if isinstance(player, Celeste.player):
            return {
                "position": (int(player.x / 8), int(player.y / 8)),
                "speed": (float(player.spd.x), float(player.spd.y)),
                "on_ground": bool(player.is_solid(0, 1)),
                "dashes_left": int(player.djump),
                "is_jumping": bool(player.jbuffer > 0),
            }
        return {
            "position": (self.previous_player_x, self.previous_player_y),
            "speed": (0.0, 0.0),
            "on_ground": False,
            "dashes_left": 0,
            "is_jumping": False,
        }

    def get_player_position(self):
        player = self.p8.game.get_player()
        if isinstance(player, Celeste.player):
            return int(player.x / 8), int(player.y / 8)
        return self.previous_player_x, self.previous_player_y

    def get_map_state(self):
        map_state = np.zeros((16, 16), dtype=np.uint8)
        for tx in range(16):
            for ty in range(16):
                tile = self.p8.mget(self.p8.game.room.x * 16 + tx, self.p8.game.room.y * 16 + ty)
                if tile == 32:
                    map_state[ty, tx] = 1
                elif tile == 17:
                    map_state[ty, tx] = 2
                elif tile == 27:
                    map_state[ty, tx] = 3
                elif tile == 43:
                    map_state[ty, tx] = 4
                elif tile == 59:
                    map_state[ty, tx] = 5
                elif tile == 18:
                    map_state[ty, tx] = 6
                elif tile == 23:
                    map_state[ty, tx] = 7
                else:
                    map_state[ty, tx] = 0
        return map_state

    def get_state(self):
        map_obs = self.get_map_state()
        p_state = self.get_player_state()
        x, y = p_state["position"]
        vx, vy = p_state["speed"]

        # Mark player on map
        if 0 <= y < 16 and 0 <= x < 16:
            map_obs[y, x] = 8
        map_obs = map_obs.astype(np.float32) / 8.0

        player_pos_norm = np.array([np.clip(x / 15.0, 0.0, 1.0), np.clip(y / 15.0, 0.0, 1.0)], dtype=np.float32)
        velocity_norm = np.array([np.clip(vx / 2.0, -1.0, 1.0), np.clip(vy / 4.0, -1.0, 1.0)], dtype=np.float32)

        return {
            "map": map_obs,
            "player_pos": player_pos_norm,
            "velocity": velocity_norm,
            "on_ground": int(bool(p_state["on_ground"])),
            "dashes_left": int(np.clip(p_state["dashes_left"], 0, 2)),
            "is_jumping": int(bool(p_state["is_jumping"])),
        }

    def calculate_reward(self):
        player = self.p8.game.get_player()
        if not isinstance(player, Celeste.player):
            return -10.0

        p_state = self.get_player_state()
        current_x, current_y = p_state["position"]
        vx, vy = p_state["speed"]

        reward = 0.0

        # Exploration bonus
        current_tile = (current_x, current_y)
        if current_tile not in self.visited_tiles:
            self.visited_tiles.add(current_tile)
            reward += 0.5  # Significant bonus for exploring new areas
        else:
            # Small penalty for revisiting familiar areas to encourage exploration
            reward -= 0.02  # Reduced penalty for revisiting

        # Progress toward current goal (RE-ENABLED)
        current_goal = self.goal
        if current_goal is not None:
            goal_x, goal_y = current_goal
            distance = np.sqrt((current_x - goal_x)**2 + (current_y - goal_y)**2)

            # Calculate progress since last step
            progress = self.previous_distance - distance

            # Reward progress towards goal
            if progress > 0:
                reward += progress * 2.0  # Encourage forward progress towards goal
            elif progress < 0:
                reward += progress * 1.0  # Small penalty for moving away from goal

            # Bonus when close to goal
            if distance <= 2:
                reward += (3 - distance) * 0.5  # Bonus for getting close

            # Major reward for reaching goal
            if distance == 0:
                reward += 50  # Success!

            self.previous_distance = distance
        else:
            # Fallback: encourage upward movement (Celeste is typically an upward climbing game)
            if current_y < self.previous_player_y:
                reward += (self.previous_player_y - current_y) * 2.0  # Stronger incentive for upward movement
            elif current_y > self.previous_player_y:
                reward -= (current_y - self.previous_player_y) * 1.0  # Penalty for downward movement

        # Spikes proximity penalty (strengthened)
        map_state = self.get_map_state()  # shape (16, 16), dtype uint8
        # Define spike tile IDs based on the get_map_state() function
        # 2: spikes, 3: upward spikes, 4: downward spikes
        spike_mask = np.isin(map_state, [2, 3, 4])
        spike_positions = np.argwhere(spike_mask)  # Get all spike coordinates (y, x)

        if spike_positions.size > 0:
            # Calculate distance to each spike
            distances = np.sqrt(
                (spike_positions[:, 1] - current_x) ** 2 +  # x direction
                (spike_positions[:, 0] - current_y) ** 2    # y direction
            )
            min_dist = np.min(distances) if len(distances) > 0 else float('inf')

            # Closer to spikes means higher penalty
            if min_dist < 3.0:
                # Strengthened penalty using inverse relationship
                spike_penalty = -max(0, (3.0 - min_dist) * 1.0)  # Max penalty of -3.0 when touching spikes
                reward += spike_penalty

        # Time alive bonus (small but consistent)
        reward += 0.01

        # Momentum reward for smooth movement (encourages consistent velocity)
        if abs(vx) > 0.5:  # Moving horizontally
            reward += 0.05  # Small bonus for movement

        # Jumping reward (for dynamic movement)
        if p_state["is_jumping"]:
            reward += 0.02  # Small bonus for jumping

        # Ground contact reward (to prevent falling)
        if p_state["on_ground"]:
            reward += 0.03  # Small bonus for staying grounded

        # Dash utilization reward
        if p_state["dashes_left"] < 2:
            reward += 0.1  # Bonus for using dashes effectively

        return reward

    def is_terminated(self):
        """Return (terminated, success)"""
        player = self.p8.game.get_player()
        current_x, current_y = self.get_player_position()

        # Death or out of bounds
        if current_y < 0 or player is None:
            self.done_counter += 1
            utils.skip_player_spawn(self.p8)
            if self.done_counter >= self.done_threshold:
                return True, False
            return False, False

        self.done_counter = 0

        # Check if reached current goal
        current_goal = self.goal
        if current_goal is not None:
            dist = np.sqrt((current_x - current_goal[0])**2 + (current_y - current_goal[1])**2)
            if dist == 0.0:
                return True, True  # Success!

        return False, False

    def is_truncated(self):
        return self.current_step >= self.max_steps

    def step(self, action):
        action = int(action)
        l, r, u, d, z, x = ACTIONS.get(action, (False, False, False, False, False, False))
        self.current_step += 1
        self.p8.set_inputs(l=l, r=r, u=u, d=d, z=z, x=x)
        self.p8.step()

        next_state = self.get_state()
        reward = self.calculate_reward()

        terminated, success = self.is_terminated()
        truncated = self.is_truncated()

        # Update history
        self.action_history.append(action)
        if len(self.action_history) > 5:
            self.action_history.pop(0)

        # Check for repeated action sequences and apply reduced penalty
        reward += self.check_repeated_sequences() * 0.1  # Reduce penalty severity

        # Reduce penalty for repeating the same action
        if action == self.previous_action:
            reward -= 0.1  # Much smaller penalty than before
        self.previous_action = action


        current_x, current_y = self.get_player_position()
        info = {
            'player_x': current_x,
            'player_y': current_y,
            'action': action,
            'episode_length': self.current_step,
            'success': success,  # Crucial for curriculum learning
        }

        self.previous_player_x, self.previous_player_y = current_x, current_y
        return next_state, reward, terminated, truncated, info


    def render(self, mode='human'):
        print(self.p8.game)
    def check_repeated_sequences(self):
        """Check for repeated action sequences and return penalty."""
        history = self.action_history
        n = len(history)

        if n < 4:
            return 0

        total_penalty = 0

        # Check for length-2 or length-3 sequence repetitions (only if recent sequence appeared before)
        for seq_len in (2, 3):
            if n < 2 * seq_len:
                continue

            recent_seq = history[-seq_len:]
            earlier_part = history[:-seq_len]

            # Check if recent_seq appeared in earlier_part (at least once)
            found = False
            for i in range(len(earlier_part) - seq_len + 1):
                if earlier_part[i:i + seq_len] == recent_seq:
                    found = True
                    break  # Found once is enough, avoid duplicate penalties

            if found:
                total_penalty -= seq_len * 0.5  # Reduced penalty (length 2 → -1, length 3 → -1.5)

        # Check for simple repetition patterns in recent 3 actions
        recent_actions = history[-3:]  # Safe: n >= 4, so we have at least 3
        unique_actions = set(recent_actions)

        if len(unique_actions) == 1:
            # [A, A, A] - all same actions
            total_penalty -= 0.5  # Much smaller penalty
        elif len(unique_actions) == 2:
            # Only 2 unique actions in 3 actions → one appears ≥2 times
            total_penalty -= 0.3  # Much smaller penalty

        return total_penalty