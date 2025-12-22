from collections import defaultdict, deque
import random
import numpy as np
from PICO8 import PICO8
from Carts.Celeste import Celeste
import CelesteUtils as utils
import gymnasium as gym
from gymnasium import spaces

ACTIONS = {
    0: (False, False, False, False, False, False),  # no input
    1: (False, True, False, False, False, False),  # right
    2: (True, False, False, False, False, False),  # left
    3: (False, True, False, False, True, False),  # right + jump
    4: (True, False, False, False, True, False),  # left + jump
    5: (False, True, False, False, False, True),  # right + dash
    6: (True, False, False, False, False, True),  # left + dash
    7: (False, True, True, False, False, True),  # right + up + dash
    8: (True, False, True, False, False, True),  # left + up + dash
    9: (False, False, True, False, False, True),  # up + dash
}


class CelesteGymEnv(gym.Env):
    """
    Gym environment for Celeste with Curriculum Learning (stage-wise goals).
    Each goal is a separate stage. Reaching a goal ends the episode successfully.
    Death or falling also ends the episode (failure).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self, custom_room=None, goal=None, level=0, render_mode=None, max_step=2000,randomize_start_position=False
    ):
        super(CelesteGymEnv, self).__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(10)
        self.custom_room = custom_room
        self.level = level
        self.goal = goal
        self.randomize_start_position=randomize_start_position
        self.max_step = max_step
        self.observation_space = spaces.Dict(
            {
                "map": spaces.Box(low=0, high=1, shape=(2, 16, 16), dtype=np.float32),
                "player_pos": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
                "velocity": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                "on_ground": spaces.Discrete(2),
                "dashes_left": spaces.Discrete(3),
                "goal_rel": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            }
        )

        self.p8 = PICO8(Celeste)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)  # 或使用 self.np_random（Gym 推荐）
        if self.custom_room:
            utils.replace_room(self.p8, self.level, self.custom_room)
        if self.level is not None:
            utils.load_room(self.p8, self.level)
        else:
            utils.load_room(self.p8, random.choice([0,2,3,4,5,6,7,8]))
        utils.skip_player_spawn(self.p8)
        player = self.p8.game.get_player()
        if self.goal==None and self.randomize_start_position:
            random_pos, random_goal=self.sample_player_pos_and_goal()
            self.goal=random_goal
            self.set_player_pos(*random_pos)
        elif self.goal==None and not self.randomize_start_position:
            _, random_goal=self.sample_player_pos_and_goal(pos=(player.x,player.y))
            self.goal=random_goal
        elif self.goal and self.randomize_start_position:
            random_pos, random_goal=self.sample_player_pos_and_goal()
            self.set_player_pos(*random_pos)
            
        self.p8.set_inputs()
        self.current_step = 0
        self.visit_count = defaultdict(int)
        pos = self.get_player_position()
        self.previous_player_x, self.previous_player_y = pos
        self.initial_player_x, self.initial_player_y = pos
        # Set previous distance to current goal (if any)
        current_goal = self.goal
        if self.goal is not None:
            self.previous_distance = np.sqrt(
                (pos[0] - current_goal[0]) ** 2 + (pos[1] - current_goal[1]) ** 2
            )
        else:
            self.previous_distance = float("inf")
        #print(f"x: {pos[0]} y: {pos[1]}")
        return self.get_state(), {} 

    def set_player_pos(self, x, y):
        player = self.p8.game.get_player()
        player.x = x
        player.y = y
    def sample_player_pos_and_goal(self,pos=None, min_dist=16, max_dist=256, max_attempts=100):
        """
        返回:
            (player_x, player_y), (goal_x, goal_y)
        """
        map_state = self.get_map_state()  # shape (16, 16), dtype uint8 or float
        
        # 找出所有可站立图块正上方的空气格子坐标 (ty, tx) -> 注意：map_state[ty, tx]
        walkable_tiles_above = []
        for ty in range(16):
            for tx in range(16):
                if map_state[ty, tx] == 1 or map_state[ty, tx] == 7:  # 可站立图块
                    if ty > 0 and map_state[ty - 1, tx] == 0:  # 正上方为空气
                        world_x = tx * 8 + 4
                        world_y = (ty - 1) * 8 + 4  # 在可站立图块正上方
                        walkable_tiles_above.append((world_x, world_y))
        if pos:
            player_pos=pos
        else:
            # 随机选择玩家起始位置
            player_pos = random.choice(walkable_tiles_above)
        # 找出所有空气格子的坐标
        air_tiles = []
        for ty in range(int(player_pos[1]//8)):
            for tx in range(16):
                if map_state[ty, tx] == 0:  # 空气
                    world_x = tx * 8 + 4
                    world_y = ty * 8 + 4
                    air_tiles.append((world_x, world_y))
        
        if len(air_tiles) < 1:
            # 兜底：返回默认位置
            return player_pos,player_pos
        
       
        
        for _ in range(max_attempts):
            # 随机选择一个空气格作为目标
            goal_pos = random.choice(air_tiles)
            
            dist = np.sqrt((player_pos[0] - goal_pos[0])**2 + (player_pos[1] - goal_pos[1])**2)
            if dist >= min_dist and dist <=max_dist:
                return player_pos, goal_pos
        
        return player_pos, goal_pos
    def get_player_state(self):
        player = self.p8.game.get_player()
        if isinstance(player, Celeste.player):
            return {
                "position": (player.x, player.y),
                "speed": (float(player.spd.x), float(player.spd.y)),
                "on_ground": bool(player.is_solid(0, 1)),  # 是否发生碰撞
                "dashes_left": int(player.djump),  # 冲刺剩余次数
            }
        return {
            "position": (self.previous_player_x, self.previous_player_y),
            "speed": (0.0, 0.0),
            "on_ground": True,
            "dashes_left": 0,
        }

    def get_player_position(self):
        player = self.p8.game.get_player()
        if isinstance(player, Celeste.player):
            return player.x, player.y
        return self.previous_player_x, self.previous_player_y

    def get_map_state(self):
        map_state = np.zeros((16, 16), dtype=np.uint8)
        for tx in range(16):
            for ty in range(16):
                tile = self.p8.mget(
                    self.p8.game.room.x * 16 + tx, self.p8.game.room.y * 16 + ty
                )
                if tile in (32,11,12,23):  # wall or ground
                    map_state[ty, tx] = 1
                elif tile == 64:
                    map_state[ty, tx] = 2
                elif tile == 17:  # up spike
                    map_state[ty, tx] = 3
                elif tile == 27:  # down spike
                    map_state[ty, tx] = 4
                elif tile == 43:  # right spike
                    map_state[ty, tx] = 5
                elif tile == 59:  # leftspike
                    map_state[ty, tx] = 6
                elif tile == 18:  # spring
                    map_state[ty, tx] = 7
                elif tile == 26:  # fruit
                    map_state[ty, tx] = 8
                else:  # air
                    map_state[ty, tx] = 0
        return map_state

    def get_state(self):
        map_obs = self.get_map_state()
        p_state = self.get_player_state()
        x, y = p_state["position"]
        vx, vy = p_state["speed"]
        rel_x = self.goal[0] - x
        rel_y = self.goal[1] - y
        # Mark player on map
        tile_x = int(x / 8)
        tile_y = int(y / 8)
        
        expanded_map = np.zeros((2, 16, 16), dtype=np.uint8)
        # 第一通道填充静态地图数据
        expanded_map[0] = map_obs
       
        map_obs = expanded_map.astype(np.float32) / 8.0
        if 0 <= tile_x < 16 and 0 <= tile_y < 16:
            map_obs[1,tile_y, tile_x] = 1  # 注意：[row, col] = [y, x] player
        player_pos_norm = np.array(
            [np.clip(x / 128.0, 0.0, 1.0), np.clip(y / 128.0, 0.0, 1.0)],
            dtype=np.float32,
        )
        goal_rel_norm = goal_rel_norm = np.array(
            [np.clip(rel_x / 128.0, -1.0, 1.0), np.clip(rel_y / 128.0, -1.0, 1.0)],
            dtype=np.float32,
        )
        velocity_norm = np.array(
            [np.clip(vx / 5.0, -1.0, 1.0), np.clip(vy / 5.0, -1.0, 1.0)],
            dtype=np.float32,
        )

        return {
            "map": map_obs,
            "player_pos": player_pos_norm,
            "velocity": velocity_norm,
            "on_ground": int(bool(p_state["on_ground"])),
            "dashes_left": int(np.clip(p_state["dashes_left"], 0, 2)),
            "goal_rel": goal_rel_norm,
        }

    def calculate_reward(self):
        current_x, current_y = self.get_player_position()
        player = self.p8.game.get_player()
        key = (int(current_x // 8), int(current_y // 8))
        
        if current_y > 128 or not isinstance(player, Celeste.player):
            return -10.0

        reward = 0.0
        if self.visit_count[key] == 0:
            reward += 0.5  # 首次进入格子给奖励 0:0.5
        elif self.visit_count[key] >5:
            reward -= 0.01
        self.visit_count[key] += 1

        # 1. 目标导向奖励（如果设置了 goal）
        if self.goal is not None:
            current_dist = np.sqrt(
                (current_x - self.goal[0]) ** 2 + (current_y - self.goal[1]) ** 2
            )
            if current_dist < 1.0:
                reward += 500
                print('success')
            # 鼓励靠近目标：每靠近 1 像素 ≈ +0.1
            reward += (self.previous_distance - current_dist) * 0.1
            self.previous_distance = current_dist
        # reward+=(self.initial_player_y-current_y)*0.1 # level 1
        # 5. 存活奖励（鼓励持续探索）
        reward += 0.01

        #7. 时间惩罚（防止拖延）
        if self.current_step > 200:  # 更早开始惩罚
            reward -= 0.01  # 轻微惩罚，避免过激

        return reward

    def is_terminated(self):
        """Return (terminated, success)"""
        player = self.p8.game.get_player()
        current_x, current_y = self.get_player_position()

        # Death or out of bounds
        if current_y > 128 or not isinstance(player, Celeste.player):
            return True, False

        # Check if reached current goal
        current_goal = self.goal
        if current_goal is not None:
            dist = np.sqrt(
                (current_x - current_goal[0]) ** 2 + (current_y - current_goal[1]) ** 2
            )
            if dist <= 1.0:
                return True, True  # Success!

        return False, False

    def is_truncated(self):
        return self.current_step > self.max_step

    def step(self, action):
        action = int(action)
        l, r, u, d, z, x = ACTIONS.get(
            action, (False, False, False, False, False, False)
        )
        self.current_step += 1
        self.p8.set_inputs(l=l, r=r, u=u, d=d, z=z, x=x)
        self.p8.step()

        next_state = self.get_state()
        reward = self.calculate_reward()

        terminated, success = self.is_terminated()
        truncated = self.is_truncated()
        current_x, current_y = self.get_player_position()
        self.previous_player_x, self.previous_player_y = current_x, current_y
        info = {
            "player_x": current_x,
            "player_y": current_y,
            "action": action,
            "episode_length": self.current_step,
            "success": success,  # Crucial for curriculum learning
        }
        #print(f"Action: {action}, Reward: {round(reward, 4)}")
        return next_state, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(self.p8.game)
        elif self.render_mode == "rgb_array":
            # 可选：返回可视化图像（用于视频记录）
            map_img = self.get_map_state().astype(np.float32) / 8.0
            x, y = self.get_player_position["position"]
            tile_x = int(x / 8)
            tile_y = int(y / 8)
            if 0 <= tile_x < 16 and 0 <= tile_y < 16:
                map_img[tile_y, tile_x] = 1  # 注意：[row, col] = [y, x] player
            rgb = np.stack([map_img] * 3, axis=-1)  # 灰度转 RGB
            return (rgb * 255).astype(np.uint8)
