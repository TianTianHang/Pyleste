import os
import subprocess
import sys
sys.path.append(os.getcwd())
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO,DQN

# 替换为你的实际模块路径
from RL.celeste_env import ACTIONS, CelesteGymEnv  # 👈 修改这里！

def test_model(
    model_path: str = "./models/celeste_ppo/best/best_model",
    goal=(64.0, 32.0),
    render_mode="human",      # 可选: "human"（打印状态）、"rgb_array"（返回图像）、None（静默）
    n_episodes=5,
    deterministic=True,
    level=0,
    custom_room=None
):
    """
    加载模型并在环境中测试。
    
    Args:
        model_path: 模型路径（不带 .zip）
        goal: 目标位置
        render_mode: 渲染模式
        n_episodes: 测试轮数
        deterministic: 是否使用确定性策略
    """
    # 加载训练好的模型
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found at {model_path}.zip")
    
    model = PPO.load(model_path)
    print(f"✅ Loaded model from {model_path}")

    # 创建环境
    env = CelesteGymEnv(goal=goal, render_mode=render_mode,level=level,randomize_start_position=False,custom_room=custom_room)
    
    success_count = 0
    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        step = 0
        done = False

        print(f"\n--- Episode {ep + 1} ---")
        encoded_actions = []
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            l, r, u, d, z, x = ACTIONS.get(int(action), (False, False, False, False, False, False))

            # 编码为整数
            encoded = l * 1 + r * 2 + u * 4 + d * 8 + z * 16 + x * 32
            encoded_actions.append(encoded)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            # 可选：实时渲染（如果支持）
            if render_mode == "human" and step%10==0:
                env.render()  # 注意：你当前的 render() 只是 print(p8.game)

            done = terminated or truncated

            # 防止无限循环（安全上限）
            if step > 1000:
                print("⚠️ Episode exceeded 1000 steps, forcing termination.")
                break

        episode_rewards.append(total_reward)
        success = info.get("success", False)
        if success:
            success_count += 1
            #subprocess.run(['love','CelesteTAS', 'celeste.p8', '-level', f'{level+1}','-tas', f'[]{','.join(str(a) for a in encoded_actions)}'], shell=True, capture_output=True, text=True)

        print(f"Episode {ep + 1}: Reward = {total_reward:.2f}, Steps = {step}, Success = {success} last pos x: {info['player_x']} y: {info['player_y']}")

    # 输出统计结果
    avg_reward = np.mean(episode_rewards)
    success_rate = success_count / n_episodes
    print("\n" + "="*50)
    print(f"📊 Test Summary over {n_episodes} episodes:")
    print(f"   Average Reward: {avg_reward:.2f}")
    print(f"   Success Rate:   {success_rate * 100:.1f}% ({success_count}/{n_episodes})")
    print("="*50)

if __name__ == "__main__":
    # 配置测试参数
    MODEL_PATH = "models/celeste_ppo/best/best_model"
    #MODEL_PATH = 'RL/finished_models/ppo/level4/best_model'
    GOAL = (108, -1.0) # 与训练时一致
    LEVEL = 0
    CUSTOM_ROOM=None
    test_model(
        model_path=MODEL_PATH,
        goal=GOAL,
        render_mode="human",   # 改为 None 可静默测试
        n_episodes=1,
        deterministic=True,
        level=LEVEL,
        custom_room=CUSTOM_ROOM
    )