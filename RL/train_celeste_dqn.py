import os
import sys
sys.path.append(os.getcwd())

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
# 假设 CelesteGymEnv 在当前路径下可导入
from RL.celeste_env import CelesteGymEnv  # 👈 替换为实际模块名

# 设置日志和模型保存路径
log_dir = "./logs/celeste_dqn/"
model_dir = "./models/celeste_dqn/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

def make_env(goal=(64.0, 32.0), custom_room=None, rank=0,max_step=2000,level=0):
    """
    返回一个环境构造函数（用于 VecEnv）
    """
    def _init():
        env = CelesteGymEnv(
            goal=goal,
            custom_room=custom_room,
            render_mode=None,
            max_step=max_step,
            level=level
        )
        env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}"))
        return env
    return _init

if __name__ == "__main__":
    # 配置目标点
    TARGET_GOAL = (108, 0.0)
    LEVEL=0
    # 注意：DQN 不支持 VecEnv 的多个并行环境进行经验回放采样（SB3 的 DQN 只能处理单环境或 DummyVecEnv(n=1)）
    # 因此我们只用 1 个环境（DummyVecEnv 仍可用，但 n_envs=1）
    num_envs = 1  # DQN 在 SB3 中不支持多进程采样到 replay buffer
    env = DummyVecEnv([make_env(goal=TARGET_GOAL, rank=i,level=LEVEL) for i in range(num_envs)])

    # 检查 action space 是否为 Discrete
    assert isinstance(env.action_space, gym.spaces.Discrete), \
        "DQN only supports discrete action spaces!"

    # 创建 DQN 模型
    model = DQN(
        "MultiInputPolicy",  # 自动处理 Dict observation（如 {"image": ..., "vector": ...}）
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=1e-4,
        buffer_size=50000,          # 经验回放缓冲区大小
        learning_starts=10000,      # 开始学习前的 warmup steps
        batch_size=32,
        tau=1.0,                    # 硬更新 target network（也可设为 0.005 软更新）
        gamma=0.99,
        train_freq=4,               # 每 4 步训练一次
        gradient_steps=1,
        target_update_interval=1000,  # 每 1000 步更新 target net
        exploration_fraction=0.2,   # 前 20% timesteps 进行 epsilon 衰减
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        seed=42,
    )

    # 回调：定期保存模型
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="celeste_dqn"
    )

    # 评估回调（使用独立环境）
    eval_env = DummyVecEnv([make_env(goal=TARGET_GOAL, rank=999,level=LEVEL)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best"),
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1,
    )

    # 开始训练
    total_timesteps = 500_000
    print(f"Starting DQN training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name="dqn_celeste_run",
        progress_bar=True,
    )

    # 保存最终模型
    model.save(os.path.join(model_dir, "celeste_dqn_final"))
    print("DQN training finished and model saved.")