import os
import sys
sys.path.append(os.getcwd())
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

# 假设 CelesteGymEnv 在当前路径下可导入
from RL.celeste_env import CelesteGymEnv  # 👈 替换为实际模块名，如 'celeste_env'

# 设置日志和模型保存路径
log_dir = "./logs/celeste_ppo/"
model_dir = "./models/celeste_ppo/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

def make_env(goal=(64.0, 32.0), custom_room=None, rank=0,max_step=2000,level=0,randomize_start_position=False):
    """
    返回一个环境构造函数（用于 VecEnv）
    """
    def _init():
        env = CelesteGymEnv(
            goal=goal,
            custom_room=custom_room,
            render_mode=None,
            max_step=max_step,
            level=level,
            randomize_start_position=randomize_start_position
        )
        env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}"))
        return env
    return _init

if __name__ == "__main__":
    # 配置目标点（可根据 curriculum 改变）
    RAND_POS=False
    TARGET_GOAL = (108, -1.0) # 示例目标：房间中某个位置
    LEVEL=0
    EVAL_GOAL=(108, -1.0)
    EVAL_LEVEL=0
    BEST_MODEL_PATH=None#'models/celeste_ppo/best/best_model.zip' #RL/finished_models/ppo/best_model.zip'
    total_timesteps = 1000_000# 根据难度调整
    CUSTOM_ROOM=None
    # 创建向量化环境（即使1个也推'荐用 DummyVecEnv）
    num_envs = 8  # 可增加到 4/8 提升样本效率（需确保 PICO8 支持多实例）
    env = DummyVecEnv([make_env(goal=TARGET_GOAL, rank=i,max_step=1200,level=LEVEL,randomize_start_position=RAND_POS,custom_room=CUSTOM_ROOM) for i in range(num_envs)])

    # 可选：如果 observation 是 Dict，SB3 默认支持，但需确认网络结构
    # PPO 默认会自动处理 spaces.Dict（使用 CombinedExtractor）

    # 创建 PPO 模型
    model = PPO(
        "MultiInputPolicy",  # 自动处理 Dict observation
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=1e-3,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=42,
    )
    if BEST_MODEL_PATH:
        print(F"load model from {BEST_MODEL_PATH}")
        model.load(BEST_MODEL_PATH)
    # 回调：定期保存模型 & 评估
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # 每 10k steps 保存一次
        save_path=model_dir,
        name_prefix="celeste_ppo"
    )

    # 可选：创建独立评估环境
    eval_env = DummyVecEnv([make_env(goal=EVAL_GOAL, rank=999,max_step=1200,level=EVAL_LEVEL,randomize_start_position=RAND_POS,custom_room=CUSTOM_ROOM)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best"),
        log_path=log_dir,
        eval_freq=5000,  # 每 5k training steps 评估一次
        deterministic=True,
        render=False,
        n_eval_episodes=10,
        verbose=1,
    )

    # 开始训练
    
    print(f"Starting PPO training for {total_timesteps} timesteps... level: {LEVEL}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name="ppo_celeste_run",
        progress_bar=True,
    )

    # 保存最终模型
    model.save(os.path.join(model_dir, "celeste_ppo_final"))
    print("Training finished and model saved.")