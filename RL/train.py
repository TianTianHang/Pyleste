import argparse
from datetime import datetime
import os
import sys
import time

sys.path.append("")
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from celeste_env import CelesteGymEnv
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def train_agent_sb3(
    algorithm="PPO",
    total_timesteps=100000,
    max_steps=500,
    done_threshold=1,
    custom_room=None,
    goal=None,
    model_path=None,
    n_envs=8,
    n_steps=64,
    learning_rate=None,
    batch_size=None,
    gamma=0.99,
    gae_lambda=None,
    clip_range=None,
    ent_coef=None,
    vf_coef=None,
    max_grad_norm=None,
    n_epochs=10,
    tensorboard_log="./celeste_tensorboard/",
    log_interval=10,
    save_freq=50000,
):
    """
    Train a reinforcement learning agent using Stable Baselines3
    """
    print(f"Training with {algorithm} algorithm...")

    # Create environment
    # For algorithms that need vectorized environments, we use make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    env = make_vec_env(
        CelesteGymEnv,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "custom_room": custom_room,
            "goal": goal,
            "max_steps": max_steps,
            "done_threshold": done_threshold,
            "render_mode": None,
        },
    )

    # Set default hyperparameters based on algorithm if not provided
    if learning_rate is None:
        learning_rate = 3e-4 if algorithm == "PPO" else (1e-4 if algorithm == "DQN" else 7e-4)
    if gae_lambda is None:
        gae_lambda = 0.95 if algorithm == "PPO" else 1.0
    if clip_range is None and algorithm == "PPO":
        clip_range = 0.2
    if ent_coef is None:
        ent_coef = 0.01
    if vf_coef is None and algorithm == "A2C":
        vf_coef = 0.25
    if max_grad_norm is None and algorithm == "A2C":
        max_grad_norm = 0.5
    if batch_size is None:
        batch_size = n_envs * n_steps

    # Load pre-trained model if provided
    if model_path:
        if algorithm == "PPO":
            model = PPO.load(model_path, env=env)  # Load with the new environment
        elif algorithm == "DQN":
            model = DQN.load(model_path, env=env)
        elif algorithm == "A2C":
            model = A2C.load(model_path, env=env)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    # Choose algorithm from scratch if no pre-trained model
    else:
        if algorithm == "PPO":
            model = PPO(
                "MultiInputPolicy",
                env,
                verbose=1,
                tensorboard_log=tensorboard_log,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
            )
        elif algorithm == "DQN":
            # Note: DQN requires some adjustments for continuous observations
            model = DQN(
                "MultiInputPolicy",
                env,
                verbose=1,
                tensorboard_log=tensorboard_log,
                learning_rate=learning_rate,
                buffer_size=50000,
                learning_starts=1000,
                batch_size=batch_size or 32,
                tau=1.0,
                gamma=gamma,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=1000,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
            )
        elif algorithm == "A2C":
            model = A2C(
                "MultiInputPolicy",
                env,
                verbose=1,
                tensorboard_log=tensorboard_log,
                learning_rate=learning_rate,
                n_steps=n_steps,
                gamma=gamma,
                gae_lambda=gae_lambda,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Define a callback for evaluation
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/",
        eval_freq=max(1000, n_envs * n_steps),  # Evaluate at least every update cycle
        deterministic=True,
        render=False,
    )


    from stable_baselines3.common.callbacks import CheckpointCallback

    # Create checkpoint callback to save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // (n_envs * n_steps),  # Convert to number of calls to _on_step
        save_path=f"./RL/{algorithm}/checkpoints/",
        name_prefix="celeste_model",
        verbose=1,
    )

    # Combine callbacks
    callbacks = [eval_callback]
    if save_freq > 0:
        callbacks.append(checkpoint_callback)

    # Print training configuration
    print(f"Training configuration:")
    print(f"  Algorithm: {algorithm}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Parallel environments: {n_envs}")
    print(f"  Steps per update: {n_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Gamma: {gamma}")
    if algorithm == "PPO":
        print(f"  GAE Lambda: {gae_lambda}")
        print(f"  Clip range: {clip_range}")
    print(f"  Entropy coefficient: {ent_coef}")
    print(f"  Log interval: {log_interval}")
    print(f"  Checkpoint save frequency: {save_freq}")
    print("-" * 60)

    # Train the agent
    print("Starting training...")
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        log_interval=log_interval  # Log training progress every log_interval episodes
    )
    training_time = time.time() - start_time

    # Print training summary
    print("-" * 60)
    print("Training completed!")
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Average speed: {total_timesteps/training_time:.2f} steps/second")

    # Save the final trained model
    current_date = datetime.now().strftime(r"%Y%m%d%H%M%S")

    # Create directory if it doesn't exist
    os.makedirs(f"./RL/{algorithm}", exist_ok=True)
    os.makedirs(f"./RL/{algorithm}/checkpoints", exist_ok=True)  # Also make checkpoints dir

    model.save(f"./RL/{algorithm}/celeste_{algorithm.lower()}_model_{current_date}")
    model.save(f"./RL/{algorithm.lower()}_model")  # Save without timestamp for easy loading
    print(
        f"Model saved as ./RL/{algorithm}/celeste_{algorithm.lower()}_model_{current_date}"
    )
    print(f"Checkpoint models saved in ./RL/{algorithm}/checkpoints/")

    return model


def test_agent(model_path, algorithm="PPO", episodes=5, goal=None, custom_room=None):
    """
    Test the trained agent
    """
    # Load environment
    env = CelesteGymEnv(goal=goal, custom_room=custom_room)

    # Load the trained model
    if algorithm == "PPO":
        model = PPO.load(model_path)
    elif algorithm == "DQN":
        model = DQN.load(model_path)
    elif algorithm == "A2C":
        model = A2C.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    print(f"Testing trained {algorithm} agent...")

    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        step_count = 0

        print(f"\n--- Test Episode {episode + 1} ---")

        while True:
            # Predict action using the trained model
            action, _states = model.predict(obs, deterministic=True)

            # Take action in environment
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1

            # Print player position occasionally
            if step_count % 50 == 0:
                print(
                    f"Step {step_count}: Player at ({info['player_x']:.1f}, {info['player_y']:.1f}), "
                    f"action: {info['action']}, reward: {reward:.2f}"
                )
                env.render()
            # Check if episode is done
            if done or truncated or step_count > 1000:
                print(
                    f"Episode {episode + 1} finished: Total reward = {total_reward:.2f}, Steps = {step_count}"
                )
                print(
                    f"Final position: ({info['player_x']:.1f}, {info['player_y']:.1f})"
                )
                break


def get_default_room():
    """Return a default Celeste room for testing."""
    return """
    w w w w w w w w w w . . . . w w
    w w w w w w w w w . . . . . < w
    w w w v v v v . . . . . . . < w
    w w > . . . . . . . w w w w w w
    w > . . . . . . w w . . . . . w
    w w . . . . . . . . . . . . . w
    w w . . . . . . . . . . . . . w
    w w w w w w . . . . . . . . . w
    . . . . . . . . . . . . . . p w
    . . . . . . . . . . . w w w w w
    . . . . . . . . . . . w . . . .
    . . . . . . . . . . . w . . . .
    . . . . . . . . w . . w . . . .
    . . . . . . . . w . . w . . . .
    . . . . . . . . w . . w . . . .
    w w w w w w w w w w w w w w w w
    """

def get_curriculum_rooms():
    """Return a set of rooms with increasing difficulty for curriculum learning."""
    return [
        # Level 1: Simple room with no obstacles
        """
        w w w w w w w w w w w w w w w w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        p . . . . . . . . . . . . . . w
        w w w w w w w w w w w w w w w w
        """,
        # Level 2: Room with some platforms
        """
        w w w w w w w w w w w w w w w w
        w w w w . . . . . . . . . . w w
        w w w w . . . . . . . . . . w w
        w w w w . . . . . . . . . . w w
        w w w w . . . . . . . . . . w w
        w w w w . . . . . . . . . . w w
        w w w w . . . . . . . . . . w w
        w w w w . . . . . . . . . . w w
        w w w w . . . . . . . . . . w w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        p . . . . . . . . . . . . . . w
        w w w w w w w w w w w w w w w w
        """,
        # Level 3: Room with gaps to jump over
        """
        w w w w w w w w w w w w w w w w
        w . . . . . w w w . . . . . . w
        w . . . . . w w w . . . . . . w
        w . . . . . w w w . . . . . . w
        w . . . . . w w w . . . . . . w
        w . . . . . w w w . . . . . . w
        w . . . . . w w w . . . . . . w
        w . . . . . w w w . . . . . . w
        w . . . . . w w w . . . . . . w
        w . . . . . w w w . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        p . . . . . . . . . . . . . . w
        w w w w w w w w w w w w w w w w
        """,
        # Level 4: Room with spikes to avoid
        """
        w w w w w w w w w w w w w w w w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        w . . . . . . . . . . . . . . w
        v v v . . . . . . . . . . . . w
        w w w . . . . . . . . . . . . w
        p . . . . . . . . . . . . . . w
        w w w w w w w w w w w w w w w w
        """,
        # Level 5: Complex room (original default)
        """
        w w w w w w w w w w . . . . w w
        w w w w w w w w w . . . . . < w
        w w w v v v v . . . . . . . < w
        w w > . . . . . . . w w w w w w
        w > . . . . . . w w . . . . . w
        w w . . . . . . . . . . . . . w
        w w . . . . . . . . . . . . . w
        w w w w w w . . . . . . . . . w
        . . . . . . . . . . . . . . p w
        . . . . . . . . . . . w w w w w
        . . . . . . . . . . . w . . . .
        . . . . . . . . . . . w . . . .
        . . . . . . . . w . . w . . . .
        . . . . . . . . w . . w . . . .
        . . . . . . . . w . . w . . . .
        w w w w w w w w w w w w w w w w
        """
    ]

def main():
    parser = argparse.ArgumentParser(description="Train or test a Celeste RL agent")
    parser.add_argument(
        "mode",
        choices=["train", "test"],
        help="Mode: train or test"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["PPO", "DQN", "A2C"],
        help="RL algorithm to use (default: PPO)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Number of training timesteps (default: 100000)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500)"
    )
    parser.add_argument(
        "--done-threshold",
        type=int,
        default=1,
        help="Done threshold (default: 1)"
    )
    parser.add_argument(
        "--goal-x",
        type=int,
        default=14,
        help="Goal X coordinate (default: 14)"
    )
    parser.add_argument(
        "--goal-y",
        type=int,
        default=8,
        help="Goal Y coordinate (default: 8)"
    )
    parser.add_argument(
        "--room-file",
        type=str,
        help="Path to file containing custom room layout"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to pre-trained model for fine-tuning"
    )
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=5,
        help="Number of test episodes (default: 5)"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        help="Path to save the trained model"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments (default: 8)"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=64,
        help="Number of steps per update (default: 64)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log interval for training progress (default: 10)"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50000,
        help="Frequency to save model checkpoints (default: 50000)"
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable curriculum learning with increasing difficulty"
    )
    parser.add_argument(
        "--curriculum-levels",
        type=int,
        default=5,
        help="Number of curriculum levels (default: 5)"
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.8,
        help="Success rate threshold to advance curriculum level (default: 0.8)"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate for curriculum progression (default: 10)"
    )

    args = parser.parse_args()

    # Load custom room from file if provided, otherwise use default
    custom_room = None
    if args.room_file:
        with open(args.room_file, 'r') as f:
            custom_room = f.read()
    else:
        # Use default room for testing
        custom_room = get_default_room()

    goal = (args.goal_x, args.goal_y)

    if args.mode == "train":
        print(f"Training with {args.algorithm} algorithm...")
        print(f"Goal: {goal}")
        print(f"Custom room: {'Yes' if custom_room else 'No'}")
        print(f"Total timesteps: {args.timesteps}")
        print(f"Max steps per episode: {args.max_steps}")

        try:
            if args.curriculum:
                print("Using curriculum learning...")

                def evaluate_agent(model, env, n_episodes=10):
                    """Evaluate the agent to determine success rate."""
                    successes = 0

                    # Create a single environment for evaluation
                    eval_env = CelesteGymEnv(custom_room=env.envs[0].custom_room, goal=env.envs[0].goal)

                    for episode in range(n_episodes):
                        obs, _ = eval_env.reset()
                        done = False

                        while not done:
                            action, _ = model.predict(obs, deterministic=True)
                            obs, reward, terminated, truncated, info = eval_env.step(action)
                            done = terminated or truncated

                            if info.get('success', False):
                                successes += 1
                                break

                    success_rate = successes / n_episodes
                    return success_rate

                def train_with_curriculum(
                    algorithm="PPO",
                    total_timesteps=100000,
                    max_steps=500,
                    done_threshold=1,
                    curriculum_levels=5,
                    success_threshold=0.8,
                    eval_episodes=10,
                    n_envs=8,
                    n_steps=64,
                    learning_rate=None,
                    batch_size=None,
                    gamma=0.99,
                    gae_lambda=None,
                    clip_range=None,
                    ent_coef=None,
                    vf_coef=None,
                    max_grad_norm=None,
                    n_epochs=10,
                    tensorboard_log="./celeste_tensorboard/",
                ):
                    """Train with curriculum learning."""
                    print(f"Starting curriculum learning with {curriculum_levels} levels...")

                    curriculum_rooms = get_curriculum_rooms()
                    goals = [(14, 14), (14, 10), (14, 6), (14, 4), (14, 8)]  # Goals for each level

                    # Start with first level
                    current_level = 0

                    # Initialize model for first level
                    env = make_vec_env(
                        CelesteGymEnv,
                        n_envs=n_envs,
                        vec_env_cls=SubprocVecEnv,
                        env_kwargs={
                            "custom_room": curriculum_rooms[current_level],
                            "goal": goals[current_level],
                            "max_steps": max_steps,
                            "done_threshold": done_threshold,
                            "render_mode": None,
                        },
                    )

                    # Set default hyperparameters based on algorithm
                    if learning_rate is None:
                        learning_rate = 3e-4 if algorithm == "PPO" else (1e-4 if algorithm == "DQN" else 7e-4)
                    if gae_lambda is None:
                        gae_lambda = 0.95 if algorithm == "PPO" else 1.0
                    if clip_range is None and algorithm == "PPO":
                        clip_range = 0.2
                    if ent_coef is None:
                        ent_coef = 0.01
                    if vf_coef is None and algorithm == "A2C":
                        vf_coef = 0.25
                    if max_grad_norm is None and algorithm == "A2C":
                        max_grad_norm = 0.5
                    if batch_size is None:
                        batch_size = n_envs * n_steps

                    # Print curriculum configuration
                    print(f"Curriculum configuration:")
                    print(f"  Algorithm: {algorithm}")
                    print(f"  Total timesteps: {total_timesteps}")
                    print(f"  Number of levels: {curriculum_levels}")
                    print(f"  Success threshold: {success_threshold}")
                    print(f"  Evaluation episodes: {eval_episodes}")
                    print(f"  Parallel environments: {n_envs}")
                    print(f"  Steps per update: {n_steps}")
                    print(f"  Learning rate: {learning_rate}")
                    print("-" * 60)

                    # Choose algorithm
                    if algorithm == "PPO":
                        model = PPO(
                            "MultiInputPolicy",
                            env,
                            verbose=1,
                            tensorboard_log=tensorboard_log,
                            learning_rate=learning_rate,
                            n_steps=n_steps,
                            batch_size=batch_size,
                            n_epochs=n_epochs,
                            gamma=gamma,
                            gae_lambda=gae_lambda,
                            clip_range=clip_range,
                            ent_coef=ent_coef,
                        )
                    elif algorithm == "DQN":
                        model = DQN(
                            "MultiInputPolicy",
                            env,
                            verbose=1,
                            tensorboard_log=tensorboard_log,
                            learning_rate=learning_rate,
                            buffer_size=50000,
                            learning_starts=1000,
                            batch_size=batch_size or 32,
                            tau=1.0,
                            gamma=gamma,
                            train_freq=4,
                            gradient_steps=1,
                            target_update_interval=1000,
                            exploration_fraction=0.1,
                            exploration_initial_eps=1.0,
                            exploration_final_eps=0.05,
                        )
                    elif algorithm == "A2C":
                        model = A2C(
                            "MultiInputPolicy",
                            env,
                            verbose=1,
                            tensorboard_log=tensorboard_log,
                            learning_rate=learning_rate,
                            n_steps=n_steps,
                            gamma=gamma,
                            gae_lambda=gae_lambda,
                            ent_coef=ent_coef,
                            vf_coef=vf_coef,
                            max_grad_norm=max_grad_norm,
                        )
                    else:
                        raise ValueError(f"Unsupported algorithm: {algorithm}")

                    # Training loop
                    start_time = time.time()
                    remaining_timesteps = total_timesteps
                    level_times = []
                    level_start_time = start_time

                    while current_level < curriculum_levels and remaining_timesteps > 0:
                        print(f"\nStarting training on level {current_level + 1}/{curriculum_levels}")

                        # Calculate timesteps for this level (distribute evenly)
                        level_timesteps = min(remaining_timesteps, total_timesteps // curriculum_levels)
                        remaining_timesteps -= level_timesteps

                        # Train on current level
                        level_start_time = time.time()
                        model.learn(total_timesteps=level_timesteps, progress_bar=True)
                        level_time = time.time() - level_start_time
                        level_times.append(level_time)

                        # Evaluate on current level
                        print(f"Evaluating on level {current_level + 1}...")
                        success_rate = evaluate_agent(model, env, eval_episodes)
                        print(f"Success rate on level {current_level + 1}: {success_rate:.2f}")

                        # Check if we can advance to the next level
                        if success_rate >= success_threshold and current_level < curriculum_levels - 1:
                            current_level += 1
                            print(f"Advancing to level {current_level + 1}")

                            # Update environment for next level
                            new_env = make_vec_env(
                                CelesteGymEnv,
                                n_envs=n_envs,
                                vec_env_cls=SubprocVecEnv,
                                env_kwargs={
                                    "custom_room": curriculum_rooms[current_level],
                                    "goal": goals[current_level],
                                    "max_steps": max_steps,
                                    "done_threshold": done_threshold,
                                    "render_mode": None,
                                },
                            )
                            model.set_env(new_env)  # Update the model's environment
                            env = new_env  # Update reference
                        else:
                            print(f"Remaining on level {current_level + 1}")

                    # Print curriculum summary
                    total_time = time.time() - start_time
                    print("-" * 60)
                    print("Curriculum learning completed!")
                    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
                    print(f"Average speed: {total_timesteps/total_time:.2f} steps/second")
                    print(f"Completed levels: {current_level + 1}/{curriculum_levels}")
                    for i, time_taken in enumerate(level_times):
                        print(f"Level {i+1} time: {time_taken:.2f} seconds")
                    print("-" * 60)

                    return model

                model = train_with_curriculum(
                    algorithm=args.algorithm,
                    total_timesteps=args.timesteps,
                    max_steps=args.max_steps,
                    done_threshold=args.done_threshold,
                    curriculum_levels=args.curriculum_levels,
                    success_threshold=args.success_threshold,
                    eval_episodes=args.eval_episodes,
                    n_envs=args.n_envs,
                    n_steps=args.n_steps,
                    learning_rate=None,  # Will use default
                    batch_size=None,     # Will use default
                    tensorboard_log="./celeste_tensorboard/",
                )
            else:
                model = train_agent_sb3(
                    algorithm=args.algorithm,
                    total_timesteps=args.timesteps,
                    max_steps=args.max_steps,
                    done_threshold=args.done_threshold,
                    goal=goal,
                    custom_room=custom_room,
                    model_path=args.model_path,
                    n_envs=args.n_envs,
                    n_steps=args.n_steps,
                    log_interval=args.log_interval,
                    save_freq=args.save_freq,
                )
            print("Training completed!")
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()

    elif args.mode == "test":
        model_path = args.model_path or f"./RL/{args.algorithm}/celeste_{args.algorithm.lower()}_model"

        print(f"Testing {args.algorithm} agent...")
        print(f"Goal: {goal}")
        print(f"Model path: {model_path}")
        print(f"Episodes: {args.test_episodes}")

        try:
            test_agent(
                model_path,
                algorithm=args.algorithm,
                episodes=args.test_episodes,
                goal=goal,
                custom_room=custom_room,
            )
        except Exception as e:
            print(f"Error during testing: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
