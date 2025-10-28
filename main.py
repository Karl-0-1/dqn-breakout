import gymnasium as gym
import torch
import numpy as np
import time
import argparse
from dqn_agent import Agent, device
from env_utils import create_env

# --- Hyperparameters ---
ENV_ID = "ALE/Breakout-v5"
BUFFER_CAPACITY = 100000
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1000000
TARGET_UPDATE_FREQ = 1000
LEARNING_RATE = 0.0001
NUM_FRAMES = 5000000
WARMUP_STEPS = 10000 # Steps to fill buffer before training
MODEL_SAVE_PATH = "models/dqn_breakout_model.pth"

def train():
    print(f"Using device: {device}")
    env = create_env(ENV_ID)
    num_actions = env.action_space.n
    print(f"Number of actions: {num_actions}")

    agent = Agent(
        num_actions=num_actions,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        eps_start=EPS_START,
        eps_end=EPS_END,
        eps_decay=EPS_DECAY,
        lr=LEARNING_RATE,
        target_update_freq=TARGET_UPDATE_FREQ
    )

    print("Starting training...")
    all_rewards = []
    episode_reward = 0
    start_time = time.time()
    state, info = env.reset()

    for frame_idx in range(1, NUM_FRAMES + 1):
        action_tensor = agent.select_action(state, exploration=True)
        action = action_tensor.item()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        done = terminated or truncated

        clipped_reward = max(-1.0, min(reward, 1.0))
        reward_tensor = torch.tensor([clipped_reward], device=device, dtype=torch.float32)
        
        next_state_for_buffer = np.array(next_state) if not done else None
        
        agent.memory.push(np.array(state), action_tensor, reward_tensor, next_state_for_buffer, done)
        state = next_state

        if done:
            all_rewards.append(episode_reward)
            episode_reward = 0
            state, info = env.reset()

        if frame_idx > WARMUP_STEPS:
            agent.optimize_model()

        if frame_idx % agent.target_update_freq == 0:
            agent.update_target_net()

        if frame_idx % 100000 == 0:
            mean_reward = np.mean(all_rewards[-100:]) if all_rewards else 0.0
            elapsed_time = (time.time() - start_time) / 60
            print(f"Frames: {frame_idx}/{NUM_FRAMES} | Mean Reward (100ep): {mean_reward:.2f} | Time: {elapsed_time:.2f} min")

    print("Training complete!")
    agent.save_model(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    env.close()

def play():
    print(f"Loading model from {MODEL_SAVE_PATH}")
    env = create_env(ENV_ID)
    num_actions = env.action_space.n
    
    # Use dummy values for agent params we don't need for playing
    agent = Agent(num_actions, 1, 1, 1, 0, 0, 1, 0, 1)
    agent.load_model(MODEL_SAVE_PATH)
    agent.policy_net.eval()
    
    # Wrap env for video recording
    video_env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda e: e == 0)

    print("Starting evaluation...")
    state, info = video_env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action_tensor = agent.select_action(state, exploration=False)
        action = action_tensor.item()
        
        next_state, reward, terminated, truncated, _ = video_env.step(action)
        state = next_state
        total_reward += reward
        done = terminated or truncated

    print(f"Evaluation finished. Total reward: {total_reward}")
    video_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN for Atari Breakout")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "play"],
                        help="Mode to run: 'train' or 'play'")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "play":
        play()
