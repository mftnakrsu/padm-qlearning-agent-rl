"""
Run Trained Q-Learning Agent
=============================
Bu script train edilmis agent'i calistirir ve pygame'de gosterir.

Kullanim:
    python run_trained_agent.py
"""

import numpy as np
import pygame
try:
    from chid_env import create_env
except ImportError:
    # Fallback for when running from parent directory
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "assignment1"))
    from chid_env import create_env
from assignment2_qlearning import QLearningAgent


def run_agent(q_table_path='q_table_final_4d.npy', num_episodes=3, delay=300):
    """
    Train edilmis agent'i calistir.

    Parameters:
    -----------
    q_table_path : str
        Q-table dosyasinin yolu
    num_episodes : int
        Kac episode calistirilacak
    delay : int
        Her adim arasindaki bekleme suresi (ms)
    """
    # Environment olustur
    env = create_env(render_mode='pygame')

    # Agent olustur ve Q-table yukle
    agent = QLearningAgent(env)
    agent.load_q_table(q_table_path)

    print("=" * 50)
    print("TRAINED AGENT DEMO")
    print("=" * 50)
    print(f"Q-table: {q_table_path}")
    print(f"Episodes: {num_episodes}")
    print("=" * 50)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        print(f"\nEpisode {ep + 1}:")

        while not done:
            env.render()

            # Greedy action (no exploration)
            action = agent.choose_action(obs, training=False)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            pygame.time.wait(delay)

        status = "SUCCESS" if info.get('success') else "FAILED"
        print(f"  {status} | Reward: {total_reward:.0f} | Steps: {steps}")

        # Episode arasi kisa bekleme
        pygame.time.wait(1000)

    print("\n" + "=" * 50)
    print("Demo bitti!")
    print("=" * 50)

    pygame.time.wait(2000)
    env.close()


if __name__ == "__main__":
    run_agent()
