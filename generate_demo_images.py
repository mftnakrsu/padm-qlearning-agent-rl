"""
Generate Demo Images for README
================================
This script generates screenshots and visualizations for the README.
"""

import pygame
import numpy as np
from pathlib import Path
import sys

# Add assignment directories to path
sys.path.insert(0, str(Path(__file__).parent / "assignment1"))
sys.path.insert(0, str(Path(__file__).parent / "assignment2"))

from chid_env import create_env
from assignment2_qlearning import QLearningAgent, visualize_q_table, visualize_policy, plot_training_curves

def generate_environment_screenshots():
    """Generate screenshots of the environment."""
    print("Generating environment screenshots...")
    
    env = create_env(render_mode="pygame")
    obs, _ = env.reset()
    
    # Initial state screenshot
    env.render()
    pygame.image.save(env.window, "assets/screenshots/initial_state.png")
    print("✓ Saved: assets/screenshots/initial_state.png")
    
    # Mid-game screenshot (after a few steps)
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            break
    
    pygame.image.save(env.window, "assets/screenshots/mid_game.png")
    print("✓ Saved: assets/screenshots/mid_game.png")
    
    env.close()

def generate_q_table_visualizations():
    """Generate Q-table heatmap visualizations."""
    print("Generating Q-table visualizations...")
    
    try:
        from assignment1_meftun import create_env
        env = create_env()
        agent = QLearningAgent(env)
        agent.load_q_table("assignment2/q_table_final_4d.npy")
        
        # Visualize without lover
        visualize_q_table(
            agent.q_table, 
            env, 
            has_lover=0,
            save_path="assets/images/q_table_no_lover.png",
            show_plot=False
        )
        print("✓ Saved: assets/images/q_table_no_lover.png")
        
        # Visualize with lover
        visualize_q_table(
            agent.q_table, 
            env, 
            has_lover=1,
            save_path="assets/images/q_table_with_lover.png",
            show_plot=False
        )
        print("✓ Saved: assets/images/q_table_with_lover.png")
        
    except FileNotFoundError:
        print("⚠ Q-table not found. Please train the agent first.")
        print("  Run: cd assignment2 && python assignment2_main.py")

def generate_policy_visualizations():
    """Generate policy arrow visualizations."""
    print("Generating policy visualizations...")
    
    try:
        from assignment1_meftun import create_env
        env = create_env()
        agent = QLearningAgent(env)
        agent.load_q_table("assignment2/q_table_final_4d.npy")
        
        # Policy without lover
        visualize_policy(
            agent.q_table,
            env,
            has_lover=0,
            save_path="assets/images/policy_no_lover.png",
            show_plot=False
        )
        print("✓ Saved: assets/images/policy_no_lover.png")
        
        # Policy with lover
        visualize_policy(
            agent.q_table,
            env,
            has_lover=1,
            save_path="assets/images/policy_with_lover.png",
            show_plot=False
        )
        print("✓ Saved: assets/images/policy_with_lover.png")
        
    except FileNotFoundError:
        print("⚠ Q-table not found. Please train the agent first.")

def main():
    """Generate all demo images."""
    print("=" * 60)
    print("Generating Demo Images for README")
    print("=" * 60)
    print()
    
    # Create directories
    Path("assets/screenshots").mkdir(parents=True, exist_ok=True)
    Path("assets/images").mkdir(parents=True, exist_ok=True)
    Path("assets/gifs").mkdir(parents=True, exist_ok=True)
    
    # Generate images
    generate_environment_screenshots()
    print()
    generate_q_table_visualizations()
    print()
    generate_policy_visualizations()
    print()
    
    print("=" * 60)
    print("✓ All images generated successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

