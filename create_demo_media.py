"""
Create Demo Media: Screenshots and GIFs
=======================================
This script runs the demos and captures screenshots/GIFs for README.
"""

import pygame
import numpy as np
from pathlib import Path
import sys
import time
from PIL import Image
import imageio

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "assignment1"))
sys.path.insert(0, str(Path(__file__).parent / "assignment2"))

def capture_screenshots():
    """Capture screenshots from environment demo."""
    print("Creating environment screenshots...")
    
    from chid_env import create_env
    
    env = create_env(render_mode="pygame")
    obs, _ = env.reset()
    
    # Initial state
    env.render()
    pygame.image.save(env.window, "assets/screenshots/initial_state.png")
    print("  Saved: assets/screenshots/initial_state.png")
    time.sleep(0.5)
    
    # Mid game
    for i in range(8):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            break
    
    pygame.image.save(env.window, "assets/screenshots/mid_game.png")
    print("  Saved: assets/screenshots/mid_game.png")
    time.sleep(0.5)
    
    # Goal reached
    env.close()
    
    # Create a new env for goal state
    env = create_env(render_mode="pygame")
    obs, _ = env.reset()
    
    # Simulate reaching goal (manually set position for demo)
    # Just take a screenshot of final state
    for i in range(15):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            pygame.image.save(env.window, "assets/screenshots/goal_reached.png")
            print("  Saved: assets/screenshots/goal_reached.png")
            break
    
    env.close()

def create_training_agent_gif():
    """Create GIF of trained agent navigating."""
    print("Creating trained agent GIF...")
    
    try:
        from assignment1_meftun import create_env
        from assignment2_qlearning import QLearningAgent
        
        env = create_env(render_mode="pygame")
        agent = QLearningAgent(env)
        
        try:
            agent.load_q_table("assignment2/q_table_final_4d.npy")
        except:
            print("  Q-table not found, training quickly...")
            agent.train(num_episodes=50, verbose=False)
        
        obs, _ = env.reset()
        frames = []
        done = False
        step = 0
        max_steps = 30
        
        print("  Capturing frames...")
        while not done and step < max_steps:
            env.render()
            
            # Capture frame
            frame = pygame.surfarray.array3d(env.window)
            frame = np.transpose(frame, (1, 0, 2))
            frames.append(frame)
            
            action = agent.choose_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            
            time.sleep(0.1)  # Small delay for GIF
        
        env.close()
        
        # Save GIF with infinite loop using PIL
        if frames:
            # Convert to PIL Images and resize
            pil_frames = []
            for frame in frames:
                img = Image.fromarray(frame)
                img = img.resize((800, 600), Image.Resampling.LANCZOS)
                pil_frames.append(img)
            
            # Save with PIL for better loop control
            pil_frames[0].save(
                "assets/gifs/trained_agent_demo.gif",
                save_all=True,
                append_images=pil_frames[1:],
                duration=200,  # milliseconds per frame
                loop=0  # 0 = infinite loop
            )
            print("  Saved: assets/gifs/trained_agent_demo.gif (infinite loop)")
        
    except Exception as e:
        print(f"  Error creating GIF: {e}")

def create_environment_demo_gif():
    """Create GIF showing environment features."""
    print("Creating environment demo GIF...")
    
    from chid_env import create_env
    
    env = create_env(render_mode="pygame")
    obs, _ = env.reset()
    
    frames = []
    
    # Capture initial state
    for _ in range(3):
        env.render()
        frame = pygame.surfarray.array3d(env.window)
        frame = np.transpose(frame, (1, 0, 2))
        frames.append(frame)
        time.sleep(0.1)
    
    # Show some random movements
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        frame = pygame.surfarray.array3d(env.window)
        frame = np.transpose(frame, (1, 0, 2))
        frames.append(frame)
        
        if terminated or truncated:
            break
        time.sleep(0.1)
    
    env.close()
    
    # Save GIF with infinite loop using PIL
    if frames:
        pil_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            img = img.resize((800, 600), Image.Resampling.LANCZOS)
            pil_frames.append(img)
        
        # Save with PIL for better loop control
        pil_frames[0].save(
            "assets/gifs/environment_demo.gif",
            save_all=True,
            append_images=pil_frames[1:],
            duration=150,  # milliseconds per frame
            loop=0  # 0 = infinite loop
        )
        print("  Saved: assets/gifs/environment_demo.gif (infinite loop)")

def create_visualization_screenshots():
    """Create screenshots of Q-table and policy visualizations."""
    print("Creating visualization screenshots...")
    
    try:
        from assignment1_meftun import create_env
        from assignment2_qlearning import QLearningAgent, visualize_q_table, visualize_policy
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        env = create_env()
        agent = QLearningAgent(env)
        
        try:
            agent.load_q_table("assignment2/q_table_final_4d.npy")
        except:
            print("  Training agent for visualizations...")
            agent.train(num_episodes=100, verbose=False)
            agent.save_q_table("assignment2/q_table_final_4d.npy")
        
        # Q-table visualization
        visualize_q_table(
            agent.q_table, env, 
            has_lover=0,
            save_path="assets/images/q_table_no_lover.png",
            show_plot=False
        )
        print("  Saved: assets/images/q_table_no_lover.png")
        
        visualize_q_table(
            agent.q_table, env,
            has_lover=1,
            save_path="assets/images/q_table_with_lover.png",
            show_plot=False
        )
        print("  Saved: assets/images/q_table_with_lover.png")
        
        # Policy visualization
        visualize_policy(
            agent.q_table, env,
            has_lover=0,
            save_path="assets/images/policy_no_lover.png",
            show_plot=False
        )
        print("  Saved: assets/images/policy_no_lover.png")
        
        visualize_policy(
            agent.q_table, env,
            has_lover=1,
            save_path="assets/images/policy_with_lover.png",
            show_plot=False
        )
        print("  Saved: assets/images/policy_with_lover.png")
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Create all demo media."""
    print("=" * 60)
    print("Creating Demo Media for README")
    print("=" * 60)
    print()
    
    # Create directories
    Path("assets/screenshots").mkdir(parents=True, exist_ok=True)
    Path("assets/images").mkdir(parents=True, exist_ok=True)
    Path("assets/gifs").mkdir(parents=True, exist_ok=True)
    
    # Create media
    capture_screenshots()
    print()
    create_environment_demo_gif()
    print()
    create_training_agent_gif()
    print()
    create_visualization_screenshots()
    print()
    
    print("=" * 60)
    print("All demo media created successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

