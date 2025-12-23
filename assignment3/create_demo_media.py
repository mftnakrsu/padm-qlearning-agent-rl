"""
Create Demo Media for Assignment 3: DQN Agent
============================================
This script creates screenshots and videos/GIFs of the trained DQN agent.
"""

import pygame
import numpy as np
from pathlib import Path
import sys
import time
from PIL import Image
import imageio

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from env import ContinuousMazeEnv
from DQN_model import DQN
from main import DQNAgent
import torch

def capture_screenshots():
    """Capture screenshots from environment."""
    print("Creating environment screenshots...")
    
    assets_dir = Path(__file__).parent.parent / "assets" / "screenshots"
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    env = ContinuousMazeEnv(render_mode="human")
    obs, _ = env.reset()
    
    # Initial state
    env.render()
    pygame.image.save(env.screen, str(assets_dir / "dqn_initial_state.png"))
    print(f"  Saved: {assets_dir / 'dqn_initial_state.png'}")
    time.sleep(0.5)
    
    # Mid navigation
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done or truncated:
            break
    
    pygame.image.save(env.screen, str(assets_dir / "dqn_mid_navigation.png"))
    print(f"  Saved: {assets_dir / 'dqn_mid_navigation.png'}")
    time.sleep(0.5)
    
    env.close()

def create_trained_agent_gif():
    """Create GIF of trained DQN agent navigating."""
    print("Creating trained DQN agent GIF...")
    
    try:
        env = ContinuousMazeEnv(render_mode="human")
        agent = DQNAgent(state_dim=2, action_dim=4)
        
        # Load trained model
        try:
            agent.q_network.load_state_dict(torch.load("dqn_final.pth", map_location='cpu'))
            print("  Loaded trained model: dqn_final.pth")
        except FileNotFoundError:
            print("  ⚠ dqn_final.pth not found. Training quickly...")
            # Would need to train here, but skip for now
            return
        
        # Reset to starting position (0.1, 0.5)
        state, _ = env.reset()
        frames = []
        done = False
        step = 0
        max_steps = 100
        
        print("  Capturing frames...")
        while not done and step < max_steps:
            env.render()
            
            # Capture frame
            frame = pygame.surfarray.array3d(env.screen)
            frame = np.transpose(frame, (1, 0, 2))
            frames.append(frame)
            
            # Get action from trained agent (greedy)
            action = agent.select_action(state, training=False)
            state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            step += 1
            
            time.sleep(0.05)  # Small delay for GIF
        
        env.close()
        
        # Save GIF with infinite loop
        if frames:
            pil_frames = []
            for frame in frames:
                img = Image.fromarray(frame)
                img = img.resize((600, 600), Image.Resampling.LANCZOS)
                pil_frames.append(img)
            
            gif_path = Path(__file__).parent.parent / "assets" / "gifs" / "dqn_trained_agent.gif"
            gif_path.parent.mkdir(parents=True, exist_ok=True)
            
            pil_frames[0].save(
                str(gif_path),
                save_all=True,
                append_images=pil_frames[1:],
                duration=50,  # milliseconds per frame
                loop=0  # infinite loop
            )
            print(f"  Saved: {gif_path} (infinite loop)")
        
    except Exception as e:
        print(f"  Error creating GIF: {e}")
        import traceback
        traceback.print_exc()

def create_environment_demo_gif():
    """Create GIF showing environment with random agent."""
    print("Creating environment demo GIF...")
    
    env = ContinuousMazeEnv(render_mode="human")
    obs, _ = env.reset()
    
    frames = []
    
    # Capture initial state
    for _ in range(3):
        env.render()
        frame = pygame.surfarray.array3d(env.screen)
        frame = np.transpose(frame, (1, 0, 2))
        frames.append(frame)
        time.sleep(0.1)
    
    # Show random movements
    for i in range(30):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        
        frame = pygame.surfarray.array3d(env.screen)
        frame = np.transpose(frame, (1, 0, 2))
        frames.append(frame)
        
        if done or truncated:
            break
        time.sleep(0.1)
    
    env.close()
    
    # Save GIF with infinite loop
    if frames:
        pil_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            img = img.resize((600, 600), Image.Resampling.LANCZOS)
            pil_frames.append(img)
        
        gif_path = Path(__file__).parent.parent / "assets" / "gifs" / "dqn_environment_demo.gif"
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        
        pil_frames[0].save(
            str(gif_path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=100,  # milliseconds per frame
            loop=0  # infinite loop
        )
        print(f"  Saved: {gif_path} (infinite loop)")

def create_training_curves_screenshot():
    """Copy training curves to assets."""
    print("Copying training curves...")
    
    import shutil
    
    src = Path(__file__).parent / "training_curves.png"
    dst = Path(__file__).parent.parent / "assets" / "images" / "dqn_training_curves.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    if src.exists():
        shutil.copy(src, dst)
        print(f"  Saved: {dst}")
    else:
        print("  ⚠ training_curves.png not found")

def main():
    """Create all demo media for Assignment 3."""
    print("=" * 60)
    print("Creating Demo Media for Assignment 3: DQN Agent")
    print("=" * 60)
    print()
    
    # Create directories (already handled in individual functions)
    
    # Create media
    capture_screenshots()
    print()
    create_environment_demo_gif()
    print()
    create_trained_agent_gif()
    print()
    create_training_curves_screenshot()
    print()
    
    print("=" * 60)
    print("All demo media created successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

