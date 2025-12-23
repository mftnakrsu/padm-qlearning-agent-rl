"""
Quick Start Demo
================
This script demonstrates both Assignment 1 and Assignment 2 in one go.
"""

import sys
from pathlib import Path

# Add assignment directories to path
sys.path.insert(0, str(Path(__file__).parent / "assignment1"))
sys.path.insert(0, str(Path(__file__).parent / "assignment2"))

def demo_assignment1():
    """Demo Assignment 1: Environment"""
    print("\n" + "=" * 60)
    print("ASSIGNMENT 1: CUSTOM ENVIRONMENT DEMO")
    print("=" * 60)
    
    from chid_env import create_env
    
    print("\nCreating environment...")
    env = create_env(render_mode="pygame")
    
    print("Resetting environment...")
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial state: Row={obs[0]}, Col={obs[1]}, Has Lover={obs[2]}")
    
    print("\nRunning 10 random steps...")
    print("(Pygame window should be open)")
    
    for step in range(10):
        env.render()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step+1}: Action={action}, Reward={reward}, "
              f"Position=({obs[0]}, {obs[1]}), Has Lover={obs[2]}")
        
        if terminated or truncated:
            print(f"Episode ended: Terminated={terminated}, Truncated={truncated}")
            break
    
    env.close()
    print("\n✓ Assignment 1 demo complete!")

def demo_assignment2():
    """Demo Assignment 2: Q-Learning Agent"""
    print("\n" + "=" * 60)
    print("ASSIGNMENT 2: Q-LEARNING AGENT DEMO")
    print("=" * 60)
    
    from assignment1_meftun import create_env
    from assignment2_qlearning import QLearningAgent
    import numpy as np
    
    print("\nCreating environment...")
    env = create_env()
    
    print("Loading trained Q-table...")
    agent = QLearningAgent(env)
    
    try:
        agent.load_q_table("assignment2/q_table_final_4d.npy")
        print("✓ Q-table loaded successfully!")
    except FileNotFoundError:
        print("⚠ Q-table not found. Training agent first...")
        print("This may take a few minutes...")
        agent.train(num_episodes=100, verbose=False)
        agent.save_q_table("assignment2/q_table_final_4d.npy")
        print("✓ Training complete!")
    
    print("\nTesting trained agent...")
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    print(f"Starting state: {obs}")
    
    while not done and steps < 50:
        action = agent.choose_action(obs, training=False)  # Greedy
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        
        if steps % 5 == 0:
            print(f"Step {steps}: Action={action}, Reward={reward}, "
                  f"Position=({obs[0]}, {obs[1]}), Has Lover={obs[2]}")
    
    print(f"\nEpisode complete!")
    print(f"  Total Steps: {steps}")
    print(f"  Total Reward: {total_reward}")
    print(f"  Success: {info.get('success', False)}")
    
    if total_reward > 600:
        print("  🎉 Excellent! Agent found optimal path (Lover → Goal)")
    elif total_reward > 0:
        print("  ✓ Good! Agent reached goal")
    else:
        print("  ⚠ Agent needs more training")
    
    print("\n✓ Assignment 2 demo complete!")

def main():
    """Run both demos."""
    print("\n" + "=" * 60)
    print("PADM Q-LEARNING AGENT - QUICK START DEMO")
    print("=" * 60)
    print("\nThis demo will:")
    print("  1. Show Assignment 1 environment (with pygame)")
    print("  2. Test Assignment 2 trained agent")
    print("\nPress Ctrl+C to skip any demo")
    
    try:
        demo_assignment1()
    except KeyboardInterrupt:
        print("\n⚠ Assignment 1 demo skipped")
    except Exception as e:
        print(f"\n⚠ Error in Assignment 1 demo: {e}")
    
    try:
        demo_assignment2()
    except KeyboardInterrupt:
        print("\n⚠ Assignment 2 demo skipped")
    except Exception as e:
        print(f"\n⚠ Error in Assignment 2 demo: {e}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
    print("\nFor more details, see:")
    print("  - assignment1/README.md")
    print("  - assignment2/README.md")
    print("  - Main README.md")

if __name__ == "__main__":
    main()

