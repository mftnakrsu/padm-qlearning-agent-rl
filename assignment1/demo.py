"""
PADM Assignment 1 - Demo Script
================================

This script demonstrates all features of the ChidEnv environment.

Usage:
    python demo.py

Author: Meftun Akarsu
"""

import pygame
from chid_env import ChidEnv, create_env


def print_env_info(env):
    """Print environment information."""
    print("=" * 60)
    print("ENVIRONMENT INFORMATION")
    print("=" * 60)
    print(f"Grid Size: {env.num_rows} x {env.num_cols}")
    print(f"Total Cells: {env.num_rows * env.num_cols}")
    print()
    print("Observation Space:")
    print(f"  Type: {type(env.observation_space).__name__}")
    print(f"  Shape: {env.observation_space.shape}")
    print(f"  Low: {env.observation_space.low}")
    print(f"  High: {env.observation_space.high}")
    print()
    print("Action Space:")
    print(f"  Type: {type(env.action_space).__name__}")
    print(f"  Number of Actions: {env.action_space.n}")
    print(f"  Actions: 0=Up, 1=Down, 2=Right")
    print()
    print("Special States:")
    print(f"  Agent Start: {env.agent_start.tolist()}")
    print(f"  Goal States: {[g.tolist() for g in env.goal_states]}")
    print(f"  Danger States (Hell): {len(env.danger_states)} states")
    print(f"  Obstacles: {len(env.obstacle_states)} states")
    print(f"  Reward States: {len(env.reward_states)} states")
    print(f"  Lover State: {env.lover_state.tolist() if env.lover_state is not None else None}")
    print()
    print("Reward Structure:")
    print(f"  Goal Reward: +{env.goal_reward}")
    print(f"  Goal with Lover: +{env.goal_reward * env.lover_multiplier}")
    print(f"  Danger Penalty: -{env.danger_penalty}")
    print(f"  Lover Bonus: +{env.goal_reward}")
    print(f"  Mini Reward: +{env.mini_reward}")
    print(f"  Living Cost: -{env.living_cost}")
    print(f"  Max Steps: {env.max_steps}")
    print("=" * 60)


def demo_step_by_step():
    """Demonstrate step-by-step execution."""
    print("\n" + "=" * 60)
    print("STEP-BY-STEP DEMO")
    print("=" * 60)

    env = create_env(render_mode="human")

    # Reset
    print("\n1. RESET")
    obs, info = env.reset()
    print(f"   Initial observation: {obs}")
    print(f"   Info: {info}")
    env.render()

    # Take some actions
    actions = [2, 2, 0, 2, 2, 2, 0]  # Right, Right, Up, Right, Right, Right, Up
    action_names = ["UP", "DOWN", "RIGHT"]

    print("\n2. TAKING ACTIONS")
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Step {i+1}: Action={action_names[action]}, "
              f"Obs={obs.tolist()}, Reward={reward:+.1f}, "
              f"Done={terminated or truncated}")

        if terminated or truncated:
            break

    env.close()
    print("\n   Demo completed!")


def demo_pygame():
    """Demonstrate pygame visualization."""
    print("\n" + "=" * 60)
    print("PYGAME VISUALIZATION DEMO")
    print("=" * 60)
    print("Watch the pygame window!")
    print("Random agent will play for up to 50 steps.")

    env = create_env(render_mode="pygame")
    obs, _ = env.reset()

    action_names = ["UP", "DOWN", "RIGHT"]
    done = False
    step = 0

    while not done and step < 50:
        env.render()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step {step+1}: {action_names[action]} -> [{obs[0]},{obs[1]}], "
              f"HasLover={obs[2]}, Reward={reward:+.1f}")

        done = terminated or truncated
        step += 1

        pygame.time.wait(300)

    if info.get("success"):
        print("\nSUCCESS! Agent reached the goal!")
    elif info.get("hit_danger"):
        print("\nFAILED! Agent hit a danger zone!")
    else:
        print("\nDemo ended (max steps or truncated)")

    print(f"Total Reward: {info['total_reward']:.1f}")

    pygame.time.wait(2000)
    env.close()


def demo_keyboard():
    """Keyboard control demo."""
    print("\n" + "=" * 60)
    print("KEYBOARD CONTROL DEMO")
    print("=" * 60)
    print("Controls:")
    print("  W or UP    = Move Up")
    print("  S or DOWN  = Move Down")
    print("  D or RIGHT = Move Right")
    print("  Q          = Quit")
    print("=" * 60)

    env = create_env(render_mode="pygame")
    obs, _ = env.reset()

    done = False

    while not done:
        env.render()

        action = None
        waiting = True

        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_w, pygame.K_UP]:
                        action = 0
                        waiting = False
                    elif event.key in [pygame.K_s, pygame.K_DOWN]:
                        action = 1
                        waiting = False
                    elif event.key in [pygame.K_d, pygame.K_RIGHT]:
                        action = 2
                        waiting = False
                    elif event.key == pygame.K_q:
                        env.close()
                        print("Quit by user.")
                        return

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.render()

    if info.get("success"):
        print("\nSUCCESS! You reached the goal!")
    elif info.get("hit_danger"):
        print("\nGAME OVER! You hit a danger zone!")
    else:
        print("\nTime's up!")

    print(f"Total Reward: {info['total_reward']:.1f}")

    pygame.time.wait(3000)
    env.close()


def main():
    """Main function."""
    env = create_env(render_mode=None)
    print_env_info(env)
    env.close()

    print("\nSelect demo mode:")
    print("1: Step-by-step (text output)")
    print("2: Pygame visualization (random agent)")
    print("3: Keyboard control (play yourself)")
    print("4: Run all demos")

    try:
        choice = input("\nYour choice (1/2/3/4): ").strip() or "2"

        if choice == "1":
            demo_step_by_step()
        elif choice == "2":
            demo_pygame()
        elif choice == "3":
            demo_keyboard()
        elif choice == "4":
            demo_step_by_step()
            demo_pygame()
            demo_keyboard()
        else:
            print("Invalid choice. Running pygame demo.")
            demo_pygame()

    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
