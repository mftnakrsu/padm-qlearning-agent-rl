"""
Main Script for Assignment 2: Q-Learning Agent
===============================================

This script trains and evaluates Q-learning agents on the custom Grid World environment.
The environment is a 7x12 maze with obstacles, danger zones, and reward states.

Author: Meftun
Date: December 2025
Course: Planning and Decision Making
"""

try:
    from chid_env import ChidEnv, create_env
except ImportError:
    # Fallback for when running from parent directory
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "assignment1"))
    from chid_env import ChidEnv, create_env
from assignment2_qlearning import (
    QLearningAgent,
    visualize_q_table,
    visualize_policy,
    plot_training_curves,
    train_with_hyperparameters
)
import numpy as np


def main():
    """
    Main function to run Q-learning training and evaluation.
    """
    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    # Training parameters
    NUM_EPISODES = 500  # More episodes for larger state space (7x12x2)

    # Hyperparameters - Optimized configuration
    DEFAULT_HYPERPARAMS = {
        'learning_rate': 0.1,
        'discount_factor': 0.99,
        'epsilon': 1.0,
        'epsilon_min': 0.05,
        'use_reverse_sigmoid': True,
        'sigmoid_k': 0.008,
        'sigmoid_t0': 300
    }

    # Multiple hyperparameter configurations for comparison
    HYPERPARAMS_LIST = [
        {
            'learning_rate': 0.08,
            'discount_factor': 0.995,
            'epsilon': 1.0,
            'epsilon_min': 0.1,
            'use_reverse_sigmoid': True,
            'sigmoid_k': 0.01,
            'sigmoid_t0': 25
        },
        {
            'learning_rate': 0.1,
            'discount_factor': 0.99,
            'epsilon': 1.0,
            'epsilon_min': 0.05,
            'use_reverse_sigmoid': False,
            'epsilon_decay': 0.995
        },
        {
            'learning_rate': 0.05,
            'discount_factor': 0.99,
            'epsilon': 1.0,
            'epsilon_min': 0.1,
            'use_reverse_sigmoid': True,
            'sigmoid_k': 0.02,
            'sigmoid_t0': 30
        }
    ]

    # ========================================================================
    # CREATE ENVIRONMENT
    # ========================================================================

    print("=" * 70)
    print("ASSIGNMENT 2: Q-LEARNING AGENT")
    print("=" * 70)

    env = create_env(render_mode=None)

    print(f"Grid Size: {env.num_rows}x{env.num_cols}")
    print(f"Obstacles: {len(env.obstacle_states)}")
    print(f"Danger States (Hell): {len(env.danger_states)}")
    print(f"Reward States: {len(env.reward_states)}")
    print(f"Goal States: {len(env.goal_states)}")
    print(f"Lover State: {env.lover_state}")
    print(f"Agent Start: {env.agent_start}")
    print(f"Actions: {env.action_space.n} (Up, Down, Right)")
    print("=" * 70)
    print()

    # ========================================================================
    # TRAINING OPTIONS
    # ========================================================================

    print("Select training mode:")
    print("1: Single training run (default hyperparameters)")
    print("2: Multiple hyperparameter configurations")
    print("3: Custom hyperparameters")

    try:
        choice = input("\nYour choice (1/2/3, default=1): ").strip() or "1"
    except KeyboardInterrupt:
        print("\nExiting...")
        return

    # ========================================================================
    # OPTION 1: SINGLE TRAINING RUN
    # ========================================================================

    if choice == "1":
        print("\n" + "=" * 70)
        print("TRAINING WITH DEFAULT HYPERPARAMETERS")
        print("=" * 70)

        # Create agent
        agent = QLearningAgent(env, **DEFAULT_HYPERPARAMS)

        # Train agent
        training_stats = agent.train(num_episodes=NUM_EPISODES, verbose=True)

        # Save Q-table
        agent.save_q_table("q_table_final.npy")

        # Visualize Q-table
        print("\nGenerating Q-table visualization...")
        visualize_q_table(
            agent.q_table,
            env,
            save_path="q_table_visualization.png",
            show_plot=True
        )

        # Visualize policy
        print("\nGenerating policy visualization...")
        visualize_policy(
            agent.q_table,
            env,
            save_path="policy_visualization.png",
            show_plot=True
        )

        # Plot training curves
        print("\nGenerating training curves...")
        plot_training_curves(
            training_stats,
            save_path="training_curves.png",
            show_plot=True
        )

        # Test agent
        print("\n" + "=" * 70)
        agent.test(num_episodes=10, render=False)

    # ========================================================================
    # OPTION 2: MULTIPLE HYPERPARAMETER CONFIGURATIONS
    # ========================================================================

    elif choice == "2":
        print("\n" + "=" * 70)
        print("TRAINING WITH MULTIPLE HYPERPARAMETER CONFIGURATIONS")
        print("=" * 70)

        trained_agents = train_with_hyperparameters(
            env,
            HYPERPARAMS_LIST,
            num_episodes=NUM_EPISODES
        )

        # Compare results
        print("\n" + "=" * 70)
        print("COMPARISON OF HYPERPARAMETER CONFIGURATIONS")
        print("=" * 70)

        for i, (agent, hyperparams) in enumerate(zip(trained_agents, HYPERPARAMS_LIST)):
            final_avg_reward = np.mean(agent.training_stats['episode_rewards'][-50:])
            final_success_rate = np.mean(agent.training_stats['episode_success'][-50:]) * 100

            print(f"\nConfiguration {i+1}:")
            print(f"  Learning Rate: {hyperparams['learning_rate']}")
            print(f"  Discount Factor: {hyperparams['discount_factor']}")
            print(f"  Epsilon Decay: {'Reverse Sigmoid' if hyperparams.get('use_reverse_sigmoid', False) else 'Multiplicative'}")
            print(f"  Final Avg Reward (last 50): {final_avg_reward:.2f}")
            print(f"  Final Success Rate (last 50): {final_success_rate:.1f}%")

        print("\n" + "=" * 70)

    # ========================================================================
    # OPTION 3: CUSTOM HYPERPARAMETERS
    # ========================================================================

    elif choice == "3":
        print("\n" + "=" * 70)
        print("CUSTOM HYPERPARAMETERS")
        print("=" * 70)

        try:
            learning_rate = float(input("Learning Rate (alpha) [default=0.08]: ") or "0.08")
            discount_factor = float(input("Discount Factor (gamma) [default=0.995]: ") or "0.995")
            epsilon = float(input("Initial Epsilon [default=1.0]: ") or "1.0")
            epsilon_min = float(input("Epsilon Min [default=0.1]: ") or "0.1")
            use_sigmoid = input("Use Reverse Sigmoid Decay? (y/n) [default=y]: ").lower() or "y"
            num_episodes = int(input(f"Number of Episodes [default={NUM_EPISODES}]: ") or str(NUM_EPISODES))

            custom_hyperparams = {
                'learning_rate': learning_rate,
                'discount_factor': discount_factor,
                'epsilon': epsilon,
                'epsilon_min': epsilon_min,
                'use_reverse_sigmoid': use_sigmoid == 'y',
            }

            if use_sigmoid == 'y':
                custom_hyperparams['sigmoid_k'] = float(input("Sigmoid K [default=0.01]: ") or "0.01")
                custom_hyperparams['sigmoid_t0'] = int(input("Sigmoid T0 [default=25]: ") or "25")
            else:
                custom_hyperparams['epsilon_decay'] = float(input("Epsilon Decay [default=0.995]: ") or "0.995")

            # Create and train agent
            agent = QLearningAgent(env, **custom_hyperparams)
            training_stats = agent.train(num_episodes=num_episodes, verbose=True)

            # Save Q-table
            config_name = f"custom_lr{learning_rate}_gamma{discount_factor}"
            agent.save_q_table(f"q_table_{config_name}.npy")

            # Visualize
            visualize_q_table(
                agent.q_table,
                env,
                save_path=f"q_table_visualization_{config_name}.png",
                show_plot=True
            )

            visualize_policy(
                agent.q_table,
                env,
                save_path=f"policy_visualization_{config_name}.png",
                show_plot=True
            )

            plot_training_curves(
                training_stats,
                save_path=f"training_curves_{config_name}.png",
                show_plot=True
            )

            # Test
            agent.test(num_episodes=10, render=False)

        except (ValueError, KeyboardInterrupt) as e:
            print(f"\nError: {e}")
            print("Exiting...")
            return

    else:
        print("Invalid choice! Using default (Option 1)...")
        agent = QLearningAgent(env, **DEFAULT_HYPERPARAMS)
        training_stats = agent.train(num_episodes=NUM_EPISODES, verbose=True)
        agent.save_q_table("q_table_final.npy")
        visualize_q_table(agent.q_table, env, save_path="q_table_visualization.png", show_plot=True)
        visualize_policy(agent.q_table, env, save_path="policy_visualization.png", show_plot=True)
        plot_training_curves(training_stats, save_path="training_curves.png", show_plot=True)
        agent.test(num_episodes=10, render=False)

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print("\n" + "=" * 70)
    print("ASSIGNMENT 2 COMPLETED!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - Q-table files (*.npy)")
    print("  - Q-table visualizations (*.png)")
    print("  - Policy visualizations (*.png)")
    print("  - Training curves (*.png)")
    print("\nThese files are ready for submission.")
    print("=" * 70)


if __name__ == "__main__":
    main()
