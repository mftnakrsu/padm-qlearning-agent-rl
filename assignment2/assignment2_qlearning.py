"""
Q-Learning Agent for PADM Assignment 2
======================================

This module implements a Q-learning agent that learns to navigate the custom
Grid World environment from Assignment 1.

Features:
- Q-learning with epsilon-greedy exploration
- Reverse sigmoid epsilon decay for smooth exploration-exploitation balance
- Q-table visualization with heatmaps
- Policy visualization with arrows
- Training curves

Author: Meftun
Date: December 2025
Course: Planning and Decision Making
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from chid_env import ChidEnv, create_env
except ImportError:
    # Fallback for when running from parent directory
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "assignment1"))
    from chid_env import ChidEnv, create_env
import os


def reverse_sigmoid_decay(t, epsilon_initial, epsilon_min, k, t0):
    """
    Reverse sigmoid function for epsilon decay (Logit Function).

    Provides smoother transition between exploration and exploitation
    compared to simple multiplicative decay.

    Parameters:
    -----------
    t : int
        Current episode number
    epsilon_initial : float
        Initial epsilon value
    epsilon_min : float
        Minimum epsilon value
    k : float
        Decay rate
    t0 : int
        Inflection point (episode where decay is fastest)

    Returns:
    --------
    float
        New epsilon value
    """
    return epsilon_min + (epsilon_initial - epsilon_min) / (1 + np.exp(k * (t - t0)))


class QLearningAgent:
    """
    Q-Learning Agent for Grid World Environment

    This agent learns the optimal policy using the Q-learning algorithm.
    It uses epsilon-greedy exploration strategy and updates Q-values using
    the Bellman equation.
    """

    def __init__(self, env, learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 use_reverse_sigmoid=True, sigmoid_k=0.01, sigmoid_t0=25):
        """
        Initialize the Q-learning agent.

        Parameters:
        -----------
        env : ChidEnv
            The environment to interact with
        learning_rate : float
            Learning rate (alpha) for Q-value updates
        discount_factor : float
            Discount factor (gamma) for future rewards
        epsilon : float
            Initial exploration rate
        epsilon_min : float
            Minimum exploration rate
        epsilon_decay : float
            Decay rate for epsilon
        use_reverse_sigmoid : bool
            Whether to use reverse sigmoid decay (True) or multiplicative decay (False)
        sigmoid_k : float
            K parameter for reverse sigmoid decay
        sigmoid_t0 : int
            t0 parameter (inflection point) for reverse sigmoid decay
        """
        self.env = env
        self.learning_rate = learning_rate  # alpha
        self.discount_factor = discount_factor  # gamma
        self.epsilon = epsilon
        self.epsilon_initial = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.use_reverse_sigmoid = use_reverse_sigmoid
        self.sigmoid_k = sigmoid_k
        self.sigmoid_t0 = sigmoid_t0

        # Initialize Q-table: shape = (num_rows, num_cols, has_lover, num_actions)
        # has_lover can be 0 or 1
        self.q_table = np.zeros((env.num_rows, env.num_cols, 2, env.action_space.n))

        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_success': [],
            'epsilon_history': []
        }

    def get_state_index(self, state):
        """Convert state array to tuple index for Q-table.
        State is (row, col, has_lover)."""
        return (int(state[0]), int(state[1]), int(state[2]))

    def choose_action(self, state, training=True):
        """
        Choose an action using epsilon-greedy policy.

        Parameters:
        -----------
        state : np.ndarray
            Current state
        training : bool
            If True, use epsilon-greedy. If False, use greedy policy.

        Returns:
        --------
        int
            Selected action
        """
        state_idx = self.get_state_index(state)

        if training and np.random.rand() < self.epsilon:
            # Exploration: choose random action
            return self.env.action_space.sample()
        else:
            # Exploitation: choose best action according to Q-table
            return int(np.argmax(self.q_table[state_idx]))

    def update_q_value(self, state, action, reward, next_state, done):
        """
        Update Q-value using the Q-learning update rule (Bellman equation).

        Q(s, a) = Q(s, a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s, a)]

        Parameters:
        -----------
        state : np.ndarray
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : np.ndarray
            Next state after taking action
        done : bool
            Whether episode ended
        """
        state_idx = self.get_state_index(state)
        next_state_idx = self.get_state_index(next_state)

        # Current Q-value
        current_q = self.q_table[state_idx][action]

        # Calculate target Q-value
        if done:
            # Terminal state: no future rewards
            target_q = reward
        else:
            # Non-terminal: add discounted future reward
            max_next_q = np.max(self.q_table[next_state_idx])
            target_q = reward + self.discount_factor * max_next_q

        # Q-learning update rule
        self.q_table[state_idx][action] = current_q + self.learning_rate * (target_q - current_q)

    def train(self, num_episodes=1000, verbose=True, save_frequency=100):
        """
        Train the Q-learning agent for specified number of episodes.

        Parameters:
        -----------
        num_episodes : int
            Number of training episodes
        verbose : bool
            Whether to print training progress
        save_frequency : int
            Save Q-table every N episodes

        Returns:
        --------
        dict
            Training statistics
        """
        # Reset epsilon for training
        self.epsilon = self.epsilon_initial

        # Reset training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_success': [],
            'epsilon_history': []
        }

        if verbose:
            print("=" * 70)
            print("Q-LEARNING TRAINING")
            print("=" * 70)
            print(f"Episodes: {num_episodes}")
            print(f"Learning Rate (alpha): {self.learning_rate}")
            print(f"Discount Factor (gamma): {self.discount_factor}")
            print(f"Initial Epsilon: {self.epsilon}")
            print(f"Epsilon Min: {self.epsilon_min}")
            print(f"Decay Method: {'Reverse Sigmoid' if self.use_reverse_sigmoid else 'Multiplicative'}")
            print(f"Environment: {self.env.num_rows}x{self.env.num_cols} grid")
            print(f"Actions: {self.env.action_space.n}")
            print("=" * 70)
            print()

        for episode in range(num_episodes):
            # Reset environment
            state, info = self.env.reset()
            total_reward = 0
            episode_length = 0
            done = False

            # Run episode
            while not done:
                # Choose action using epsilon-greedy
                action = self.choose_action(state, training=True)

                # Take action in environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Update Q-value using Q-learning rule
                self.update_q_value(state, action, reward, next_state, done)

                # Update statistics
                total_reward += reward
                episode_length += 1
                state = next_state

            # Decay epsilon
            if self.use_reverse_sigmoid:
                self.epsilon = reverse_sigmoid_decay(
                    episode, self.epsilon_initial, self.epsilon_min,
                    self.sigmoid_k, self.sigmoid_t0
                )
            else:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Record statistics
            self.training_stats['episode_rewards'].append(total_reward)
            self.training_stats['episode_lengths'].append(episode_length)
            self.training_stats['episode_success'].append(info.get('reached_goal', False))
            self.training_stats['epsilon_history'].append(self.epsilon)

            # Print progress
            if verbose and (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.training_stats['episode_rewards'][-50:])
                success_rate = np.mean(self.training_stats['episode_success'][-50:]) * 100
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Avg Reward (last 50): {avg_reward:.2f} | "
                      f"Success Rate: {success_rate:.1f}% | "
                      f"Epsilon: {self.epsilon:.3f}")

            # Save Q-table periodically
            if (episode + 1) % save_frequency == 0:
                self.save_q_table(f"q_table_episode_{episode + 1}.npy")

        if verbose:
            print("\n" + "=" * 70)
            print("TRAINING COMPLETED")
            print("=" * 70)
            final_avg_reward = np.mean(self.training_stats['episode_rewards'][-100:])
            final_success_rate = np.mean(self.training_stats['episode_success'][-100:]) * 100
            print(f"Final Average Reward (last 100 episodes): {final_avg_reward:.2f}")
            print(f"Final Success Rate (last 100 episodes): {final_success_rate:.1f}%")
            print("=" * 70)

        return self.training_stats

    def test(self, num_episodes=10, render=False):
        """
        Test the trained agent (no exploration, only exploitation).

        Parameters:
        -----------
        num_episodes : int
            Number of test episodes
        render : bool
            Whether to render the environment

        Returns:
        --------
        dict
            Test statistics
        """
        test_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_success': []
        }

        print("=" * 70)
        print("TESTING TRAINED AGENT")
        print("=" * 70)

        for episode in range(num_episodes):
            state, info = self.env.reset()
            total_reward = 0
            episode_length = 0
            done = False

            while not done:
                # Use greedy policy (no exploration)
                action = self.choose_action(state, training=False)

                # Take action
                next_state, reward, terminated, truncated, info = self.env.step(action)

                if render:
                    self.env.render()

                total_reward += reward
                episode_length += 1
                state = next_state
                done = terminated or truncated

            test_stats['episode_rewards'].append(total_reward)
            test_stats['episode_lengths'].append(episode_length)
            test_stats['episode_success'].append(info.get('reached_goal', False))

            success = "SUCCESS" if info.get('reached_goal', False) else "FAILED"
            print(f"Episode {episode + 1}: {success} | "
                  f"Reward: {total_reward:.2f} | "
                  f"Steps: {episode_length}")

        print("=" * 70)
        avg_reward = np.mean(test_stats['episode_rewards'])
        success_rate = np.mean(test_stats['episode_success']) * 100
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Success Rate: {success_rate:.1f}%")
        print("=" * 70)

        return test_stats

    def save_q_table(self, filepath="q_table.npy"):
        """Save Q-table to file."""
        np.save(filepath, self.q_table)
        print(f"Q-table saved to {filepath}")

    def load_q_table(self, filepath="q_table.npy"):
        """Load Q-table from file."""
        self.q_table = np.load(filepath)
        print(f"Q-table loaded from {filepath}")


def visualize_q_table(q_table, env, save_path=None, show_plot=True, has_lover=0):
    """
    Visualize Q-table using heatmaps for each action.

    Shows Q-values as heatmaps and marks special states (goal, danger, obstacles).

    Parameters:
    -----------
    q_table : np.ndarray
        Q-table of shape (num_rows, num_cols, 2, num_actions)
    env : ChidEnv
        Environment instance
    save_path : str, optional
        Path to save the visualization
    show_plot : bool
        Whether to display the plot
    has_lover : int
        0 or 1, which lover state to visualize
    """
    num_actions = env.action_space.n
    action_names = ["Up", "Down", "Right"][:num_actions]

    # Get special state coordinates
    goal_coords = [tuple(g) for g in env.goal_states]
    danger_coords = [tuple(d) for d in env.danger_states]
    obstacle_coords = [tuple(o) for o in env.obstacle_states]
    lover_coord = tuple(env.lover_state) if env.lover_state is not None else None

    # Create subplots for each action
    lover_label = "with Lover" if has_lover else "without Lover"
    fig, axes = plt.subplots(1, num_actions, figsize=(18, 5))
    fig.suptitle(f'Q-Table Visualization ({lover_label})', fontsize=14, fontweight='bold')

    for i, action_name in enumerate(action_names):
        ax = axes[i]

        # Extract Q-values for this action (for specific has_lover state)
        heatmap_data = q_table[:, :, has_lover, i].copy()

        # Create mask for special states
        mask = np.zeros_like(heatmap_data, dtype=bool)
        for goal in goal_coords:
            mask[goal] = True
        for danger in danger_coords:
            mask[danger] = True
        for obstacle in obstacle_coords:
            mask[obstacle] = True

        # Create heatmap
        sns.heatmap(heatmap_data,
                   annot=True,
                   fmt=".1f",
                   cmap="viridis",
                   ax=ax,
                   cbar=True,
                   mask=mask,
                   annot_kws={"size": 6},
                   vmin=np.min(q_table),
                   vmax=np.max(q_table))

        ax.set_facecolor("black")

        # Mark goal states
        for goal in goal_coords:
            ax.text(goal[1] + 0.5, goal[0] + 0.5, 'G',
                   color='green', ha='center', va='center',
                   weight='bold', fontsize=12)

        # Mark danger states
        for danger in danger_coords:
            ax.text(danger[1] + 0.5, danger[0] + 0.5, 'H',
                   color='red', ha='center', va='center',
                   weight='bold', fontsize=12)

        # Mark obstacles
        for obstacle in obstacle_coords:
            ax.text(obstacle[1] + 0.5, obstacle[0] + 0.5, 'O',
                   color='blue', ha='center', va='center',
                   weight='bold', fontsize=10)

        # Mark lover state
        if lover_coord and not mask[lover_coord]:
            ax.text(lover_coord[1] + 0.5, lover_coord[0] + 0.5, 'L',
                   color='pink', ha='center', va='center',
                   weight='bold', fontsize=12)

        ax.set_title(f'Q-values for Action: {action_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Q-table visualization saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_policy(q_table, env, save_path=None, show_plot=True, has_lover=0):
    """
    Visualize the learned policy with arrows showing optimal actions.

    Parameters:
    -----------
    q_table : np.ndarray
        Q-table of shape (num_rows, num_cols, 2, num_actions)
    env : ChidEnv
        Environment instance
    save_path : str, optional
        Path to save the visualization
    show_plot : bool
        Whether to display the plot
    has_lover : int
        0 or 1, which lover state to visualize
    """
    # Get policy (best action for each state)
    # q_table shape: (rows, cols, has_lover, actions)
    policy = np.argmax(q_table[:, :, has_lover, :], axis=2)
    nrows, ncols = policy.shape

    # Get special state coordinates
    goal_coords = [tuple(g) for g in env.goal_states]
    danger_coords = [tuple(d) for d in env.danger_states]
    obstacle_coords = [tuple(o) for o in env.obstacle_states]
    lover_coord = tuple(env.lover_state) if env.lover_state is not None else None

    # Create figure
    lover_label = "with Lover" if has_lover else "without Lover"
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(f'Policy Visualization ({lover_label})', fontsize=14, fontweight='bold')

    # Set limits
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)

    # Draw arrows for each state
    for i in range(nrows):
        for j in range(ncols):
            pos = (i, j)

            # Skip special states
            if pos in obstacle_coords or pos in danger_coords or pos in goal_coords:
                continue

            # Determine arrow direction based on policy
            action = policy[i, j]
            if action == 0:  # Up
                dx, dy = 0, -0.3
            elif action == 1:  # Down
                dx, dy = 0, 0.3
            elif action == 2:  # Right
                dx, dy = 0.3, 0
            else:
                dx, dy = 0, 0

            # Draw arrow
            ax.arrow(j + 0.5, i + 0.5, dx, dy,
                    head_width=0.2, head_length=0.1,
                    fc='blue', ec='blue')

    # Mark goal states
    for goal in goal_coords:
        ax.text(goal[1] + 0.5, goal[0] + 0.5, 'G',
               color='green', ha='center', va='center',
               weight='bold', fontsize=14,
               bbox=dict(boxstyle='circle', facecolor='lightgreen', edgecolor='green'))

    # Mark danger states
    for danger in danger_coords:
        ax.text(danger[1] + 0.5, danger[0] + 0.5, 'H',
               color='red', ha='center', va='center',
               weight='bold', fontsize=14,
               bbox=dict(boxstyle='circle', facecolor='lightcoral', edgecolor='red'))

    # Mark obstacles
    for obstacle in obstacle_coords:
        ax.add_patch(plt.Rectangle((obstacle[1], obstacle[0]), 1, 1,
                                   facecolor='gray', edgecolor='black'))
        ax.text(obstacle[1] + 0.5, obstacle[0] + 0.5, 'O',
               color='white', ha='center', va='center',
               weight='bold', fontsize=10)

    # Mark lover state
    if lover_coord:
        ax.text(lover_coord[1] + 0.5, lover_coord[0] + 0.5, 'L',
               color='hotpink', ha='center', va='center',
               weight='bold', fontsize=14,
               bbox=dict(boxstyle='circle', facecolor='pink', edgecolor='hotpink'))

    # Mark start state
    start = tuple(env.agent_start)
    ax.text(start[1] + 0.5, start[0] + 0.5, 'S',
           color='orange', ha='center', va='center',
           weight='bold', fontsize=14,
           bbox=dict(boxstyle='circle', facecolor='lightyellow', edgecolor='orange'))

    # Draw grid
    ax.set_xticks(np.arange(ncols + 1))
    ax.set_yticks(np.arange(nrows + 1))
    ax.grid(which='major', color='black', linestyle='-', linewidth=1)
    ax.invert_yaxis()

    ax.set_title('Learned Policy (Arrows show best action)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # Add legend
    legend_text = "Actions: Up=^, Down=v, Right=>"
    ax.text(0.02, -0.08, legend_text, transform=ax.transAxes, fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Policy visualization saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_training_curves(training_stats, save_path=None, show_plot=True):
    """
    Plot training curves (rewards, success rate, epsilon).

    Parameters:
    -----------
    training_stats : dict
        Training statistics from agent.train()
    save_path : str, optional
        Path to save the plot
    show_plot : bool
        Whether to display the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    episodes = range(1, len(training_stats['episode_rewards']) + 1)

    # Plot 1: Episode Rewards
    axes[0].plot(episodes, training_stats['episode_rewards'], alpha=0.6, linewidth=0.5)
    # Moving average
    window = min(50, len(training_stats['episode_rewards']) // 10)
    if window > 1:
        moving_avg = np.convolve(training_stats['episode_rewards'],
                                np.ones(window)/window, mode='valid')
        axes[0].plot(range(window, len(episodes) + 1), moving_avg,
                   color='red', linewidth=2, label=f'Moving Average (window={window})')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Reward')
    axes[0].set_title('Training: Episode Rewards', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Success Rate
    success_rate = np.array(training_stats['episode_success'], dtype=float) * 100
    axes[1].plot(episodes, success_rate, alpha=0.6, linewidth=0.5, color='green')
    # Moving average
    if window > 1:
        moving_avg = np.convolve(success_rate,
                                np.ones(window)/window, mode='valid')
        axes[1].plot(range(window, len(episodes) + 1), moving_avg,
                   color='darkgreen', linewidth=2, label=f'Moving Average (window={window})')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Success Rate (%)')
    axes[1].set_title('Training: Success Rate', fontweight='bold')
    axes[1].set_ylim([0, 105])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot 3: Epsilon (Exploration Rate)
    axes[2].plot(episodes, training_stats['epsilon_history'], color='orange', linewidth=2)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Epsilon (Exploration Rate)')
    axes[2].set_title('Training: Exploration Rate Decay', fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def train_with_hyperparameters(env, hyperparams_list, num_episodes=500):
    """
    Train agent with multiple hyperparameter configurations.

    Parameters:
    -----------
    env : ChidEnv
        Environment instance
    hyperparams_list : list of dict
        List of hyperparameter dictionaries
    num_episodes : int
        Number of episodes per configuration

    Returns:
    --------
    list
        List of trained agents
    """
    trained_agents = []

    print("=" * 70)
    print("TRAINING WITH MULTIPLE HYPERPARAMETER CONFIGURATIONS")
    print("=" * 70)

    for i, hyperparams in enumerate(hyperparams_list):
        print(f"\n{'='*70}")
        print(f"Configuration {i+1}/{len(hyperparams_list)}")
        print(f"{'='*70}")
        print(f"Learning Rate: {hyperparams['learning_rate']}")
        print(f"Discount Factor: {hyperparams['discount_factor']}")
        print(f"Epsilon Decay: {hyperparams.get('epsilon_decay', 'reverse sigmoid')}")
        print()

        # Create agent with these hyperparameters
        agent = QLearningAgent(env, **hyperparams)

        # Train agent
        agent.train(num_episodes=num_episodes, verbose=True)

        # Save Q-table with descriptive name
        config_name = f"lr{hyperparams['learning_rate']}_gamma{hyperparams['discount_factor']}"
        q_table_path = f"q_table_{config_name}.npy"
        agent.save_q_table(q_table_path)

        # Visualize Q-table
        viz_path = f"q_table_visualization_{config_name}.png"
        visualize_q_table(agent.q_table, env, save_path=viz_path, show_plot=False)

        # Visualize policy
        policy_path = f"policy_visualization_{config_name}.png"
        visualize_policy(agent.q_table, env, save_path=policy_path, show_plot=False)

        # Plot training curves
        curves_path = f"training_curves_{config_name}.png"
        plot_training_curves(agent.training_stats, save_path=curves_path, show_plot=False)

        trained_agents.append(agent)

    return trained_agents


if __name__ == "__main__":
    # Configuration
    NUM_EPISODES = 150

    # Hyperparameters
    LEARNING_RATE = 0.08
    DISCOUNT_FACTOR = 0.995
    EPSILON = 1.0
    EPSILON_MIN = 0.1
    EPSILON_DECAY = 0.01  # For reverse sigmoid
    SIGMOID_T0 = 25  # Inflection point

    # Create environment (no pygame rendering during training)
    env = create_env(render_mode=None)

    print("=" * 70)
    print("ASSIGNMENT 2: Q-LEARNING AGENT")
    print("=" * 70)
    print(f"Environment: {env.num_rows}x{env.num_cols} grid")
    print(f"Obstacles: {len(env.obstacle_states)}")
    print(f"Danger States: {len(env.danger_states)}")
    print(f"Goal States: {len(env.goal_states)}")
    print(f"Reward States: {len(env.reward_states)}")
    print(f"Actions: {env.action_space.n} (Up, Down, Right)")
    print("=" * 70)

    # Create and train agent
    agent = QLearningAgent(env,
                          learning_rate=LEARNING_RATE,
                          discount_factor=DISCOUNT_FACTOR,
                          epsilon=EPSILON,
                          epsilon_min=EPSILON_MIN,
                          use_reverse_sigmoid=True,
                          sigmoid_k=EPSILON_DECAY,
                          sigmoid_t0=SIGMOID_T0)

    # Train agent
    training_stats = agent.train(num_episodes=NUM_EPISODES, verbose=True)

    # Save final Q-table
    agent.save_q_table("q_table_final.npy")

    # Visualize Q-table
    print("\nGenerating Q-table visualization...")
    visualize_q_table(agent.q_table, env, save_path="q_table_visualization.png", show_plot=True)

    # Visualize policy
    print("\nGenerating policy visualization...")
    visualize_policy(agent.q_table, env, save_path="policy_visualization.png", show_plot=True)

    # Plot training curves
    print("\nGenerating training curves...")
    plot_training_curves(training_stats, save_path="training_curves.png", show_plot=True)

    # Test trained agent
    agent.test(num_episodes=10, render=False)

    print("\nQ-learning training and evaluation completed!")
