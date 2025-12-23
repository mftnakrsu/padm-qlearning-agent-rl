"""
Main Training Script for Assignment 3: DQN Agent
================================================

Trains a DQN agent on the continuous maze environment.

Requirements:
- Epsilon decays to 0.1 (fixed minimum)
- After epsilon reaches 0.1, agent must reach goal for 100 consecutive episodes
- Test starting point: (0.1, 0.5)

Author: Meftun
Date: December 2025
Course: Planning and Decision Making
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque

from env import ContinuousMazeEnv
from DQN_model import DQN
from utils import ReplayBuffer, EpsilonDecay, compute_td_target, update_target_network


class DQNAgent:
    """
    DQN Agent with experience replay and target network.
    """
    
    def __init__(self, state_dim=2, action_dim=4, lr=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995,
                 batch_size=64, buffer_size=10000, target_update_freq=100,
                 device='cpu'):
        """
        Initialize DQN agent.
        
        Parameters:
        -----------
        state_dim : int, optional (default=2)
            State dimension (x, y coordinates)
            
        action_dim : int, optional (default=4)
            Number of actions
            
        lr : float, optional (default=0.001)
            Learning rate
            
        gamma : float, optional (default=0.99)
            Discount factor
            
        epsilon : float, optional (default=1.0)
            Initial exploration rate
            
        epsilon_min : float, optional (default=0.1)
            Minimum exploration rate (FIXED at 0.1 for assignment)
            
        epsilon_decay : float, optional (default=0.995)
            Epsilon decay rate
            
        batch_size : int, optional (default=64)
            Batch size for training
            
        buffer_size : int, optional (default=10000)
            Replay buffer size
            
        target_update_freq : int, optional (default=100)
            Frequency of target network updates (in steps)
            
        device : str, optional (default='cpu')
            Device to use ('cpu' or 'cuda')
        """
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Networks
        self.q_network = DQN(state_dim, action_dim).to(device)
        self.target_network = DQN(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Epsilon (exploration)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min  # FIXED at 0.1 for assignment
        self.epsilon_decay = epsilon_decay
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_success': [],
            'epsilon_history': [],
            'loss_history': []
        }
        
        self.step_count = 0
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.
        
        Parameters:
        -----------
        state : np.ndarray
            Current state
            
        training : bool, optional (default=True)
            Whether in training mode (uses epsilon if True)
            
        Returns:
        --------
        int
            Selected action
        """
        if training and np.random.rand() < self.epsilon:
            # Exploration: random action
            return np.random.randint(0, 4)
        else:
            # Exploitation: greedy action
            state_tensor = torch.FloatTensor(state).to(self.device)
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = torch.argmax(q_values).item()
            return action
    
    def train_step(self):
        """
        Perform one training step using experience replay.
        
        Returns:
        --------
        float or None
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute TD targets
        td_targets = compute_td_target(rewards, next_states, dones, self.target_network, self.gamma)
        
        # Compute loss
        loss = F.mse_loss(current_q_values, td_targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network periodically
        if self.step_count % self.target_update_freq == 0:
            update_target_network(self.q_network, self.target_network, tau=1.0)
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay epsilon using linear decay strategy."""
        self.epsilon = EpsilonDecay.linear_decay(
            self.epsilon, 
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay
        )


def train_dqn(env, agent, num_episodes=5000, max_steps_per_episode=500,
              learning_starts=1000, verbose=True, save_frequency=500):
    """
    Train DQN agent.
    
    Parameters:
    -----------
    env : ContinuousMazeEnv
        Environment instance
        
    agent : DQNAgent
        DQN agent instance
        
    num_episodes : int, optional (default=5000)
        Maximum number of training episodes
        
    max_steps_per_episode : int, optional (default=500)
        Maximum steps per episode
        
    learning_starts : int, optional (default=1000)
        Number of steps before starting training
        
    verbose : bool, optional (default=True)
        Whether to print training progress
        
    save_frequency : int, optional (default=500)
        Save model every N episodes
        
    Returns:
    --------
    dict
        Training statistics
    """
    consecutive_successes = 0
    max_consecutive_successes = 0
    epsilon_reached_min = False
    training_complete = False
    
    if verbose:
        print("=" * 70)
        print("DQN TRAINING")
        print("=" * 70)
        print(f"Episodes: {num_episodes}")
        print(f"Learning Rate: {agent.optimizer.param_groups[0]['lr']}")
        print(f"Discount Factor: {agent.gamma}")
        print(f"Initial Epsilon: {agent.epsilon}")
        print(f"Epsilon Min: {agent.epsilon_min} (FIXED)")
        print(f"Epsilon Decay: {agent.epsilon_decay}")
        print(f"Batch Size: {agent.batch_size}")
        print("=" * 70)
        print()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_length = 0
        done = False
        episode_losses = []
        
        while not done and episode_length < max_steps_per_episode:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take step in environment
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store experience in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Increment step count (for learning_starts check)
            agent.step_count += 1
            
            # Train agent (after learning_starts steps)
            if agent.step_count >= learning_starts:
                loss = agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            episode_length += 1
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Check if epsilon reached minimum
        if not epsilon_reached_min and agent.epsilon <= agent.epsilon_min:
            epsilon_reached_min = True
            if verbose:
                print(f"\n{'='*70}")
                print(f"✅ Epsilon reached minimum (0.1) at episode {episode + 1}")
                print(f"{'='*70}\n")
        
        # Check if goal was reached
        reached_goal = done and total_reward > 0  # Positive reward indicates goal
        
        # Track consecutive successes (only after epsilon reached min)
        if epsilon_reached_min:
            if reached_goal:
                consecutive_successes += 1
                max_consecutive_successes = max(max_consecutive_successes, consecutive_successes)
            else:
                consecutive_successes = 0
            
            # Check if training is complete (100 consecutive successes)
            if consecutive_successes >= 100:
                training_complete = True
                if verbose:
                    print(f"\n{'='*70}")
                    print(f"🎉 TRAINING COMPLETE!")
                    print(f"✅ 100 consecutive successes achieved at episode {episode + 1}")
                    print(f"{'='*70}\n")
        
        # Record statistics
        agent.training_stats['episode_rewards'].append(total_reward)
        agent.training_stats['episode_lengths'].append(episode_length)
        agent.training_stats['episode_success'].append(reached_goal)
        agent.training_stats['epsilon_history'].append(agent.epsilon)
        if episode_losses:
            agent.training_stats['loss_history'].append(np.mean(episode_losses))
        else:
            agent.training_stats['loss_history'].append(0.0)
        
        # Print progress
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(agent.training_stats['episode_rewards'][-100:])
            success_rate = np.mean(agent.training_stats['episode_success'][-100:]) * 100
            # Calculate average loss (only for episodes where training occurred)
            recent_losses = [l for l in agent.training_stats['loss_history'][-100:] if l > 0]
            avg_loss = np.mean(recent_losses) if recent_losses else 0.0
            
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Success Rate: {success_rate:.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Avg Loss: {avg_loss:.4f}", end="")
            
            if epsilon_reached_min:
                print(f" | Consecutive Successes: {consecutive_successes}/100")
            else:
                print()
        
        # Save model periodically
        if (episode + 1) % save_frequency == 0:
            torch.save(agent.q_network.state_dict(), f"dqn_checkpoint_ep{episode + 1}.pth")
        
        # Early stopping if training complete
        if training_complete:
            break
    
    # Save final model
    torch.save(agent.q_network.state_dict(), "dqn_final.pth")
    if verbose:
        print(f"\n✅ Final model saved to dqn_final.pth")
        print(f"Max consecutive successes: {max_consecutive_successes}")
    
    return agent.training_stats


def plot_training_curves(stats, save_path="training_curves.png"):
    """
    Plot training curves.
    
    Parameters:
    -----------
    stats : dict
        Training statistics
        
    save_path : str, optional (default="training_curves.png")
        Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes = range(1, len(stats['episode_rewards']) + 1)
    
    # Plot 1: Episode Rewards
    axes[0, 0].plot(episodes, stats['episode_rewards'], alpha=0.6, linewidth=0.5)
    window = min(100, len(episodes) // 10)
    if window > 1:
        moving_avg = np.convolve(stats['episode_rewards'], 
                                np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window, len(episodes) + 1), moving_avg, 
                       color='red', linewidth=2, label=f'Moving Avg (window={window})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Episode Reward')
    axes[0, 0].set_title('Training: Episode Rewards', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Success Rate
    success_rate = np.array(stats['episode_success'], dtype=float) * 100
    axes[0, 1].plot(episodes, success_rate, alpha=0.6, linewidth=0.5, color='green')
    if window > 1:
        moving_avg = np.convolve(success_rate, 
                                np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window, len(episodes) + 1), moving_avg, 
                       color='darkgreen', linewidth=2, label=f'Moving Avg (window={window})')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Success Rate (%)')
    axes[0, 1].set_title('Training: Success Rate', fontweight='bold')
    axes[0, 1].set_ylim([0, 105])
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Epsilon Decay
    axes[1, 0].plot(episodes, stats['epsilon_history'], color='orange', linewidth=2)
    axes[1, 0].axhline(y=0.1, color='r', linestyle='--', label='Epsilon Min (0.1)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon (Exploration Rate)')
    axes[1, 0].set_title('Training: Epsilon Decay', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: Loss
    if stats['loss_history'] and any(l > 0 for l in stats['loss_history']):
        loss_episodes = [i for i, l in enumerate(stats['loss_history'], 1) if l > 0]
        losses = [l for l in stats['loss_history'] if l > 0]
        axes[1, 1].plot(loss_episodes, losses, alpha=0.6, linewidth=0.5, color='purple')
        if len(losses) > window:
            moving_avg = np.convolve(losses, 
                                    np.ones(window)/window, mode='valid')
            axes[1, 1].plot(loss_episodes[window-1:], moving_avg, 
                           color='darkviolet', linewidth=2, label=f'Moving Avg (window={window})')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Training: Loss', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'No loss data yet', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Training: Loss', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def test_agent(env, agent, start_pos=(0.1, 0.5), num_episodes=10, render=False):
    """
    Test trained agent from specified starting position.
    
    Parameters:
    -----------
    env : ContinuousMazeEnv
        Environment instance
        
    agent : DQNAgent
        Trained DQN agent
        
    start_pos : tuple, optional (default=(0.1, 0.5))
        Starting position (x, y) for testing
        
    num_episodes : int, optional (default=10)
        Number of test episodes
        
    render : bool, optional (default=False)
        Whether to render environment
        
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
    print(f"Starting position: {start_pos}")
    print("=" * 70)
    
    for episode in range(num_episodes):
        # Reset environment (default is (0.1, 0.5), which matches requirement)
        state, _ = env.reset()
        total_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < 500:
            # Use greedy policy (no exploration)
            action = agent.select_action(state, training=False)
            
            # Take step
            next_state, reward, done, truncated, _ = env.step(action)
            
            if render:
                env.render()
            
            state = next_state
            total_reward += reward
            episode_length += 1
            done = done or truncated
        
        reached_goal = done and total_reward > 0
        test_stats['episode_rewards'].append(total_reward)
        test_stats['episode_lengths'].append(episode_length)
        test_stats['episode_success'].append(reached_goal)
        
        success = "✅ SUCCESS" if reached_goal else "❌ FAILED"
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


if __name__ == "__main__":
    # ========================================================================
    # HYPERPARAMETERS (Final tuned values)
    # ========================================================================
    
    # Training parameters
    NUM_EPISODES = 5000
    MAX_STEPS_PER_EPISODE = 500
    LEARNING_STARTS = 500  # Start training after collecting this many experiences (reduced for faster start)
    
    # DQN hyperparameters
    LEARNING_RATE = 0.001
    GAMMA = 0.99
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000
    TARGET_UPDATE_FREQ = 100  # Update target network every N steps
    
    # Epsilon parameters (epsilon_min FIXED at 0.1 for assignment)
    EPSILON = 1.0
    EPSILON_MIN = 0.1  # FIXED for assignment
    EPSILON_DECAY = 0.995
    
    # Device
    DEVICE = 'cpu'  # CPU training (as mentioned in assignment)
    
    # ========================================================================
    # CREATE ENVIRONMENT AND AGENT
    # ========================================================================
    
    print("=" * 70)
    print("ASSIGNMENT 3: DQN AGENT TRAINING")
    print("=" * 70)
    
    env = ContinuousMazeEnv(render_mode=None)  # No rendering during training
    
    agent = DQNAgent(
        state_dim=2,
        action_dim=4,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        buffer_size=BUFFER_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        device=DEVICE
    )
    
    # ========================================================================
    # TRAIN AGENT
    # ========================================================================
    
    training_stats = train_dqn(
        env=env,
        agent=agent,
        num_episodes=NUM_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        learning_starts=LEARNING_STARTS,
        verbose=True,
        save_frequency=500
    )
    
    # ========================================================================
    # PLOT TRAINING CURVES
    # ========================================================================
    
    plot_training_curves(training_stats, save_path="training_curves.png")
    
    # ========================================================================
    # TEST AGENT
    # ========================================================================
    
    # Test from starting position (0.1, 0.5) - this is the default reset position
    test_stats = test_agent(env, agent, start_pos=(0.1, 0.5), num_episodes=10, render=False)
    
    print("\n✅ Training and testing completed!")
    print("Files generated:")
    print("  - dqn_final.pth (final trained model)")
    print("  - training_curves.png (training statistics)")


