"""
Utility Functions for DQN Training
===================================

Replay buffer, epsilon decay strategies, and helper functions for DQN training.

Author: Meftun
Date: December 2025
Course: Planning and Decision Making
"""

import numpy as np
import random
from collections import deque
import torch


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    
    Stores (state, action, reward, next_state, done) tuples and samples
    batches for training.
    """
    
    def __init__(self, capacity=10000):
        """
        Initialize replay buffer.
        
        Parameters:
        -----------
        capacity : int, optional (default=10000)
            Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Add experience to buffer.
        
        Parameters:
        -----------
        state : np.ndarray
            Current state
            
        action : int
            Action taken
            
        reward : float
            Reward received
            
        next_state : np.ndarray
            Next state
            
        done : bool
            Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences.
        
        Parameters:
        -----------
        batch_size : int
            Number of experiences to sample
            
        Returns:
        --------
        tuple
            (states, actions, rewards, next_states, dones) as torch tensors
        """
        batch = random.sample(self.buffer, batch_size)
        
        # Convert to numpy arrays first (faster), then to tensors
        states = torch.FloatTensor(np.array([e[0] for e in batch]))
        actions = torch.LongTensor(np.array([e[1] for e in batch]))
        rewards = torch.FloatTensor(np.array([e[2] for e in batch]))
        next_states = torch.FloatTensor(np.array([e[3] for e in batch]))
        dones = torch.BoolTensor(np.array([e[4] for e in batch]))
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)


class EpsilonDecay:
    """
    Epsilon decay strategies for exploration.
    
    Assignment requirement: epsilon_min = 0.1 (fixed)
    """
    
    @staticmethod
    def linear_decay(epsilon, epsilon_min=0.1, epsilon_decay=0.995):
        """
        Linear decay: epsilon = max(epsilon_min, epsilon * decay_rate)
        
        Parameters:
        -----------
        epsilon : float
            Current epsilon value
            
        epsilon_min : float, optional (default=0.1)
            Minimum epsilon (fixed at 0.1 for assignment)
            
        epsilon_decay : float, optional (default=0.995)
            Decay rate per episode
            
        Returns:
        --------
        float
            Decayed epsilon value
        """
        return max(epsilon_min, epsilon * epsilon_decay)
    
    @staticmethod
    def exponential_decay(epsilon, epsilon_min=0.1, decay_rate=0.99):
        """
        Exponential decay: epsilon = max(epsilon_min, epsilon * decay_rate)
        
        Parameters:
        -----------
        epsilon : float
            Current epsilon value
            
        epsilon_min : float, optional (default=0.1)
            Minimum epsilon (fixed at 0.1 for assignment)
            
        decay_rate : float, optional (default=0.99)
            Decay rate per episode
            
        Returns:
        --------
        float
            Decayed epsilon value
        """
        return max(epsilon_min, epsilon * decay_rate)
    
    @staticmethod
    def step_decay(epsilon, episode, epsilon_min=0.1, 
                   initial_epsilon=1.0, decay_steps=500, decay_factor=0.5):
        """
        Step decay: epsilon decreases at specific episode milestones.
        
        Parameters:
        -----------
        epsilon : float
            Current epsilon value
            
        episode : int
            Current episode number
            
        epsilon_min : float, optional (default=0.1)
            Minimum epsilon (fixed at 0.1 for assignment)
            
        initial_epsilon : float, optional (default=1.0)
            Starting epsilon value
            
        decay_steps : int, optional (default=500)
            Episodes between decay steps
            
        decay_factor : float, optional (default=0.5)
            Factor to multiply epsilon by at each step
            
        Returns:
        --------
        float
            Decayed epsilon value
        """
        if episode == 0:
            return initial_epsilon
        
        steps = episode // decay_steps
        epsilon = initial_epsilon * (decay_factor ** steps)
        return max(epsilon_min, epsilon)


def compute_td_target(rewards, next_states, dones, target_network, gamma=0.99):
    """
    Compute TD target for DQN loss.
    
    target = reward + gamma * max(Q(s', a')) if not done
    target = reward if done
    
    Parameters:
    -----------
    rewards : torch.Tensor
        Batch of rewards
        
    next_states : torch.Tensor
        Batch of next states
        
    dones : torch.Tensor
        Batch of done flags
        
    target_network : DQN
        Target network for computing Q-values
        
    gamma : float, optional (default=0.99)
        Discount factor
        
    Returns:
    --------
    torch.Tensor
        TD targets
    """
    with torch.no_grad():
        next_q_values = target_network(next_states)
        max_next_q = torch.max(next_q_values, dim=1)[0]
        
        # TD target: reward + gamma * max(Q(s', a')) if not done, else just reward
        td_targets = rewards + (gamma * max_next_q * ~dones)
    
    return td_targets


def update_target_network(source_network, target_network, tau=1.0):
    """
    Update target network weights.
    
    Parameters:
    -----------
    source_network : DQN
        Source network (main network)
        
    target_network : DQN
        Target network to update
        
    tau : float, optional (default=1.0)
        Soft update coefficient (1.0 = hard update, <1.0 = soft update)
    """
    if tau == 1.0:
        # Hard update: copy all weights
        target_network.load_state_dict(source_network.state_dict())
    else:
        # Soft update: weighted average
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)


def save_checkpoint(model, optimizer, episode, filepath):
    """
    Save model checkpoint.
    
    Parameters:
    -----------
    model : DQN
        DQN model to save
        
    optimizer : torch.optim.Optimizer
        Optimizer state to save
        
    episode : int
        Current episode number
        
    filepath : str
        Path to save checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episode
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """
    Load model checkpoint.
    
    Parameters:
    -----------
    model : DQN
        DQN model to load into
        
    optimizer : torch.optim.Optimizer
        Optimizer to load into
        
    filepath : str
        Path to load checkpoint from
        
    Returns:
    --------
    int
        Episode number from checkpoint
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    episode = checkpoint['episode']
    print(f"Checkpoint loaded from {filepath}, starting from episode {episode}")
    return episode


