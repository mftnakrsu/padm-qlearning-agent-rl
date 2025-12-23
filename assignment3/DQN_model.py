"""
DQN Model for Assignment 3
===========================

Deep Q-Network (DQN) implementation using PyTorch for continuous maze navigation.

Author: Meftun
Date: December 2025
Course: Planning and Decision Making
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Deep Q-Network for continuous state space (2D position).
    
    Input: 2D continuous state [x, y] in [0, 1]^2
    Output: Q-values for 4 discrete actions [up, down, left, right]
    """
    
    def __init__(self, state_dim=2, action_dim=4, hidden_dims=[128, 128, 64]):
        """
        Initialize DQN network.
        
        Parameters:
        -----------
        state_dim : int, optional (default=2)
            Dimension of state space (x, y coordinates)
            
        action_dim : int, optional (default=4)
            Number of discrete actions
            
        hidden_dims : list, optional (default=[128, 128, 64])
            Dimensions of hidden layers
        """
        super(DQN, self).__init__()
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer (Q-values for each action)
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        state : torch.Tensor
            State tensor of shape (batch_size, state_dim) or (state_dim,)
            
        Returns:
        --------
        torch.Tensor
            Q-values for each action, shape (batch_size, action_dim) or (action_dim,)
        """
        # Handle single state (add batch dimension)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        q_values = self.network(state)
        
        # Remove batch dimension if input was single state
        if state.shape[0] == 1 and state.dim() == 2:
            q_values = q_values.squeeze(0)
        
        return q_values
    
    def get_action(self, state, epsilon=0.0):
        """
        Get action using epsilon-greedy policy.
        
        Parameters:
        -----------
        state : torch.Tensor or np.ndarray
            Current state
            
        epsilon : float, optional (default=0.0)
            Exploration probability (0.0 = greedy, 1.0 = random)
            
        Returns:
        --------
        int
            Selected action
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        # Random action (exploration)
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, 4, (1,)).item()
        
        # Greedy action (exploitation)
        self.eval()
        with torch.no_grad():
            q_values = self.forward(state)
            action = torch.argmax(q_values).item()
        
        return action


# Import numpy for type checking
import numpy as np


