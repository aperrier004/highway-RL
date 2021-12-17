# Author : Alban PERRIER (https://github.com/aperrier004/highway-RL)
# Conteq_valuest : RL project at Bordeauq_values INP (ENSC - IA)
# Goal : Implementation of DQN on the highway env (https://github.com/eleurent/highway-env)

import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    """
    A class used to create a Deep Q Network

    Parameters
    ----------
    nn : Tensor
        The neural network

    Attributes
    ----------
    layers : Tensor
        A sequential container

    Methods
    -------
    forward(q_values)
        Give the neural network
    """
    def __init__(self, in_channels, num_actions):
        """
        Parameters
        ----------
        in_channels : int
            The number of inputs channels
        num_actions : int 
            The number of actions
        """
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, (5,5)), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, q_values):
        """Choose an action for an epoch
    
        Args:
            q_values (float32 Tensor): the neural network
            
        Returns:
            q_values (float32 Tensor): Tensor list of the q values
        """
        return self.layers(q_values)

class Dueling_DQN(nn.Module):
    """
    A class used to create a Dueling Deep Q Network

    Parameters
    ----------
    nn : Tensor
        The neural network

    Attributes
    ----------
    layers : Tensor
        A sequential container

    Methods
    -------
    forward(q_values)
        Give the neural network
    """
    def __init__(self, in_channels, num_actions):
        """
        Parameters
        ----------
        in_channels : int
            The number of inputs channels
        num_actions : int 
            The number of actions
        """
        super(Dueling_DQN, self).__init__()
        self.num_actions = num_actions
        
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=4)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=2)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1)

        self.advantage_1 = nn.Linear(in_features=64, out_features=512)
        self.value_1 = nn.Linear(in_features=64, out_features=512)

        self.advantage_2 = nn.Linear(in_features=512, out_features=num_actions)
        self.value_2 = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, q_values):
        """Choose an action for an epoch
    
        Args:
            q_values (Tensor): the neural network
            
        Returns:
            q_values (float32 Tensor): Tensor list of the q values
        """
        q_values = self.relu(self.conv_1(q_values))
        q_values = self.relu(self.conv_2(q_values))
        q_values = self.relu(self.conv_3(q_values))
        # New tensor with the same data as the q_values tensor but of a shape 0.
        q_values = q_values.view(q_values.size(0), -1)

        advantages = self.relu(self.advantage_1(q_values))
        values = self.relu(self.value_1(q_values))

        advantages = self.advantage_2(advantages)
        # Returns a new view of the q_values tensor with singleton dimensions expanded to a larger size.
        values = self.value_2(values).expand(q_values.size(0), self.num_actions)
        
        q_values = values + advantages - advantages.mean(1).unsqueeze(1).expand(q_values.size(0), self.num_actions)
        return q_values