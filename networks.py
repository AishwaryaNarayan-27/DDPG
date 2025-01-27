import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CriticNetwork(nn.Module):
    """
    A neural network class representing the Critic part of the DDPG algorithm.
    It estimates the Q-value for a given state-action pair, providing the feedback
    for the agent's actions in reinforcement learning.
    """
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        # Initialize the network layers and parameters
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')

        # Fully connected layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # Layer Normalization
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # Action value layer to add the contribution of the action in Q-value estimation
        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        
        # Output layer for the Q-value estimation
        self.q = nn.Linear(self.fc2_dims, 1)

        # Initialize weights using Xavier initialization
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        # Small initialization for Q-value and action value layers
        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)

        f4 = 1./np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)

        # Adam optimizer to optimize the network parameters
        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=0.01)
        
        # Device configuration: Use MPS (Metal Performance Shaders) if available, else use CPU
        self.device = T.device('mps' if T.backends.mps.is_available() else 'cpu')

        self.to(self.device)
        print(f'Running on device: {self.device}')

    def forward(self, state, action):
        """
        Forward pass of the Critic Network, which takes a state and an action 
        to estimate the Q-value (state-action value).
        """
        state_value = self.fc1(state)  # Pass state through the first layer
        state_value = self.bn1(state_value)  # Apply batch normalization
        state_value = F.relu(state_value)  # Apply ReLU activation
        state_value = self.fc2(state_value)  # Pass state through the second layer
        state_value = self.bn2(state_value)  # Apply batch normalization
        action_value = self.action_value(action)  # Process action through action_value layer
        state_action_value = F.relu(T.add(state_value, action_value))  # Combine state and action values
        state_action_value = self.q(state_action_value)  # Output the Q-value

        return state_action_value

    def save_checkpoint(self):
        """
        Save the current model parameters to a checkpoint file.
        This allows restoring the model later during training or evaluation.
        """
        # Ensure the checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Save the model's state dictionary to the checkpoint file
        T.save(self.state_dict(), os.path.join(self.checkpoint_dir, 'critic_model.pth'))
        print(f"Critic model saved to {self.checkpoint_dir}/critic_model.pth")

    def load_checkpoint(self):
        """
        Load the model parameters from a checkpoint file.
        """
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        """
        Save the best performing model's state dictionary.
        This is useful to keep track of the highest performing model during training.
        """
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        T.save(self.state_dict(), checkpoint_file)

class ActorNetwork(nn.Module):
    """
    A neural network class representing the Actor part of the DDPG algorithm.
    It outputs the action to be taken based on the current state, learning the optimal policy.
    """
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        # Initialize the network layers and parameters
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')

        # Fully connected layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # Layer Normalization
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # Output layer to predict the action (in the continuous action space)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        # Initialize weights using Xavier initialization
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        # Small initialization for the output layer (mu)
        f3 = 0.003
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)

        # Adam optimizer to optimize the network parameters
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
       
        # Device configuration: Use MPS (Metal Performance Shaders) if available, else use CPU
        self.device = T.device('mps' if T.backends.mps.is_available() else 'cpu')

        self.to(self.device)
        print(f'Running on device: {self.device}')

    def forward(self, state):
        """
        Forward pass of the Actor Network, which takes the state as input 
        and outputs the action to be taken (policy output).
        """
        x = self.fc1(state)  # Pass state through the first layer
        x = self.bn1(x)  # Apply batch normalization
        x = F.relu(x)  # Apply ReLU activation
        x = self.fc2(x)  # Pass state through the second layer
        x = self.bn2(x)  # Apply batch normalization
        x = F.relu(x)  # Apply ReLU activation
        x = T.tanh(self.mu(x))  # Output the action (in the range [-1, 1])

        return x

    def save_checkpoint(self):
        """
        Save the current model parameters to a checkpoint file.
        """
        # Ensure the checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Save the model's state dictionary to the checkpoint file
        T.save(self.state_dict(), os.path.join(self.checkpoint_dir, 'actor_model.pth'))
        print(f"Actor model saved to {self.checkpoint_dir}/actor_model.pth")

    def load_checkpoint(self):
        """
        Load the model parameters from a checkpoint file.
        """
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        """
        Save the best performing model's state dictionary.
        """
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        T.save(self.state_dict(), checkpoint_file)
