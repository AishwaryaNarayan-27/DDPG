import numpy as np
import torch as T
import torch.nn.functional as F
from networks import ActorNetwork, CriticNetwork
from buffer import ReplayBuffer
from noise import OUActionNoise  # Import the noise class

class Agent:
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
                 n_actions=2, max_size=1000000, layer1_size=400,
                 layer2_size=300, batch_size=64, noise=0.1):
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        # Initialize actor and critic networks
        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size,
                                  n_actions=n_actions, name='Actor')
        self.critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size,
                                    n_actions=n_actions, name='Critic')

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size,
                                         n_actions=n_actions, name='TargetActor')
        self.target_critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size,
                                           n_actions=n_actions, name='TargetCritic')

        # Initialize noise for action exploration (Ornstein-Uhlenbeck noise)
        self.noise = OUActionNoise(mu=np.zeros(n_actions), sigma=0.1, theta=0.2)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()

        # Debugging: print the observation to check its shape
        print(f"Original observation: {observation}")

        # Ensure observation is converted to a NumPy array and reshaped properly
        observation = np.array(observation, dtype=np.float32)

        # Debugging: print the shape after conversion to numpy
        print(f"Observation shape after conversion: {observation.shape}")

        # If it's a 1D array, reshape it to (1, -1) to ensure it's in the correct shape
        if observation.ndim == 1:
            observation = observation[np.newaxis, :]
        elif observation.ndim > 2:  # If it's higher-dimensional, flatten it
            observation = observation.reshape(-1)

        # Debugging: print the shape after reshaping
        print(f"Observation shape after reshaping: {observation.shape}")

        # Convert observation to a tensor and move to the actor's device
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)

        # Get the action using the actor network
        mu = self.actor.forward(state).to(self.actor.device)

        # Add noise to the action
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)

        self.actor.train()
        return mu_prime.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, dones = \
            self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        dones = T.tensor(dones).to(self.actor.device)

        target_actions = self.target_actor.forward(states_)
        target_critic_value = self.target_critic.forward(states_, target_actions)

        target_critic_value[dones] = 0.0
        target_critic_value = target_critic_value.view(-1)

        critic_value = self.critic.forward(states, actions).view(-1)
        target = rewards + self.gamma * target_critic_value
        critic_loss = F.mse_loss(target, critic_value)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        actor_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_state_dict = dict(target_critic_params)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_state_dict[name].clone()

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                       (1 - tau) * target_critic_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)
        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
