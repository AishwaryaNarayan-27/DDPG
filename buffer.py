import numpy as np

class ReplayBuffer():
    """
    A class to implement the experience replay buffer for storing agent experiences.
    This buffer allows the agent to randomly sample past experiences, which helps 
    break correlation between consecutive samples and improve the stability of training.
    """
    def __init__(self, max_size, input_shape, n_actions):
        # Initialize the replay buffer with the given size and dimensions
        self.mem_size = max_size  # Maximum size of the buffer
        self.mem_cntr = 0  # Counter to track the current position in the buffer
        self.state_memory = np.zeros((self.mem_size, *input_shape))  # Memory for states
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))  # Memory for next states
        self.action_memory = np.zeros((self.mem_size, n_actions))  # Memory for actions
        self.reward_memory = np.zeros(self.mem_size)  # Memory for rewards
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)  # Memory for done flags

    def store_transition(self, state, action, reward, state_, done):
        """
        Store a new experience in the replay buffer.
        
        This function stores the current state, action, reward, next state, and done flag.
        The memory is circular, so it overwrites older experiences once the buffer is full.
        """
        index = self.mem_cntr % self.mem_size  # Calculate the index to store the experience
        self.state_memory[index] = state  # Store the state
        self.action_memory[index] = action  # Store the action taken
        self.reward_memory[index] = reward  # Store the reward received
        self.new_state_memory[index] = state_  # Store the next state
        self.terminal_memory[index] = done  # Store whether the episode is done

        self.mem_cntr += 1  # Increment the memory counter to track buffer usage

    def sample_buffer(self, batch_size):
        """
        Randomly sample a batch of experiences from the buffer for training.
        
        A batch of experiences is selected from the buffer to break temporal correlations 
        between consecutive states and actions, improving learning stability.
        """
        max_mem = min(self.mem_cntr, self.mem_size)  # Get the current size of the buffer

        # Randomly choose a batch of experiences from the available memory
        batch = np.random.choice(max_mem, batch_size)

        # Retrieve the states, actions, rewards, next states, and done flags for the batch
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones  # Return the sampled batch
