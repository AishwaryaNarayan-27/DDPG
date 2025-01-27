import numpy as np

class OUActionNoise():
    """
    A class that implements the Ornstein-Uhlenbeck (OU) process for adding noise to actions.
    This noise helps in exploration by adding a temporal correlation to the noise, which 
    is commonly used in reinforcement learning algorithms like DDPG.
    """
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        # Initialize parameters for the Ornstein-Uhlenbeck noise process
        self.theta = theta  # Rate of mean reversion
        self.mu = mu  # The long-run mean of the process (target)
        self.sigma = sigma  # Standard deviation (volatility) of the noise
        self.dt = dt  # Time step for the process
        self.x0 = x0  # Initial state of the noise process
        self.reset()  # Call reset to initialize the noise state

    def __call__(self):
        """
        Generate the next noise value using the Ornstein-Uhlenbeck process.
        
        The process adds noise with temporal correlation, ensuring smoother exploration.
        """
        # Calculate the next value in the noise process using the OU equation
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        
        self.x_prev = x  # Update the previous state for the next call

        return x  # Return the generated noise value

    def reset(self):
        """
        Reset the noise process to its initial state.
        
        This is useful for reinitializing the process at the start of an episode or after training.
        """
        # If an initial state (x0) is provided, use it; otherwise, start from a zero state
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


