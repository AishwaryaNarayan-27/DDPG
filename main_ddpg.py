import numpy as np
import gym
from ddpg_torch import Agent
from utils import plot_learning_curve

# Create the LunarLander environment for continuous control tasks
env = gym.make('LunarLanderContinuous-v2')

# Initialize the agent with necessary parameters (Learning rate, gamma, input dimensions, etc.)
agent = Agent(alpha=0.0001, beta=0.001, input_dims=[8], tau=0.001,
              env=env, gamma=0.99, n_actions=env.action_space.shape[0])

# Variables to track the learning progress
n_games = 500  # Total number of training episodes
filename = 'LunarLander_ddpg_learning_curve.png'  # Output file name for the learning curve plot
figure_file = 'LunarLander_ddpg_learning_curve_plot.png'  # Save the plot as a separate image file

# Initialize variables for tracking the best score and the history of scores
best_score = env.reward_range[0]  # Initialize best score
score_history = []  # List to store the scores for each episode

# Start the training loop for the agent
print("Training has begun...")  # Print confirmation that training has started

for i in range(n_games):
    observation, info = env.reset()  # Reset the environment to get the initial state
    done = False  # Flag to track the end of the episode
    score = 0  # Initialize the score for the current episode
    timestep = 0  # Track the number of timesteps in the episode

    print(f"Training Episode {i + 1}/{n_games} begins...")  # Print the start of each episode

    # Run the episode until the environment is done
    while not done:
        # Select an action using the agent's policy based on the current observation
        action = agent.choose_action(observation)

        # Apply the action and get the next state, reward, and other info
        observation_, reward, terminated, truncated, info = env.step(action)
        
        # Check if the episode is finished due to termination or truncation
        done = terminated or truncated  
        
        # Store the experience in the agent's memory
        agent.remember(observation, action, reward, observation_, done)
        
        # Learn from the stored experiences
        agent.learn()
        
        # Update the episode score with the reward
        score += reward
        observation = observation_  # Update the observation for the next timestep

        # Print useful details during the episode for debugging and analysis
        print(f"Episode: {i + 1}, Timestep: {timestep}, Observation: {observation}, Reward: {reward}")
        timestep += 1  # Increment timestep counter

    # Append the total score for this episode to the score history
    score_history.append(score)

    # Calculate the moving average of the last 100 scores to track progress
    avg_score = np.mean(score_history[-100:])
    
    # Print summary information at the end of each episode
    print(f"Episode {i + 1}/{n_games} finished, Score: {score}, Avg. Score: {avg_score}")

    # Save the agent's models every 10 episodes to avoid losing progress
    if (i + 1) % 10 == 0:
        agent.save_models()

# After training, plot the learning curve to visualize agent performance
plot_learning_curve(score_history, figure_file, filename)
