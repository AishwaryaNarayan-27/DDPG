import numpy as np
import gym
from ddpg_torch import Agent
from utils import plot_learning_curve

# Create the environment
env = gym.make('LunarLanderContinuous-v2')

# Set up the agent with relevant parameters
agent = Agent(alpha=0.0001, beta=0.001, input_dims=[8], tau=0.001,
              env=env, gamma=0.99, n_actions=env.action_space.shape[0])

# Track the learning curve
n_games = 500
filename = 'LunarLander_ddpg_learning_curve.png'  # Output file for the plot
figure_file = 'LunarLander_ddpg_learning_curve_plot.png'  # Save the learning curve plot separately

best_score = env.reward_range[0]
score_history = []

# Train the agent
print("Starting training...")  # Debug print to confirm the script is running

for i in range(n_games):
    observation, info = env.reset()  # Get initial state (LunarLanderContinuous-v2 returns two variables)
    done = False
    score = 0
    timestep = 0  # Initialize timestep counter

    print(f"Starting Episode {i+1}/{n_games}...")  # Log the start of an episode

    while not done:
        # Choose an action from the agent based on the current observation
        action = agent.choose_action(observation)
        
        # Take a step in the environment
        observation_, reward, terminated, truncated, info = env.step(action)
        
        done = terminated or truncated  # Handle termination from both flags
        agent.remember(observation, action, reward, observation_, done)
        agent.learn()
        score += reward
        observation = observation_

        # Log observation, timestep, and episode details
        print(f"Episode: {i+1}, Timestep: {timestep}, Observation: {observation}, Reward: {reward}")
        timestep += 1  # Increment timestep counter

    # Append the score for the episode to the score history
    score_history.append(score)
    
    # Calculate the average score for the last 100 episodes
    avg_score = np.mean(score_history[-100:])
    
    # Log episode summary details
    print(f"Episode {i+1}/{n_games} completed, Score: {score}, Avg. Score: {avg_score}")

    # Save models every 10 episodes
    if (i + 1) % 10 == 0:
        agent.save_models()

# Plot learning curve
plot_learning_curve(score_history, figure_file, filename)
