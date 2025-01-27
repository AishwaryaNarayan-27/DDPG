# DDPG

# Deep Deterministic Policy Gradient (DDPG) for LunarLanderContinuous-v2

## Overview
This project demonstrates the implementation of the **Deep Deterministic Policy Gradient (DDPG)** algorithm to train an agent to land a spaceship smoothly on a lunar surface in the **LunarLanderContinuous-v2** environment from OpenAI Gym.

DDPG is an actor-critic algorithm that works in continuous action spaces. It combines the strengths of deterministic policy gradients and deep learning, using two neural networks (actor and critic) to model the policy and value functions.

---

## Features
- **Reinforcement Learning with DDPG**: Implements the DDPG algorithm to solve continuous control problems.
- **Target Networks**: Stabilizes training with target actor and critic networks.
- **Experience Replay**: Improves sample efficiency by reusing past experiences.
- **Training and Evaluation Modes**: Supports both exploration (training) and deterministic evaluation modes.
- **Noise for Exploration**: Uses Ornstein-Uhlenbeck noise to facilitate exploration in continuous action spaces.
- **Save and Load Models**: Saves actor and critic models periodically during training for future evaluation.

---

## Environment: LunarLanderContinuous-v2
The **LunarLanderContinuous-v2** environment requires the agent to control the spaceship's two thrusters to safely land it on a target. The action space is continuous, with:
- **Actions**: Two continuous values between -1 and 1, representing the power of the main engine and the lateral thrusters.
- **State Space**: An 8-dimensional vector representing the lander's position, velocity, orientation, and leg contact status.
- **Rewards**: Positive rewards for reaching the target and penalties for crashing or using excessive fuel.

---

## Requirements

### Dependencies
Install the required libraries using the following:

```bash
pip install numpy gym torch matplotlib
```

### File Structure
```
project/
|-- main_ddpg.py               # Main script to train and evaluate the agent
|-- ddpg_torch.py              # DDPG agent implementation
|-- utils.py                   # Utility functions for plotting and saving models
|-- models/                    # Directory to save trained models
|-- results/                   # Directory to save training results (plots, metrics)
```

---

## How to Run

### 1. Training the Agent
To train the agent using DDPG on the LunarLanderContinuous-v2 environment:
```bash
python main_ddpg.py
```
During training:
- The agent learns by interacting with the environment.
- Models are saved every 10 episodes in the `models/` directory.
- A learning curve is generated and saved in the `results/` directory.



### 2. Visualize Training Results
After training, a learning curve is saved as a plot showing the episode scores and the running average score. Open the file in `results/` to analyze the agent’s performance.

---

## Key Components

### 1. **Deep Deterministic Policy Gradient (DDPG)**
- **Actor Network**: Maps states to deterministic actions.
- **Critic Network**: Estimates the Q-value for state-action pairs.
- **Target Networks**: Stabilize training by slowly updating copies of the actor and critic networks.

### 2. **Experience Replay Buffer**
Stores transitions (state, action, reward, next state, done) to sample and reuse during training. This helps decorrelate experiences and improves learning stability.

### 3. **Noise for Exploration**
Adds Ornstein-Uhlenbeck noise to the actions to encourage exploration in continuous action spaces.

---

## Results
The trained agent learns to land the spaceship effectively by maximizing the reward over time. The average score improves as the agent trains over 500 episodes.

### Example Output
```
Episode 500/500 completed, Score: 11.37, Avg. Score: -0.58
... saving models ...
Actor model saved to ./models/actor_model.pth
Critic model saved to ./models/critic_model.pth
```

---

## Improvements
- Implement **Prioritized Experience Replay** to sample important transitions more frequently.
- Experiment with hyperparameters like learning rates, replay buffer size, and discount factor.
- Try different exploration strategies for better policy discovery.
- Extend the project to other continuous control environments (e.g., Pendulum-v0, BipedalWalker-v3).

---

## References
1. [DDPG Paper](https://arxiv.org/abs/1509.02971)
2. [OpenAI Gym Documentation](https://www.gymlibrary.dev/)
3. [PyTorch Documentation](https://pytorch.org/docs/)

---

## Author
**Aishwarya Narayan**

