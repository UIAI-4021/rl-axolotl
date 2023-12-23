import numpy as np
import random
import gym
import gym_maze

# Create the maze environment
env = gym.make("maze-random-10x10-plus-v0")

# Discretize the continuous state space
def discretize_state(state):
    # Define the state boundaries for discretization
    state_bounds = [(-1, 1), (-1, 1)]  # Example state boundaries for a 2D state space

    # Discretize the state values
    discretized_state = []
    for i in range(len(state)):
        lower_bound, upper_bound = state_bounds[i]
        discrete_val = int((state[i] - lower_bound) / (upper_bound - lower_bound) * 10)  # 10 buckets
        discretized_state.append(max(0, min(discrete_val, 9)))  # Clip values to [0, 9]

    return tuple(discretized_state)

# Initialize Q-table with zeros
num_actions = env.action_space.n
Q = np.zeros((10, 10, num_actions))  # Assuming a 2D state space with 10x10 discretized values

# Q-learning parameters
alpha = 0.5  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.99  # Epsilon-greedy parameter
min_epsilon = 0.1  # Minimum epsilon value
epsilon_decay = 0.99  # Epsilon decay rate

# Maximum episodes
NUM_EPISODES = 1000

for episode in range(NUM_EPISODES):
    state = discretize_state(env.reset())
    total_reward = 0

    while True:
        env.render()

        # Epsilon-greedy policy for action selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Random action
        else:
            action = np.argmax(Q[state])  # Greedy action based on Q-values

        # Perform the chosen action
        next_state, reward, done, _ = env.step(action)
        next_state = discretize_state(next_state)

        # Update Q-value for the state-action pair
        Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

        total_reward += reward
        state = next_state

        if done:
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
            break

    # Epsilon decay after each episode
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Close the environment
env.close()
