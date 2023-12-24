import matplotlib.pyplot as plt
import numpy as np
import gym
import matplotlib.pyplot as plot
import gym_maze

# Create an environment
env = gym.make("maze-random-10x10-plus-v0")
observation = env.reset()


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
gamma = 0.99  # Discount factor
epsilon = 0.99  # Epsilon-greedy parameter
min_epsilon = 0.1  # Minimum epsilon value
epsilon_decay = 0.99  # Epsilon decay rate

# Maximum episodes
NUM_EPISODES = 2000
episodes = list()

for episode in range(NUM_EPISODES):
    State = discretize_state(env.reset())
    total_reward = 0
    steps = 0
    done = False

    while not done and steps <= 100:
        # env.render()
        steps += 1
        action = np.argmax(Q[State])  # Greedy action based on Q-values

        # Perform the chosen action
        next_state, reward, done, _ = env.step(action)
        next_state = discretize_state(next_state)

        # Update Q-value for the state-action pair
        Q[State][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[State][action])

        total_reward += reward
        State = next_state

        if done:
            episodes.append(episode)
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}, epsilon: {epsilon}")
            break

    # Epsilon decay after each episode
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

y = list()
for i in range(len(episodes) - 1):
    y.append(episodes[i+1] - episodes[i])
y.append(0)
x = episodes
plot.plot(x,y)
plot.show()
# Close the environment
env.close()
