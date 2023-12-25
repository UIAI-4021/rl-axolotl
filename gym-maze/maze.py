import matplotlib.pyplot as plt
import numpy as np
import gym
import matplotlib.pyplot as plot
import gym_maze

# Create an environment
env = gym.make("maze-random-10x10-plus-v0")
observation = env.reset()

# Initialize Q-table with zeros
num_actions = env.action_space.n
Q = np.zeros((100, num_actions))  # Assuming a 2D state space with 10x10 discretized values

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.99  # Epsilon-greedy parameter
min_epsilon = 0.1  # Minimum epsilon value
epsilon_decay = 0.99  # Epsilon decay rate

# Maximum episodes
NUM_EPISODES = 1000
episodes = list()

for episode in range(NUM_EPISODES):

    state = env.reset()
    state = int(state[0] * 10 + state[1])

    total_reward = 0
    steps = 0
    done = False

    while not done and steps <= 100:
        # env.render()

        action = np.argmax(Q[state])  # Greedy action based on Q-values

        # Perform the chosen action
        next_state, reward, done, _ = env.step(action)
        next_state = int(next_state[0] * 10 + next_state[1])

        # Update Q-value for the state-action pair
        Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

        total_reward += reward
        state = next_state

        if done:
            episodes.append(episode)
            print(f"Wins at Episode: {episode + 1}, Total Reward: {total_reward}, epsilon: {epsilon}")
            break

        steps += 1

    # Epsilon decay after each episode
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

y = list()
for i in range(len(episodes) - 1):
    y.append(episodes[i + 1] - episodes[i])
y.append(0)
x = episodes
plot.plot(x, y)
plot.show()
# Close the environment
env.close()
