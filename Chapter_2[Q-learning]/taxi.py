import numpy as np 
import gym
import random
from matplotlib import pyplot as plt

# Hyperparameter
total_episodes = 50000
max_steps = 99

learning_rate = 0.7
gamma = 0.618

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

test_episodes = 100
train = False
test = True


# Make Env
env = gym.make("Taxi-v3")

# Initialize Q-Table
action_size = env.action_space.n
state_size = env.observation_space.n
qtable = np.zeros((state_size, action_size))

# Train
if train:
    rewards = []

    for episode in range(total_episodes):
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0

        for step in range(max_steps):
            tradeoff = random.uniform(0, 1)
            if tradeoff > epsilon:
                # Exploitation
                action = np.argmax(qtable[state, :])
            else:
                # Exploration
                action = env.action_space.sample()
        
            new_state, reward, done, info = env.step(action)

            # Update Q Table
            dq = reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action]
            qtable[state, action] = qtable[state, action] + learning_rate * dq
            
            # Move to next state
            total_rewards += reward
            state = new_state

            if done:
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        rewards.append(total_rewards)
    np.save('Taxi-Qtable.npy', qtable)
    print('Score over time: ' + str(sum(rewards)/total_episodes))
    print('Q-Table'.center(40,'*'))
    print(qtable)
    plt.plot(range(1,total_episodes+1), np.cumsum(rewards)/range(1,total_episodes+1))
    plt.title("Training Process")
    plt.xlabel('Train Episode')
    plt.ylabel('Average Total Rewards')
    plt.xlim(0, total_episodes)
    plt.ylim(0, 15)
    plt.show()

# Test
if test:
    rewards = []
    qtable = np.load('Taxi-Qtable.npy')
    for episode in range(test_episodes):
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0
        print(('EPSIODE = %d'%(episode+1)).center(40,'*'))

        for step in range(max_steps):
            action = np.argmax(qtable[state, :])
            new_state, reward, done, info = env.step(action)

            total_rewards += reward
            state = new_state

            if done:
                env.render()
                print('Number of steps %d'%step)
                break

        rewards.append(total_rewards)
    plt.bar(range(1, test_episodes+1), rewards)
    plt.title("Test Process[Average Score:%.2f]"%(sum(rewards)/len(rewards)))
    plt.xlabel('Test Episode')
    plt.ylabel('Total Rewards')
    plt.xlim(0, test_episodes+1)
    plt.ylim(bottom=0)
    plt.show()        

env.close()