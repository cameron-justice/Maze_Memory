#!/usr/bin/env python3

from hrr import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Initial Setup

maze_size = 20
goal_state_red = 7
goal_state_green = 191

rewards = [-1] * maze_size

hrr_size = 64
gamma = 0.9
alpha = 0.001
epsilon = 0.99
epsilon_gamma = 0.99

epochs = 1000
maxMoves = maze_size

testepochs = 50

# Policy Network

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64,input_dim=hrr_size, use_bias=False))
model.add(tf.keras.layers.Dense(32,use_bias=False))
model.add(tf.keras.layers.Dense(2, activation='softmax',use_bias=False))
model.compile(loss=tf.keras.losses.mse,
                optimizer=tf.keras.optimizers.SGD(lr=alpha))

# Allows an hrr to be passed directly for prediction
def predict(rep, model):
    rep = [rep]
    rep = np.array(rep)
    return model.predict(rep)[0]


# Functions relating to rewards

def reset_rewards_state(r_set):
    r_set[goal_state_red] = 0
    r_set[goal_state_green] = 0
    return r_set

def set_rewards_state(r_set, goalIsRed):
    if goalIsRed:
        r_set[goal_state_red] = 0
    else:
        r_set[goal_state_green] = 0

    return r_set

# Generating HRRs for all possible states

ltm = LTM(N=hrr_size, normalized=True)

for state in range(maze_size):
    ltm.encode('s' + str(state))

# Functions for managing the agent

def get_action(state_hrr, model):
    V = predict(state_hrr, model)
    return np.argmax(V) # Highest value action

def move(moveLeft, index):
    index = index - 1 if moveLeft else index + 1
    if index < 0:
        index = 0
    elif index >= maze_size:
        index = 0
    elif index < 0:
        index = maze_size - 1
    return index

def check_goal(index, rewards):
    return index == goal_state_red

# Start training

rat_index = np.random.randint(maze_size)
set_rewards_state(rewards, goalIsRed=True)

print("Goal State: %d" % (goal_state_red))

for epoch in range(epochs):
    rat_index = np.random.randint(maze_size)
    while rat_index == goal_state_red:
        rat_index = np.random.randint(maze_size)
    move_count = 0
    reward = 0
    done = False
    start_state = rat_index
    while not done:
        state = ltm.lookup('s' + str(rat_index))
        action = get_action(state, model) if np.random.random() > epsilon else np.random.choice([0,1])
        rat_index = move(action, rat_index)
        next_state = ltm.lookup('s' + str(rat_index))
        reward = rewards[rat_index]
        done = check_goal(rat_index, rewards)
        move_count += 1

        target = reward + gamma * np.max(predict(next_state, model))
        target_vec = predict(state, model)
        target_vec[action] = target
        model.fit(state.reshape(-1,hrr_size), target_vec.reshape(-1,2), epochs=1, verbose=0)

        if move_count >= maxMoves:
            done = True

    epsilon *= epsilon_gamma
    print("Epoch: %d, Reward: %d, Start: %d, Optimal: %d, Moves: %d, Epsilon: %f" % (epoch+1, reward, start_state, abs(goal_state_red - start_state), move_count, epsilon))

tracker = deque(maxlen=epochs)
optimal = 0
    
for epoch in range(testepochs):
    rat_index = np.random.randint(maze_size)
    while rat_index == goal_state_red:
        rat_index = np.random.randint(maze_size)
    move_count = 0
    done = False
    start_state = rat_index
    while not done:
        state = ltm.lookup('s' + str(rat_index))
        action = get_action(state, model)
        rat_index = move(action, rat_index)
        move_count += 1
        done = check_goal(rat_index, rewards)
        if move_count >= maxMoves:
            done = True
    tracker.append(move_count - abs(goal_state_red - start_state))
    if (move_count - abs(goal_state_red - start_state)) == 0:
        optimal += 1
    
print("Agent achieved the optimal path on %d/%d tests" % (optimal, testepochs))
    
plt.plot(range(testepochs), tracker)
plt.ylabel('Distance from Optimal')
plt.xlabel('Epoch')
plt.savefig('1d_red_from_optimal.png')
