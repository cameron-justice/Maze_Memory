#!/usr/bin/env python3

from hrr import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

maze_size = 20
goal = 3
r_normal = -1
r_goal = 0

hrr_size = 1024

gamma = 0.9
alpha = 0.001
epsilon = 0.02
p_lambda = 0.1

epochs = 1000
max_moves = maze_size*2

n_strats = 1

bias = 0.0
eligibility = np.zeros([1,1,hrr_size])
targets = np.zeros([1,n_strats,1])

choice_function = np.mean

linear = True
n_outputs = 1
h_size = 32
input_layer = tf.keras.layers.Input(shape=[1,hrr_size])

suboptimal_steps = []

if linear:
    output_layers = [tf.keras.layers.Dense(n_outputs,use_bias=False)(input_layer) for x in range(n_strats)]
else:
    hidden_layers = [tf.keras.layers.Dense(h_size,activation='relu',use_bias=False)(input_layer) for x in range(n_strats)]
    hidden_layers = [tf.keras.layers.Dense(h_size,activation='relu',use_bias=False)(x) for x in hidden_layers]
    output_layers = [tf.keras.layers.Dense(n_outputs,use_bias=False)(x) for x in hidden_layers]

if n_strats > 1:
    c_layer = tf.keras.layers.Concatenate()(output_layers)
else:
    c_layer = output_layers[0]

output_layer = tf.keras.layers.Reshape((n_strats,n_outputs))(c_layer)
model = tf.keras.models.Model([input_layer],[output_layer])

model.compile(loss=tf.keras.losses.mse,
              optimizer=tf.keras.optimizers.SGD(learning_rate=alpha))
model.summary()

ltm = LTM(N=hrr_size, normalized=True)

def selection_function(action_votes):
    (x,y) = np.unique(action_votes, return_counts=True)
    return x[y.argmax()]

def logmod(x):
    return np.sign(x)*np.log(np.abs(x)+1.0)

def encode(state, action):
    comp = "state"+str(state)+"*"+str(action)
    return np.array([ltm.encode(comp)])

def maxq(state):
    hrr = np.vstack([encode(state,"left"), encode(state, "right")])
    values = model.predict(hrr)+bias
    max_actions = values.argmax(0)
    action = max_actions[values.argmax()%n_strats][0]
    return [np.hstack([values[max_actions[x],x:x+1,:] for x in range(n_strats)]),
            hrr[action:action+1,:,:],
            ["left","right"][action]]

def greedyq(state):
    hrr = np.vstack([encode(state,"left"), encode(state, "right")])
    values = model.predict(hrr)+bias
    max_actions = values.argmax(0)
    action = max_actions[values.argmax()%n_strats][0]
    return [values[action:action+1,:,:],
            hrr[action:action+1,:,:],
            ["left","right"][action]]

def randomgreedyq(state):
    hrr = np.vstack([encode(state,"left"), encode(state, "right")])
    values = model.predict(hrr)+bias
    max_actions = values.argmax(0)
    action = max_actions[np.random.choice(n_strats)][0]
    return [values[action:action+1,:,:],
            hrr[action:action+1,:,:],
            ["left","right"][action]]

def randomq(state):
    eligibility[:,:,:] = 0.0
    action = np.random.choice(["left","right"])
    hrr = encode(state, action)
    value = model.predict(hrr)+bias
    return [value,
            hrr,
            action]

for episode in range(epochs):
    eligibility[:,:,:] = 0.0
    current_state = np.random.choice(range(maze_size))
    optimal_steps = min((current_state-goal)%maze_size,(goal-current_state)%maze_size)

    if (episode+1) % 100 == 0:
        print('\rEpisode:', (episode+1),end='')

    for step in range(max_moves):
        if current_state == goal:
            break

        if np.random.random() < epsilon:
            [current_value, current_hrr, current_action] = randomq(current_state)
        else:
            [current_value, current_hrr, current_action] = greedyq(current_state)

        previous_hrr = current_hrr
        previous_value = current_value
        previous_state = current_state
        previous_action = current_action

        if(current_action == "left"):
            current_state = (current_state-1)%maze_size
        else:
            current_state = (current_state+1)%maze_size

        if current_state == goal:
            delta = r_goal - previous_value
            target = logmod(delta) + previous_value
        else:
            [current_value,_,_] = maxq(current_state)
            delta = (r_normal + (gamma*current_value)) - previous_value
            target = logmod(delta) + previous_value

        if n_strats > 1:
            for p in range(n_strats):
                targets[:,p,:] = choice_function(np.delete(target,p,1))
        else:
            targets[:,:,:] = target

        eligibility = (p_lambda * eligibility) + previous_hrr

        model.fit(eligibility, targets-bias,verbose=0)

    suboptimal_steps.append(step - optimal_steps)

plt.plot(range(epochs), suboptimal_steps)
plt.ylabel('Distance from Optimal')
plt.xlabel('Epoch')
plt.savefig('1d_from_optimal.png')
