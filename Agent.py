import numpy
import numpy as np
import shelve
import os
import glob
import gc
import itertools
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from hrr import *

class Agent:
    def __init__(self, action_set, hidden_size, ltm, 
                 hrr_size=1024, policy_epsilon=0.03, policy_gamma=0.9, 
                 policy_lambda=0.1, policy_alpha=0.001):
        self.hrr_size = hrr_size
        self.policy_epsilon = policy_epsilon
        self.policy_gamma = policy_gamma
        self.policy_lambda = policy_lambda
        self.policy_alpha = policy_alpha
        self.ltm = ltm
        
        self.action_set = action_set
        
        self.bias = 0
        
        self.reward_illegal = -2
        self.reward_bad = -1
        self.reward_goal = 0
        
        self.targets = np.zeros([1,1,1]) # Q-Factored Qt values
        
        self.eligibility = np.zeros([1,1,hrr_size])
        
        self.current_value = 0.0
        self.current_hrr = np.zeros([hrr_size])
        self.current_action = action_set[0]
        
        self.previous_value = self.current_value
        self.previous_hrr = self.current_hrr
        
        self.previous_wm = "I"
        
        # Make model
        
        input_layer = tf.keras.layers.Input(shape=[1,hrr_size]) # Defined input layer for HRR

        # First hidden layer - nonlinear
        hidden_layer = tf.keras.layers.Dense(hidden_size, activation='relu', use_bias=False)(input_layer)

        output_layer = tf.keras.layers.Dense(1, use_bias=False)(hidden_layer)

        net = tf.keras.models.Model([input_layer],[output_layer])

        net.compile(loss=tf.keras.losses.mse,
                     optimizer=tf.keras.optimizers.SGD(learning_rate=policy_alpha))
        
        self.model = net
        
    def selection_function(self, action_votes):
        (x,y) = np.unique(action_votes, return_counts=True)
        return x[y.argmax()]

    def logmod(self, x):
        return np.sign(x)*np.log(np.abs(x)+1.0)
        
    def encode(self, state, action, signal, wm="I"):
        comp = "state"+str(state)+"*"+str(action)+"*"+str(signal) + "*"+"wm_"+str(wm)
        return np.array([self.ltm.encode(comp)])
        
    def randomq(self, state, signal, wm):
        self.eligibility[:,:,:] = 0.0
        action = np.random.choice(self.action_set)
        hrr = self.encode(state, action, 'sig_' + signal, 'wm_' + wm)
        value = self.model.predict(hrr) + self.bias
        return [value, hrr, action]
        
    def greedyq(self, state, signal, wm):
        hrrs = np.vstack([self.encode(state, action, 'sig_' + signal, 'wm_' + wm) for action in self.action_set])
        values = self.model.predict(hrrs) + self.bias
        max_actions = values.argmax(0)
        
        action = max_actions[0][0]
        return [values[action:action+1,:,:],
               hrrs[action:action+1,:,:],
               self.action_set[action]]
        
    def maxq(self, state, signal, wm):
        hrrs = np.vstack([self.encode(state, action, 'sig_' + signal, 'wm_' + wm) for action in self.action_set])
        values = self.model.predict(hrrs)+self.bias
        max_actions = values.argmax(0)
        action = max_actions[0][0]
        
        return [values[max_actions[0],0:1,:],
                hrrs[action:action+1,:,:],
                self.action_set[action]]
    
    def update(self, state, signal, wm, is_goal, illegal_move=False):
        if is_goal:
            delta = self.reward_goal - self.previous_value
            target = self.logmod(delta) + self.previous_value
        else:
            reward = self.reward_illegal if illegal_move else self.reward_bad
            
            [self.current_value,_,_] = self.maxq(state, signal, wm)
            delta = (reward + (self.policy_gamma * self.current_value)) - self.previous_value
            target = self.logmod(delta) + self.previous_value
    
        self.targets[:,:,:] = target
        self.eligibility = (self.policy_lambda * self.eligibility) + self.previous_hrr
        
        self.model.fit(self.eligibility, self.targets-self.bias, verbose=0)
    
    def policy(self, state, signal, wm):
        if np.random.random() < self.policy_epsilon:
            [self.current_value, self.current_hrr, self.current_action] = self.randomq(state, signal, wm)
        else:
            [self.current_value, self.current_hrr, self.current_action] = self.greedyq(state, signal, wm)       
            
            self.previous_value = self.current_value
            self.previous_hrr = self.current_hrr
            self.previous_wm = wm