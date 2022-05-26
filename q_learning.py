# -*- coding: utf-8 -*-
"""
Created on Sun May 22 10:42:20 2022

@author: sundip r desai

Q-learning 

Update rule:
    Qnew(s,a) = Qcurrent(s,a) + alpha*(R(s,a) + gamma*maxQ(s',a') - Qcurrent(s,a))

Assumptions/Rules:    
    * Simple example taken from https://valohai.com/blog/reinforcement-learning-tutorial-part-1-q-learning/
    * From any state the agent takes the action left, it will be brought to left most state with an immediate reward
    * Any time the agent takes the action 'right' it will go one state over to the right
    * If an action 'a' is taken in state 's' that results in hitting a boundary, the agent remains in state 's'
        
"""
import numpy as np

# Generate the state-space
num_states = 5

# Generate action space
actions = ['right', 'left']

# Reward states 
R = np.zeros(num_states)
R[0] = 2
R[4] = 10

# Q-table
# rows are actions, states are columns
# row = 0, right turn
# row = 1, left turn
Q = np.zeros((len(actions),num_states))

# Gamma (discount rate)
gamma = 0.95

# Learning rate
alpha = 0.1

# Exploration rate (large initially and decays by 'decay' thereafter)
epsilon = 1.0
decay = 0.9

# Number of iterations
num_iter = 2000
eps_iter = 1000 # number of iterations before exploration rates decays
#%% Q-learning
# start state
s = 0

def stateTransition(s,a,ns):
    '''
    stateTransition
    Desc: Returns the state, s' from s by taking action a
    '''    
    # if action is left, the agent immediately goes to the left most state 
    if a == 'left':
        return 0
    elif a == 'right':
        if s+1 > ns-1: # reached the end, agent stays put
            return s
        return s+1 

# Q-learning
for i in range(num_iter):
    # Explore 
    if np.random.uniform(0,1) < epsilon:
        a = round(np.random.uniform(0,1)) # pick random state  
    else:
        #exploit
        if Q[0][s] == Q[1][s]: #if there is a tie, then pick an action randomly
            a = round(np.random.uniform(0,1))
        else: # pick the action that has a better Q-value
            a = np.argmax([Q[0][s], Q[1][s]])    
    sp = stateTransition(s, actions[a], num_states) #get transition state, s-prime

    # Q-learning with temporal differencing
    Q[a][s] +=  alpha*(R[sp] + gamma*np.max([Q[0][sp], Q[1][sp]]) - Q[a][s])
    s = sp # update the state
    
    if i > eps_iter: #after a number of iterations, start decaying the exploration
        epsilon *= decay
    
print("Q-table: \n", Q)
#%%  END

'''
Notes:
    
    
'''



