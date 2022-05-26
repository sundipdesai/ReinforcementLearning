# -*- coding: utf-8 -*-
"""
Created on Wed May 18 19:49:39 2022

@author: sundip r desai

Value iteration
"""

'''
Simple value iteration on grid world

Vk[s] = max(a) SUM(s') [ P(s'|s,a) * (R(s,a,s') + gamma*Vk-1(s')) ]

Assumptions:
    * Any time an agent hits a wall, the agent stays in the same state. 
    * The agent has a (1-noise)% chance of not going where it was supposed to go,
      that is, a random action is taken from the remaining actions 
    * Rewards are terminal states, there is no further action when the agent 
     takes action a from state s that lands it in s' which is a reward state. This
     includes positive or negative rewards. Therefore the value iteration does not
     iterate through reward states (or obstacles)

'''
import numpy as np
#%% Define parameters of GridWorld

#Grid world dims
num_rows = 3
num_cols = 4

# Reward states 
R = np.zeros((num_rows,num_cols))
R[0][3] = 1
R[1][3] = -1

# Obstacles in GridWorld
obstacle = (1,1)

# Gamma (discount rate)
gamma = 0.9

# Noise (probability of taking wrong action)
noise = 0.8

#%% Define the GridWorld (states, actions, values)
# Value grid initialization
V = np.zeros((num_rows,num_cols))
V[0][3] = 1 # Values of reward states pre-canned 
V[1][3] = -1

# States 
states = []
for i in range(num_rows):
    for j in range(num_cols):        
        states.append((i,j))

# Define states that are not evaluated (terminal or obstacles)
remove_states = [(0,3), (1,3)]
remove_states.append(obstacle)

# Remove terminal states and obstacles from iteration
for sr in remove_states:
    states.remove(sr)

# Actions
# left, right, up, down in a grid
actions = [(-1,0), (1,0), (0,-1),(0,1)]

# Convergence metrics
biggest_change = -1 # Biggest change in value for any state 's' between timestep t and t+1
eps = 0.001 # tolerance for convergence (to stop learning)

def getNewState(s,a,nr,nc):
    '''
    Returns indicies of next state, given current state, s, and action, a
    
    If taking action a in state s results in barrier, the agent stays in 
    state s
    
    s: state
    a: action
    return: state that results from taking action a from state s
    '''
    vert = s[0] + a[0]
    horz = s[1] + a[1]
    
    # Agent stays in current state if it hits a wall
    if vert > nr-1 or vert < 0:
        return s
    elif horz > nc-1 or horz < 0:
        return s
    return (vert,horz)

def getRandomAction(a,k):
    '''
    Pick a random action 
    k: index of primary action (highest probability)
    a: list of actions
    returns: a new action that is not k that is probabilistically determined (uniform)
    '''
    random_index = np.random.randint(0,len(a)-1)
    random_action = [i for i in range(len(a)) if i != k]
    return a[random_action[random_index]]
#%% Value iteration
unconverged = False
it = 0
while not unconverged:
    it += 1 
    v_old = V.copy() # copy the value grid from previous time step
    for s in states:        
        v_best = v_old[s[0]][s[1]]
        for k, a in enumerate(actions): 
            # get new state
            s_prime = getNewState(s,a,num_rows,num_cols)
            
            # get random action
            a_rand = getRandomAction(actions, k)
            s_rand = getNewState(s,a_rand,num_rows,num_cols)        
                        
            # Update the value for state 's'        
            v_new = noise*(R[s[0]][s[1]] + gamma*(v_old[s_prime[0]][s_prime[1]])) + (1-noise)*(R[s[0]][s[1]] + gamma*v_old[s_rand[0]][s_rand[1]])
            
            if v_new > v_best: # max value over all actions in state s
                v_best = v_new                
        V[s[0]][s[1]] = v_best #update value with the best so far             
    
        # check if the biggest deviation in value is < eps                 
        delta = abs(v_best - v_old[s[0]][s[1]])  
        biggest_change = max(delta, biggest_change)
                   
    if biggest_change < eps: 
        unconverged = True
    else: 
        biggest_change = -1     # reset the biggest_change for next sweep       

print("Number of iterations: ", str(it))
print("Final value function:")
print(V)
#%% END






