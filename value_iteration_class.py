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
#%% Gridworld class 
class gridWorld: 
    def __init__(self, R, V, obs, num_rows=3, num_cols=4, gamma=0.9, noise=0.8, epsilon=0.001):        
        
        #Grid world dims
        self.num_rows = 3
        self.num_cols = 4
        
        #Reward function 
        self.R = R

        #Initial Value function
        self.V = V
        
        #Obstacles
        self.obstacle = obs
        
        #Discount rate
        self.gamma = gamma
        
        #Noise (probability of taking wrong action)
        self.noise = noise

        #Epsilon (convergence tolerance)
        self.epsilon = epsilon
        
        #Construct the grid
        self.initGrid()
            
    def initGrid(self):
        # States 
        self.states = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):        
                self.states.append((i,j))
        
        # Remove states that are not evaluated (terminal or obstacles)
        for ts in np.transpose(np.nonzero(R)):       
            print(tuple(ts))
            print(self.states)
            self.states.remove(tuple(ts))
        for obs in self.obstacle:
            self.states.remove(obs)
                
        
        # Actions
        # left, right, up, down in a grid
        self.actions = [(-1,0), (1,0), (0,-1),(0,1)]        

    def getNewState(self,s,a):
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
        if vert > self.num_rows-1 or vert < 0:
            return s
        elif horz > self.num_cols-1 or horz < 0:
            return s
        return (vert,horz)

    def getRandomAction(self,a,k):
        '''
        Pick a random action 
        k: index of primary action (highest probability)
        a: list of actions
        returns: a new action that is not k that is probabilistically determined (uniform)
        '''
        random_index = np.random.randint(0,len(a)-1)
        random_action = [i for i in range(len(a)) if i != k]
        return a[random_action[random_index]]

    def valueIteration(self):
        unconverged = False
        it = 0
        biggest_change = -100 # Biggest change in value for any state 's' between timestep t and t+1
        while not unconverged:
            it += 1 
            v_old = self.V.copy() # copy the value grid from previous time step
            for s in self.states:        
                v_best = v_old[s[0]][s[1]]
                for k, a in enumerate(self.actions): 
                    # get new state
                    s_prime = self.getNewState(s,a)
                    
                    # get random action
                    a_rand = self.getRandomAction(self.actions, k)
                    s_rand = self.getNewState(s,a_rand)        
                                
                    # Update the value for state 's'        
                    v_new = self.noise*(self.R[s[0]][s[1]] + self.gamma*(v_old[s_prime[0]][s_prime[1]])) + (1-self.noise)*(self.R[s[0]][s[1]] + self.gamma*v_old[s_rand[0]][s_rand[1]])
                    
                    # max value over all actions in state s
                    v_best = max(v_new, v_best)                
                self.V[s[0]][s[1]] = v_best #update value with the best so far             
            
                # check if the biggest deviation in value is < eps                 
                delta = abs(v_best - v_old[s[0]][s[1]])  
                biggest_change = max(delta, biggest_change)
                           
            if biggest_change < self.epsilon: 
                unconverged = True
            else: 
                biggest_change = -1     # reset the biggest_change for next sweep       
        
        print("Number of iterations: ", str(it))
        print("Final value function:")
        print(self.V)
#%% Simulation setup
# Reward states 
rows=3
cols=4
R = np.zeros((rows,cols))
R[0][3] = 1
R[1][3] = -1

# Obstacles in GridWorld
obstacle = [(1,1)]

# Value grid initialization
V = np.zeros((rows,cols))
V[0][3] = 1 # Values of reward states pre-canned 
V[1][3] = -1
#%% Simulation run
gw=gridWorld(R, V, obstacle)
gw.valueIteration()




















