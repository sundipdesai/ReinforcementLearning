'''
Multi-Armed Bandit Simulation

S. Desai 2020

Summary:
--------
Class 'multi_arm_bandit_testbed' creates a test bed for evaluating
the multi-arm (k-arm) bandit problem, especially the tradeoff
between exploration and exploitation of choices.

This test bed allows experimentation of:
    - Incremental Updating
    - Epsilon Greedy 
    - Optimistic Initial Values
    - Upper Confidence Bound (UCB) Action Selection 

References:
-----------
Reinforcement Learning (Sutton, Barton), 2nd Edition

'''
import numpy as np
import matplotlib.pyplot as plt

class multi_arm_bandit_testbed():
    def __init__(self,num_arms=10, num_steps=1000, num_runs=2000, epsilon=-1.0, opt_init=0, ucb=False, c=2):
        self.num_arms = num_arms
        self.steps = num_steps
        self.runs = num_runs
        self.reward = np.zeros((self.runs,self.steps))
        self.epsilon = epsilon
        self.opt_init = opt_init
        self.ucb = ucb
        self.c = c
    
    def breakTies(self,Q):
        #break ties randomly
        ties=[x for x,q in enumerate(Q) if q == max(Q)]
        return np.random.choice(ties)
        
    def multi_arm_bandit_run(self):  
        for run in range(self.runs):
            print("Simulating Run #",run)
            # initialize after every run      
            self.Q = self.opt_init*np.ones(self.num_arms)
            self.N = np.zeros(self.num_arms)  
            self.qstar = np.random.randn(self.num_arms)
            for step in range(self.steps):            
                if self.epsilon > np.random.rand(): # explore randomly with a probability of epsilon 
                    self.action = np.random.randint(0,self.num_arms)
                elif self.ucb: # apply upper bound uncertainty to estimate and take best estimate (UCB)  
                    self.action = self.breakTies(np.add(self.Q,self.c*np.sqrt(np.log(step+1)/self.N)))          
                else:  # greedy approach
                    self.action = self.breakTies(self.Q)
                tmp_action = self.action
                self.N[tmp_action] += 1
                # update the value estimate for the current action
                self.reward[run][step] = self.qstar[tmp_action] + np.random.randn()
                qstar = self.qstar[tmp_action] + np.random.randn()
                self.Q[tmp_action] += 1/self.N[tmp_action]*(qstar-self.Q[tmp_action])
        self.mean_reward = np.mean(self.reward,axis=0)        

#%% Test greedy, epsilon-greedy
plt.figure()
epsilons = [0.0,0.01,0.1,0.25]
for epsilon in epsilons:
    multi_arm = multi_arm_bandit_testbed(epsilon=epsilon)
    multi_arm.multi_arm_bandit_run()  
    plt.plot(multi_arm.mean_reward, label='Epsilon = ' + str(epsilon))
    plt.legend('Epsilon = ' + str(epsilon))
plt.title('Mean Reward for ' + str(multi_arm.num_arms) + ' Bandits over ' + str(multi_arm.runs) + ' Runs')
plt.xlabel('Step')
plt.ylabel('Mean Reward')
plt.grid(b=True)
plt.legend()

#%% Test greedy, epsilon greedy and optimistic initial values
plt.figure()
optimistic_val = [0, 5]
epsilons = [0.0, 0.01, 0.1]
for opt_val in optimistic_val:
    for epsilon in epsilons:
        multi_arm = multi_arm_bandit_testbed(epsilon=epsilon, opt_init=opt_val)
        multi_arm.multi_arm_bandit_run()  
        plt.plot(multi_arm.mean_reward, label='Epsilon = '+str(epsilon)+' Initial Value = '+str(opt_val))
        plt.legend(loc='lower right')
plt.title('Mean Reward for '+str(multi_arm.num_arms)+' Bandits over '+str(multi_arm.runs)+' Runs')
plt.xlabel('Step')
plt.ylabel('Mean Reward')
plt.grid(b=True)
plt.legend()       

#%% Test Upper confidence bound against greedy, epsilon-greedy
plt.figure()
parameters = [(0.1, False), (0.0, True)] #(ucb, epsilon)
for param in parameters:
    multi_arm=multi_arm_bandit_testbed(epsilon=param[0], ucb=param[1])
    multi_arm.multi_arm_bandit_run()  
    plt.plot(multi_arm.mean_reward, label='Epsilon = '+str(param[0])+' UCB = '+str(param[1]))
    plt.legend(loc='lower right')
plt.title('Mean Reward for '+str(multi_arm.num_arms)+' Bandits over '+str(multi_arm.runs)+' Runs')
plt.xlabel('Step')
plt.ylabel('Mean Reward')
plt.grid(b=True)
plt.legend()    
#%%END
 
        
    


   