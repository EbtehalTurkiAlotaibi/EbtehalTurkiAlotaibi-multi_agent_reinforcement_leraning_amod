from collections import namedtuple
import random
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

class Environment(object):
    def __init__(self, seed=None):
        self.episode_limit = 2*3*2*10 # additional 2 for agents
        self.n_agents = 2 # number of taxis
        self.n_actions = 7
        self._episode_steps = 0
        self.last_action = [np.zeros(self.n_actions) for _ in range(self.n_agents)]
        self.fileresults = open('learning.data', "w")
        self.run = 0
        self.battery = [10,20,30,40,50,60,70,80,90,100]
        self.num_states =  2*3*2*10# taxi_row * taxi_column * passenger_status * battery_level
        self.passenger = [0,1] #0: origin, 1:in Taxi
        self.environment_dim = [2,3]
        # building states
        self.state = namedtuple('state' , 'id, taxi_row, taxi_column, passenger_status, battery_level')
        self.reset() 
        super(Environment, self).__init__()
   
    def step(self, actions):
        
        self._episode_steps += 1
        # compute rewards
        r=0 
        # Special implementation for two agents:
        act1 = actions[0]
        act2 = actions[1]
        for i in range(self.n_agents):  ####### HERE IS MY change !!!!!!!!!
            r +=self.calculate_reward(self.states[i][self._episode_steps], actions[i])
            if act1 == act2 and act1 == 4: # both pickup
                r = -100000
            if act1 == act2 and act1 == 5: # both dropoff
                r = -100000
            if act1 == 5 and act2 != 5: # both dropoff
                r +=10000

            if act2 == 5 and act1 != 5: # both dropoff
                r +=10000

        reward = r

        self.rinfo[0] +=reward
        if self._episode_steps >= self.episode_limit-1:
            terminated = True
        else:
            terminated = False
        
        info = {}
        self.obs = [self.states_values[i][self._episode_steps] for i in range(self.n_agents)]
        return  reward, terminated, info
    
    
    def calculate_reward(self,state,action):
        # illegal actions got -10, illegal actions are allowed.
        if action==0 and state.taxi_row==1:
            return -100
        if action==1 and state.taxi_row==0:
            return -100
        if action==2 and state.taxi_column==0:
            return -100
        if action==3 and state.taxi_column==2:
            return -100

        reward = -1

        if action==4: #pick-up
            if state.passenger_status==0 and state.taxi_row==0 and state.taxi_column==2:
                reward = 10
            else:
                reward = -10

        if action==5:#dropoff
            if state.passenger_status==1 and state.taxi_row==1 and state.taxi_column==0:
                reward = 20
            else:
                reward = -10

        if action==6:#charge
            if  state.taxi_row==1 and state.taxi_column==1:
                reward = -1
            else:
                reward = -20
            if state.battery_level<=30:
                reward = reward + 5


        # print ("Reward before =",reward)
        depreciation = reward * (state.battery_level/100)
        reward = reward - ( abs(reward) - abs(depreciation) )
        # print ("Reward after =",reward)

        # next state part

        return reward

    def get_obs(self):
        """ Returns all agent observations in a list """
        # this observation for all agent  (list of obs)
        obs_n = self.obs
        return obs_n


    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        #one agent
        return self.get_obs()[agent_id]


    def get_obs_size(self):
        """ Returns the shape of the observation """
        return len(self.get_obs_agent(0))


    def get_state(self):
        state = np.concatenate(self.get_obs())
        state = state.astype(dtype=np.float32)

        return state
        
    def get_state_size(self):
        """ Returns the shape of the state"""
        state_size = self.get_obs_size() * self.n_agents
        return state_size


    def get_avail_actions(self):
        #in this enviroment alowase there is action so this why we return once
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]
 
 
    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(self.n_actions)


    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.n_actions
 
 
    def reset(self):
        """ Returns initial observations and states"""
        self._episode_steps = 0
        if self.run == 0:
            print('Simulation Start')
            self.close()
        else:
            self.end_episode()
        
        self.run += 1
        # for multi agents, this would be generalized to all agents
        # states =[[], []]
        self.states =[[], []]
        self.states_values =[[], []]
        # creat all states
        ids = np.zeros(2)
        for i in range(self.n_agents):
            for taxi_row in range(0,self.environment_dim[0]):
                for taxi_col in range (0,self.environment_dim[1]):
                    for passenger_status in range(0,2):
                        for bat in range (10,101,10):
                            s = self.state(ids[i],taxi_row, taxi_col, passenger_status, bat)
                            self.states_values[i].append([ids[i], taxi_row, taxi_col, passenger_status, bat])
                            self.states[i].append(s)
                            ids[i] = ids[i]+1

        state = [self.states_values[i][self._episode_steps] for i in range(self.n_agents)]
        self.obs = state
        self.rinfo = np.array([0.0])

        return state, self.get_state()

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        return None

    def save_replay(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def end_episode(self):
        self.fileresults.write(','.join(self.rinfo.flatten().astype('str')) + '\n')
        self.fileresults.flush()
