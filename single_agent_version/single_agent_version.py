import pprint
from collections import namedtuple
import random
import numpy as np
import math

import matplotlib.pyplot as plt
# This code represent Taxi problem when the Taxi should deliver passenger if the battery level is > 30%
# when the battery is <= 30% the charging action has the higest priority ! 
# The code is working properly but the problem is that there is a discountinouse action when the battery falls to 30% and the Taxi
# has a passenger, in this case, the agent decided to charge with the passenger !! 


###############################
# Home  |  charger |	      #
# ----------------------------#
#       |		   |passenger #
###############################


# Environment coding 

################
# 10 | 11 | 12 #
# -------------#
# 00 | 01 | 02 #
################

class Environment(object):
	def __init__(self):
		self.environment_dim = [2,3]
		self.passenger = [0,1] #0: origin, 1:in Taxi
		self.actions = (0, 1, 2, 3, 4, 5, 6)  # actions (0=up, 1=down, 2=left, 3=right, 4=pickup, 5=dropoff, 6=charge )
		self.num_actions = len(self.actions)
		self.battery = [10,20,30,40,50,60,70,80,90,100]
		self.num_states =  2*3*2*10# taxi_row * taxi_column * passenger_status * battery_level

		self.Q = [[1 for k in range (0,self.num_actions)]  for i in range (0,self.num_states)]	
		self.rewards = [[0 for k in range (0,self.num_actions)]  for i in range (0,self.num_states)]	
		self.s_next = [[0 for k in range (0,self.num_actions)]  for i in range (0,self.num_states)]	

		# building states
		self.states =[]
		self.state = namedtuple('state' , 'id, taxi_row, taxi_column, passenger_status, battery_level')




	def reward_matrix (self,s,a):

		# illegal actions got -10
		if a==0 and self.states[s].taxi_row==1:
			return -100
		if a==1 and self.states[s].taxi_row==0:
			return -100
		if a==2 and self.states[s].taxi_column==0:
			return -100
		if a==3 and self.states[s].taxi_column==2:
			return -100


		reward = -1

		if a ==4: #pick-up
			if self.states[s].passenger_status==0 and self.states[s].taxi_row==0 and self.states[s].taxi_column==2:
				reward = 10
			else:
				reward = -10

		if a==5:#dropoff
			if self.states[s].passenger_status==1 and self.states[s].taxi_row==1 and self.states[s].taxi_column==0:
				reward = 20
			else:
				reward = -10

		if a==6:#charge
			if  self.states[s].taxi_row==1 and self.states[s].taxi_column==1:
				reward = -1
			else:
				reward = -10

		depreciation = reward * (self.states[s].battery_level/100)
		reward = reward - ( abs(reward) - abs(depreciation) )

		return reward


	def reset(self):
		# creat all states
		ids = 0
		for taxi_row in range(0,self.environment_dim[0]):
			for taxi_col in range (0,self.environment_dim[1]):
				for passenger_status in range(0,2):
					for bat in range (10,101,10):
						s = self.state (ids,taxi_row, taxi_col, passenger_status, bat) 
						self.states.append (s)
						ids = ids+1



		for s in range (0,self.num_states):
			for act in range (0,len(self.actions)):

				
				##################################################################################################
				##################################################################################################
				#												UP 												 #
				##################################################################################################
				##################################################################################################
				if act == 0 : #Up
					if self.states[s].taxi_row == 1: # top self.states 
						self.s_next[s][act]= self.states[s].id #same state

					else: # not top self.states (col= col, row =row+1, bat = bat-20)
						for s_ in range (0,self.num_states):
							if self.states[s_].taxi_row == self.states[s].taxi_row+1 and self.states[s_].taxi_column == self.states[s].taxi_column and self.states[s_].passenger_status == self.states[s].passenger_status  and self.states[s_].battery_level == self.states[s].battery_level-10:
								self.s_next[s][act] = self.states[s_].id
								break
							self.s_next[s][act]= self.states[s].id # if didnt find in for .. 


						

				##################################################################################################
				##################################################################################################
				#											Down 												 #
				##################################################################################################
				##################################################################################################

				if act == 1 : #Down
					if self.states[s].taxi_row == 0: # down self.states 
						self.s_next[s][act]= self.states[s].id #same state


					else: # not bottom self.states (col= col, row =row-1, bat = bat-20)
						for s_ in range (0,self.num_states):
							if self.states[s_].taxi_row == self.states[s].taxi_row-1 and self.states[s_].taxi_column == self.states[s].taxi_column and self.states[s_].passenger_status == self.states[s].passenger_status  and self.states[s_].battery_level == self.states[s].battery_level-10:
								self.s_next[s][act] = self.states[s_].id
								break
							self.s_next[s][act]= self.states[s].id

				##################################################################################################
				##################################################################################################
				#												LEFT											 #
				##################################################################################################
				##################################################################################################

				if act == 2 : #Left
					if self.states[s].taxi_column == 0: # lefmost self.states 
						self.s_next[s][act]= self.states[s].id #same state

					else: # not top self.states (col= col, row =row-1, bat = bat-20)
						for s_ in range (0,self.num_states):
							if self.states[s_].taxi_row == self.states[s].taxi_row and self.states[s_].taxi_column == self.states[s].taxi_column-1 and self.states[s_].passenger_status == self.states[s].passenger_status  and self.states[s_].battery_level == self.states[s].battery_level-10:
								self.s_next[s][act] = self.states[s_].id
								break
							self.s_next[s][act]= self.states[s].id


				##################################################################################################
				##################################################################################################
				#												Right 											 #
				##################################################################################################
				##################################################################################################

				if act == 3 : #Right
					if self.states[s].taxi_column == 2: # rightmost self.states 
						self.s_next[s][act]= self.states[s].id #same state
						

					else: # not top self.states (col= col, row =row-1, bat = bat-20)
						for s_ in range (0,self.num_states):
							if self.states[s_].taxi_row == self.states[s].taxi_row and self.states[s_].taxi_column == self.states[s].taxi_column+1 and self.states[s_].passenger_status == self.states[s].passenger_status  and self.states[s_].battery_level == self.states[s].battery_level-10:
								self.s_next[s][act] = self.states[s_].id
								break
							self.s_next[s][act]= self.states[s].id
			

				##################################################################################################
				##################################################################################################
				#											PICK UP												 #
				##################################################################################################
				##################################################################################################
				if act == 4 : #Pickup
					if self.states[s].passenger_status==1 or self.states[s].taxi_row!=0 or self.states[s].taxi_column!=2 : #already has passenger onboard!!!!!
						self.s_next[s][act]=self.states[s].id #same state
					else: 
						for s_ in range (0,self.num_states):
							if self.states[s_].taxi_row == self.states[s].taxi_row and self.states[s_].taxi_column == self.states[s].taxi_column and self.states[s_].passenger_status ==1  and self.states[s_].battery_level == self.states[s].battery_level:
								self.s_next[s][act] = self.states[s_].id
								break
							self.s_next[s][act]= self.states[s].id

				##################################################################################################
				##################################################################################################
				#											DROP OFF											 #
				##################################################################################################
				##################################################################################################
				if act == 5 : #Dropoff
					if self.states[s].passenger_status==0 or self.states[s].taxi_row!=1 or self.states[s].taxi_column!=0 : #already has passenger onboard!!!!!
						self.s_next[s][act]=self.states[s].id #same state
					else: 
						for s_ in range (0,self.num_states):
							if self.states[s_].taxi_row == self.states[s].taxi_row and self.states[s_].taxi_column == self.states[s].taxi_column and self.states[s_].passenger_status ==0  and self.states[s_].battery_level == self.states[s].battery_level:
								self.s_next[s][act] = self.states[s_].id
								break
							self.s_next[s][act]= self.states[s].id


				##################################################################################################
				##################################################################################################
				#											CHARGE												 #
				##################################################################################################
				##################################################################################################
				if act == 6 : #Charge
					for s_ in range (0,self.num_states):
						if self.states[s_].taxi_row == self.states[s].taxi_row and self.states[s_].taxi_column == self.states[s].taxi_column and self.states[s_].passenger_status == self.states[s].passenger_status and self.states[s_].battery_level==100:
							self.s_next[s][act] = self.states[s_].id
							break
						#self.s_next[s][act]= self.states[s].id
				
				self.rewards[s][act]=self.reward_matrix(s,act)
	def search (self,st, act):
		return self.s_next[st][act]


	def e_greedy (self,s,epsilon):
		r = random.random()
		if (r<epsilon):
			return random.randint(0,5) # random action
		else:
			return np.argmax(self.Q[s])


	def update_policy (self):
		p = []
		for row in range(0,len(self.Q)):
			p.append(np.argmax(self.Q[row]))
		return p


	def schedule_hyperparameters(self,timestep, max_timestep): 
		max_deduct, decay = 0.95, 0.07
		epsilon = 1.0 - (min(1.0, timestep/(decay * max_timestep))) * max_deduct
		return epsilon




#############################################################
#					  statistics 							#
#############################################################
def statistics_illegal (a,s,env):
	if a==0 and env.states[s].taxi_row==1:
		return 1
	if a==1 and env.states[s].taxi_row==0:
		return 1
	if a==2 and env.states[s].taxi_column==0:
		return 1
	if a==3 and env.states[s].taxi_column==2:
		return 1
		
	if a ==4: #pick-up
		if env.states[s].passenger_status==0 and env.states[s].taxi_row==0 and env.states[s].taxi_column==2:
			return 0 
		else:
			return 1

	if a==5:#dropoff
		if env.states[s].passenger_status==1 and env.states[s].taxi_row==1 and env.states[s].taxi_column==0:
			return 0
		else:
			return 1
			
	return 0

def statistics_outChharge (a,s,env):
	if env.states[s].battery_level==20 and a!=6: 
		return 1
	else:
		return 0
#############################################################
#							QLearning						#
#############################################################
CONFIG = {
    "env": "Taxi-v3",
    "total_eps": 10000,
    "eps_max_steps": 100,
    "eval_episodes": 500,
    "eval_freq": 1000,
    "gamma": 0.99,
    "alpha": 0.5,
    "epsilon": 1.0,
    "num_iter": 1000,
    "epoch_num":1000
}

def train (CONFIG, env):
	epsilon = CONFIG["epsilon"]
	learning_rate = CONFIG["alpha"]
	gamma=CONFIG["gamma"]
	num_iter = CONFIG["num_iter"]
	epoch_num = CONFIG["epoch_num"]
	policy = [0 for s in range(0,env.num_states)]
	reward_during_learning = []
	total_illegal_moves = []
	total_ofCharge = [] 
	for j in range (0,num_iter):
		average_rewards = 0 
		illegal_moves = 0
		out_of_charge = 0 
		s = random.randint(0,env.num_states-1)
		for i in range (0,epoch_num): 
			a = env.e_greedy(s,epsilon)
			illegal_moves = illegal_moves + statistics_illegal(a,s,env)
			out_of_charge = out_of_charge + statistics_outChharge (a,s,env)
			env.Q[s][a] = env.Q[s][a] + learning_rate*(env.rewards[s][a]+ gamma* np.amax(env.Q[env.s_next[s][a]]) - env.Q[s][a])	
			average_rewards =env.Q[s][a]
			s = env.s_next[s][a]

		epsilon = env.schedule_hyperparameters(j,num_iter)
		reward_during_learning.append(average_rewards)
		total_illegal_moves.append(illegal_moves)
		total_ofCharge.append(out_of_charge)

	return reward_during_learning, total_illegal_moves, total_ofCharge

#############################################################
#						Print results						#
#############################################################
def print_policy(env):
	policy = env.update_policy ()
	counter = 0 
	s=-1
	for i in range (0,len(env.states)):
		if env.states[i].taxi_row==0 and env.states[i].taxi_column==0 and env.states[i].passenger_status==0 and env.states[i].battery_level==40:
			s=i
			break
	while counter <20:
		print("loc (",env.states[s].taxi_row,env.states[s].taxi_column,") passenger (",env.states[s].passenger_status,", BAT=(",env.states[s].battery_level,"), Go", policy[s])
		
		if (policy[s]==0):
			print("UP")
		if (policy[s]==1):
			print("DOWN")
		if (policy[s]==2):
			print("LEFT")
		if (policy[s]==3):
			print("RIGHT")
		if (policy[s]==4):
			print("Pick Up")
		if (policy[s]==5):
			print("Drop Off")
			break
		if (policy[s]==6):
			print("Charge")
			#break
		s = env.s_next[s][policy[s]]

		counter = counter+1

#############################################################
#						Plot statistics 					#
#############################################################

def plot_results (reward_during_learning,total_illegal_moves,total_ofCharge):
	reward_during_learning = reward_during_learning[1:]
	x_axis1 = np.zeros(len(reward_during_learning))
	for i in range (0,len(reward_during_learning)):
		x_axis1[i] = i

	y_axies1 = reward_during_learning
	plt.plot(x_axis1, y_axies1, label = "rewards")
	plt.xlabel('#Episodes')
	plt.ylabel('Accumulated Rewards')
	plt.title('QLearning')
	plt.legend()
	plt.savefig('single_agent_results/reward_during_learning.png')
	plt.clf()


	total_illegal_moves = total_illegal_moves[1:]
	x_axis2 = np.zeros(len(total_illegal_moves))
	for i in range (0,len(total_illegal_moves)):
		x_axis2[i] = i

	y_axies2 = total_illegal_moves
	plt.plot(x_axis2, y_axies2, label = "total illegal moves")
	plt.xlabel('#Episodes')
	plt.ylabel('Total Illegal Moves')
	plt.title('QLearning')
	plt.legend()
	plt.savefig('single_agent_results/total_illegal_moves.png')
	plt.clf()

	total_ofCharge = total_ofCharge[1:]
	x_axis3 = np.zeros(len(total_ofCharge))
	for i in range (0,len(total_ofCharge)):
		x_axis3[i] = i

	y_axies3 = total_ofCharge
	plt.plot(x_axis3, y_axies3, label = "total out of charge event")
	plt.xlabel('#Episodes')
	plt.ylabel('Total out of Charge events')
	plt.title('QLearning')
	plt.legend()
	plt.savefig('single_agent_results/total_ofCharge.png')
	plt.clf()
	# Using Numpy to create an array X
	# Plotting both the curves simultaneously
	plt.plot(x_axis1, y_axies1, color='r', label='rewards')
	plt.plot(x_axis2, y_axies2, color='b', label='total illegal moves')
	plt.plot(x_axis3, y_axies3, color='g', label='total out of charge event')
	  
	# Naming the x-axis, y-axis and the whole graph
	plt.xlabel("Episodes")
	plt.ylabel("Training Behaviour Descriptors")
	plt.title("Training Data")
	  
	# Adding legend, which helps us recognize the curve according to it's color
	plt.legend()
	  
	# To load the display window
	plt.savefig('single_agent_results/behaviour_descriptors.png')

if __name__ == '__main__':
	env = Environment()
	env.reset()
	reward_during_learning, total_illegal_moves, total_ofCharge = train (CONFIG, env)
	reward_during_learning = reward_during_learning[1:]
	total_illegal_moves = total_illegal_moves[1:]
	total_ofCharge = total_ofCharge[4:]
	plot_results (reward_during_learning,total_illegal_moves,total_ofCharge)
	print_policy(env)





