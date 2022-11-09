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
environment_dim = [2,3]
passenger = [0,1] #0: origin, 1:in Taxi
actions = (0, 1, 2, 3, 4, 5, 6)  # actions (0=up, 1=down, 2=left, 3=right, 4=pickup, 5=dropoff, 6=charge )
num_actions = len(actions)
battery = [10,20,30,40,50,60,70,80,90,100]
num_states =  2*3*2*10# taxi_row * taxi_column * passenger_status * battery_level

Q = [[1 for k in range (0,num_actions)]  for i in range (0,num_states)]	
rewards = [[0 for k in range (0,num_actions)]  for i in range (0,num_states)]	
s_next = [[0 for k in range (0,num_actions)]  for i in range (0,num_states)]	

# building states
states =[]
state = namedtuple('state' , 'id, taxi_row, taxi_column, passenger_status, battery_level')




# For later use 
def distance (row, col) :
	dist = -1
	max=-1
	min=-1
	row_org =row
	col_org = col
	row_dest= 2
	col_dest = 2
	return (abs(row_org-row_dest) + abs(col_org-col_dest))




def reward_matrix (s,a):

	# illegal actions got -10
	if a==0 and states[s].taxi_row==1:
		return -100
	if a==1 and states[s].taxi_row==0:
		return -100
	if a==2 and states[s].taxi_column==0:
		return -100
	if a==3 and states[s].taxi_column==2:
		return -100


	reward = -1

	if a ==4: #pick-up
		if states[s].passenger_status==0 and states[s].taxi_row==0 and states[s].taxi_column==2:
			reward = 10
		else:
			reward = -10

	if a==5:#dropoff
		if states[s].passenger_status==1 and states[s].taxi_row==1 and states[s].taxi_column==0:
			reward = 20
		else:
			reward = -10

	if a==6:#charge
		if  states[s].taxi_row==1 and states[s].taxi_column==1:
			reward = -1
		else:
			reward = -20
		if states[s].battery_level<=30:
			reward = reward + 5


	print ("Reward before =",reward)
	depreciation = reward * (states[s].battery_level/100)
	reward = reward - ( abs(reward) - abs(depreciation) )
	print ("Reward after =",reward)


	return reward



# creat all states
ids = 0
for taxi_row in range(0,environment_dim[0]):
	for taxi_col in range (0,environment_dim[1]):
		for passenger_status in range(0,2):
			for bat in range (10,101,10):
				s = state (ids,taxi_row, taxi_col, passenger_status, bat) 
				states.append (s)
				ids = ids+1



for s in range (0,num_states):
	for act in range (0,len(actions)):

		
		##################################################################################################
		##################################################################################################
		#												UP 												 #
		##################################################################################################
		##################################################################################################
		if act == 0 : #Up
			if states[s].taxi_row == 1: # top states 
				s_next[s][act]= states[s].id #same state

			else: # not top states (col= col, row =row+1, bat = bat-20)
				for s_ in range (0,num_states):
					if states[s_].taxi_row == states[s].taxi_row+1 and states[s_].taxi_column == states[s].taxi_column and states[s_].passenger_status == states[s].passenger_status  and states[s_].battery_level == states[s].battery_level-10:
						s_next[s][act] = states[s_].id
						break
					s_next[s][act]= states[s].id # if didnt find in for .. 


				

		##################################################################################################
		##################################################################################################
		#											Down 												 #
		##################################################################################################
		##################################################################################################

		if act == 1 : #Down
			if states[s].taxi_row == 0: # down states 
				s_next[s][act]= states[s].id #same state


			else: # not bottom states (col= col, row =row-1, bat = bat-20)
				for s_ in range (0,num_states):
					if states[s_].taxi_row == states[s].taxi_row-1 and states[s_].taxi_column == states[s].taxi_column and states[s_].passenger_status == states[s].passenger_status  and states[s_].battery_level == states[s].battery_level-10:
						s_next[s][act] = states[s_].id
						break
					s_next[s][act]= states[s].id

		##################################################################################################
		##################################################################################################
		#												LEFT											 #
		##################################################################################################
		##################################################################################################

		if act == 2 : #Left
			if states[s].taxi_column == 0: # lefmost states 
				s_next[s][act]= states[s].id #same state

			else: # not top states (col= col, row =row-1, bat = bat-20)
				for s_ in range (0,num_states):
					if states[s_].taxi_row == states[s].taxi_row and states[s_].taxi_column == states[s].taxi_column-1 and states[s_].passenger_status == states[s].passenger_status  and states[s_].battery_level == states[s].battery_level-10:
						s_next[s][act] = states[s_].id
						break
					s_next[s][act]= states[s].id


		##################################################################################################
		##################################################################################################
		#												Right 											 #
		##################################################################################################
		##################################################################################################

		if act == 3 : #Right
			if states[s].taxi_column == 2: # rightmost states 
				s_next[s][act]= states[s].id #same state
				

			else: # not top states (col= col, row =row-1, bat = bat-20)
				for s_ in range (0,num_states):
					if states[s_].taxi_row == states[s].taxi_row and states[s_].taxi_column == states[s].taxi_column+1 and states[s_].passenger_status == states[s].passenger_status  and states[s_].battery_level == states[s].battery_level-10:
						s_next[s][act] = states[s_].id
						break
					s_next[s][act]= states[s].id
	

		##################################################################################################
		##################################################################################################
		#											PICK UP												 #
		##################################################################################################
		##################################################################################################
		if act == 4 : #Pickup
			if states[s].passenger_status==1 or states[s].taxi_row!=0 or states[s].taxi_column!=2 : #already has passenger onboard!!!!!
				s_next[s][act]=states[s].id #same state
			else: 
				for s_ in range (0,num_states):
					if states[s_].taxi_row == states[s].taxi_row and states[s_].taxi_column == states[s].taxi_column and states[s_].passenger_status ==1  and states[s_].battery_level == states[s].battery_level:
						s_next[s][act] = states[s_].id
						break
					s_next[s][act]= states[s].id

		##################################################################################################
		##################################################################################################
		#											DROP OFF											 #
		##################################################################################################
		##################################################################################################
		if act == 5 : #Dropoff
			if states[s].passenger_status==0 or states[s].taxi_row!=1 or states[s].taxi_column!=0 : #already has passenger onboard!!!!!
				s_next[s][act]=states[s].id #same state
			else: 
				for s_ in range (0,num_states):
					if states[s_].taxi_row == states[s].taxi_row and states[s_].taxi_column == states[s].taxi_column and states[s_].passenger_status ==0  and states[s_].battery_level == states[s].battery_level:
						s_next[s][act] = states[s_].id
						break
					s_next[s][act]= states[s].id


		##################################################################################################
		##################################################################################################
		#											CHARGE												 #
		##################################################################################################
		##################################################################################################
		if act == 6 : #Charge
			for s_ in range (0,num_states):
				if states[s_].taxi_row == states[s].taxi_row and states[s_].taxi_column == states[s].taxi_column and states[s_].passenger_status == states[s].passenger_status and states[s_].battery_level==100:
					s_next[s][act] = states[s_].id
					break
				#s_next[s][act]= states[s].id
		
		rewards[s][act]=reward_matrix(s,act)



print (len(states))


# Set policy iteration parameters
max_policy_iter = 10000  # Maximum number of policy iterations
max_value_iter = 10000  # Maximum number of value iterations


num_iter = 0 


# to help in finding the index of the next state (s') where a specific action leads to.
def search (st, act):
	return s_next[st][act]


def e_greedy (s,epsilon):
	r = random.random()
	if (r<epsilon):
		return random.randint(0,5) # random action
	else:
		return np.argmax(Q[s])


def update_policy (policy):
	p = policy
	for row in range(0,len(Q)):
		p[row] = np.argmax(Q[row])	
	return p


def schedule_hyperparameters(timestep, max_timestep): 
	max_deduct, decay = 0.95, 0.07
	epsilon = 1.0 - (min(1.0, timestep/(decay * max_timestep))) * max_deduct
	return epsilon

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
    "epsilon": 0.0,
}


epsilon = CONFIG["epsilon"]
learning_rate = CONFIG["alpha"]
gamma=CONFIG["gamma"]
policy = [0 for s in range(0,num_states)]
reward_during_learning = []

for j in range (0,1000):
	average_rewards = 0 
	s = random.randint(0,num_states-1)
	for i in range (0,1000): 
		a = e_greedy(s,epsilon)
		Q[s][a] = Q[s][a] + learning_rate*(rewards[s][a]+ gamma* np.amax(Q[s_next[s][a]]) - Q[s][a])	
		average_rewards =Q[s][a]
		s = s_next[s][a]

	epsilon = schedule_hyperparameters(j,1000)
	reward_during_learning.append(average_rewards)


#############################################################
#						Print results						#
#############################################################
policy = update_policy (policy)
counter = 0 


s=-1
for i in range (0,len(states)):
	if states[i].taxi_row==0 and states[i].taxi_column==0 and states[i].passenger_status==0 and states[i].battery_level==40:
		s=i
		break

#s=random.randint(0,12)

while counter <20:
	print("loc (",states[s].taxi_row,states[s].taxi_column,") passenger (",states[s].passenger_status,", BAT=(",states[s].battery_level,"), Go", policy[s])
	
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
	s = s_next[s][policy[s]]

	counter = counter+1


x_axis = np.zeros(len(reward_during_learning))
for i in range (0,len(reward_during_learning)):
	x_axis[i] = i

y_axies = reward_during_learning
plt.plot(x_axis, y_axies, label = "rewards")
plt.xlabel('#Episodes')
plt.ylabel('Accumulated Rewards')
plt.title('QLearning')
plt.legend()
plt.show()
