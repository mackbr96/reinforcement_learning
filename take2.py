# Playing with OpenAI Gym: CartPole-v0
import math
import time
import gym
import numpy as np
import pandas as pd
import random


##################################################################################################
class Qlearn:
	def __init__(self,buckets=10, alpha=0.001, epsilon=0.001, gamma=1.0, thetaScale=10):
		self.buckets = buckets
		self.thetaScale = thetaScale
		self.alpha_min = alpha # learning rate
		self.epsilon_min = epsilon # exploration rate
		self.alpha = alpha
		self.epsilon = epsilon
		self.gamma = gamma # discount factor
		self.ada_divisor = 5
		self.env = gym.make('CartPole-v0')

		n_obs = 2
		n_actions = self.env.action_space.n

		self.theta_buckets = pd.cut([self.env.observation_space.low[2], self.env.observation_space.high[2]], self.buckets, retbins=True)[1]
		self.thetaPrime_buckets = pd.cut([0, self.thetaScale], self.buckets, retbins=True)[1]

		self.qTable = np.zeros([buckets ** n_obs, n_actions])
		# blah = []
		# for x in self.qTable:
		# 	tri = []
		# 	for y in x:
		# 		y = random.randint(0,1)
		# 		tri.append(y)
		# 	blah.append(tri)
		# self.qTable = np.array(blah)
	

	def update_q(self, state, action, reward, old_state):
		self.qTable[old_state, action] = (1-self.alpha) * self.qTable[old_state, action] + self.alpha * (reward + self.gamma * np.max(self.qTable[state]))
		

	def choose_action(self, state):
		if (np.random.random() <= self.epsilon):
			return self.env.action_space.sample()
		else:
			return np.argmax(self.qTable[state])
              
	def observationToState(self, obs):
		numBuckets = self.buckets
		actions = len(obs)
		return sum([numBuckets ** (actions - i - 1 ) * obs[i] for i in range(len(obs))])

	def updateValues(self, t):
		self.epsilon =  max(self.epsilon_min, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))
		self.alpha = max(self.alpha_min, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))


def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def naive_policy(obs):
	print(obs)
	print("done")
	angle = obs[2]
	return 0 if angle < 0 else 1

def random_policy(obs):
	angle = obs[2]
	return 0 if np.random.uniform() < 0.5 else 1

def better_policy(obs, learn):
	return learn.choose_action(obs[2])


	


##################################################################################################

def naive_main( policy ):
	RL = Qlearn()
	debug = True
	env = gym.make('CartPole-v0')
	env.render()
	avg = 0
	# episodic reinforcement learning
	totals = []
	for episode in range(100):
		current_state = env.reset()
		buckets = [
				np.digitize(current_state[2], RL.theta_buckets  ),
				np.digitize(sigmoid(current_state[3]) * RL.thetaScale, RL.thetaPrime_buckets)
			]
		current_state = RL.observationToState(buckets)
		episode_rewards = 0
		RL.updateValues(episode)
		for step in range(10000):
			#action = policy(test, RL)
			action = RL.choose_action(current_state)
			newObs, reward, done, info = env.step(action)
			
			buckets = [
				np.digitize(newObs[2], RL.theta_buckets  ) - 1,
				np.digitize(sigmoid(newObs[3]) * RL.thetaScale, RL.thetaPrime_buckets) -1
			]
			newObs = RL.observationToState(buckets)
			RL.update_q(newObs, action, reward, current_state)
			
			#env.render()
			current_state = newObs

			#time.sleep(0.1)
			episode_rewards += reward
			if done:
				print ("Game over. Number of steps = ", step)
				#env.render()
				#time.sleep(1)
				break
		totals.append(episode_rewards)
		avg = np.mean(totals)
		#print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))
	print("Max is ",max(totals))
	print("Last 10 avg ", sum([totals[x] for x in range(-1, -10, -1)] )/ 10)
	print("Total Avg is ",np.mean(totals) )
	print(RL.qTable)

##################################################################################################

if __name__ == "__main__":
	naive_main( better_policy )

##################################################################################################
