# pass the test in about 50 iterations

import egdnn_python as model
import gym
import numpy as np
import random
import matplotlib.pyplot as plt

# parameters
alpha = 0.1
gamma = 0.99
memory_size = 5000
episode_num = 100000
env = gym.make('CartPole-v0')

class Memory:
	class Data:
		def __init__(self, observation_now = np.zeros(4), observation_next = np.zeros(4), action = 0, reward = 0, done = False):
			self.observation_now = observation_now
			self.observation_next = observation_next
			self.action = action
			self.reward = reward
			self.done = done
	
	def __init__(self, size):
		self.size = size
		self.memory = [Memory.Data() for _ in range(memory_size)] # memory for storing learning data
		self.full = False # if memory is not full, we can only use memory in range (0, pos - 1), otherwise we can use any memory
		self.pos = 0 # for insert new memory
	
	def insert(self, observation_now, observation_next, action, reward, done):
		self.memory[self.pos] = Memory.Data(observation_now, observation_next, action, reward, done)
		self.pos += 1
		if self.pos == self.size:
			self.pos = 0
			self.full = True
	
	def random_get(self):
		'''
		if random.uniform(0,1) < 0.05:
			while True:
				if self.full:
					index = np.random.randint(self.size)
				else:
					index = np.random.randint(self.pos)
				if self.memory[index].done == True:
					break
		else:
			while True:
				if self.full:
					index = np.random.randint(self.size)
				else:
					index = np.random.randint(self.pos)
				if self.memory[index].done == False:
					break
		'''
	
		if self.full:
			index = np.random.randint(self.size)
		else:
			index = np.random.randint(self.pos)
		
		if random.uniform(0,1) < 0.05:
			for i in range(self.size):
				if self.memory[i].done == True:
					index = i
					break
		
		return self.memory[index]

MEM = Memory(memory_size)
result = np.zeros(episode_num)

# settings
input_N = 4
output_N = 2

populationSize = 3
learning_rate = 1e-3
velocity_decay = 0.9
regularization_l2 = 1e-2
gradientClip = 1

iterNum = 1
batchSize = 100

# model
model.init(input_N, output_N, populationSize, learning_rate, velocity_decay, regularization_l2, gradientClip)

for episode_cnt in range(episode_num):
	if episode_cnt % 200 == 0 and episode_cnt != 0:
		print('result average = {}'.format(np.mean(result[episode_cnt-200:episode_cnt])))
		plt.plot(range(episode_cnt), result[:episode_cnt])
		plt.show()
		
	net_score = np.zeros(populationSize)
	for _ in range(10):
		for netId in range(populationSize):
			observation_now = env.reset()
			for t in range(10000):
				# have a play and store the state action sequence
				env.render()
				score = model.predict(netId, np.array(observation_now))
				action = np.argmax(score)
				if random.uniform(0, 1) < max(0.01, 1 - episode_cnt / 20):
					action = np.random.randint(2)
				observation_next, reward, done, info = env.step(action)
				MEM.insert(observation_now, observation_next, action, reward, done)
				observation_now = observation_next
		
				if done or t > 195:
					net_score[netId] += t + 1
					break
					
			# train the model
			if result[episode_cnt - 1] > 196.91 and netId == 0:
				continue
			for _ in range(10):
				data = [Memory.Data() for _ in range(batchSize)]
				observation_now = np.zeros((batchSize, 4))
				observation_next = np.zeros((batchSize, 4))
				for i in range(batchSize):
					data[i] = MEM.random_get()
					observation_now[i] = data[i].observation_now
					observation_next[i] = data[i].observation_next
				x_train = observation_now
				y_train = model.predict_batch(netId, observation_now)
				score = model.predict_batch(netId, observation_next)
				for i in range(batchSize):
					if data[i].done:
						y_train[i][data[i].action] = -1
					else:
						y_train[i][data[i].action] = (1 - alpha) * y_train[i][data[i].action] + alpha * (data[i].reward + gamma * max(score[i]))
				model.fit(netId, x_train, y_train, iterNum, batchSize)
	net_score /= 10
	result[episode_cnt] = np.max(net_score)
	print('episode_cnt = {} : Episode finished after {} timesteps'.format(episode_cnt, result[episode_cnt]))
	print(net_score)
	model.display()
	model.evolution(np.argmax(net_score))
