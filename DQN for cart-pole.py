import gym
import numpy as np
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import time
import random
import matplotlib.pyplot as plt
import getch

# parameters
batch_size = 1000
gamma = 0.99
memory_size = 100000
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
		if self.full:
			index = np.random.randint(self.size)
		else:
			index = np.random.randint(self.pos)
		
		if random.uniform(0,1) < 0.1:
			for i in range(self.size):
				if self.memory[i].done == True:
					index = i
					break
		
		return self.memory[index]

MEM = Memory(memory_size)
result = np.zeros(episode_num)
cnt_0 = np.zeros(episode_num)
cnt_1 = np.zeros(episode_num)

# model
model = Sequential()
model.add(Dense(125, activation = 'relu', input_dim = 5))
model.add(Dense(125, activation = 'relu'))
model.add(Dense(125, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))
model.compile(loss = 'mse', optimizer = 'rmsprop')

for episode_cnt in range(episode_num):
	if episode_cnt % 200 == 0 and episode_cnt != 0:
		print('result average = {}'.format(np.mean(result[episode_cnt-200:episode_cnt])))
		print('cnt_0 = {}'.format(np.sum(cnt_0[episode_cnt-200:episode_cnt])))
		print('cnt_1 = {}'.format(np.sum(cnt_1[episode_cnt-200:episode_cnt])))
		plt.plot(range(episode_cnt), result[:episode_cnt])
		plt.show()
	observation_now = env.reset()
	for t in range(10000):
		# have a play and store the state action sequence
		#if episode_cnt % 10 == 0:
		env.render()
		input_0 = np.append(observation_now, np.array([0]))
		input_1 = np.append(observation_now, np.array([1]))
		score = model.predict(np.array([input_0, input_1]))
		if score[0] > score[1]:
			action = 0
			cnt_0[episode_cnt] += 1
		else:
			action = 1
			cnt_1[episode_cnt] += 1
		if random.uniform(0, 1) < 0.1 - episode_cnt / 1000:
			action = np.random.randint(2)
		observation_next, reward, done, info = env.step(action)
		MEM.insert(observation_now, observation_next, action, reward, done)
		observation_now = observation_next
		
		if done or t > 195:
			result[episode_cnt] = t + 1
			print('episode_cnt = {} : Episode finished after {} timesteps'.format(episode_cnt, t+1))
			break
	# train the model
	data = [Memory.Data() for _ in range(batch_size)]
	x_train = np.zeros((batch_size, 5))
	input_0 = np.zeros((batch_size, 5))
	input_1 = np.zeros((batch_size, 5))
	for i in range(batch_size):
		data[i] = MEM.random_get()
		x_train[i] = np.append(data[i].observation_now, np.array([data[i].action]))
		input_0[i] = np.append(data[i].observation_next, np.array([0]))
		input_1[i] = np.append(data[i].observation_next, np.array([1]))
	score_0 = model.predict(input_0)
	score_1 = model.predict(input_1)
	y_train = np.zeros((batch_size,))
	for i in range(batch_size):
		if data[i].done:
			y_train[i] = -1
		else:
			y_train[i] = data[i].reward + gamma * max(score_0[i], score_1[i])
	model.fit(x_train, y_train, epochs = 1, batch_size = 100, verbose = 0)
