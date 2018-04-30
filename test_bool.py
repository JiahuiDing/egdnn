import egdnn_python as model
import numpy as np
import matplotlib.pylab as plt

data_N = 10000

for N in range(10, 16):
	print('N =', N)
	with open('resultFile.txt', 'a') as f:
		f.write('N = {}\n'.format(N))
	# dataset
	x = np.zeros([data_N, 20])
	y = np.zeros([data_N, 2])

	for i in range(data_N):
		for j in range(20):
			x[i][j] = np.random.randint(0,2)
			
	choice = np.zeros([N,3], dtype = 'int32')
	for i in range(N):
		choice[i] = np.random.choice(20, 3)

	for i in range(data_N):
		y[i][0] = True
		for j in range(N):
			tmp = False
			for k in range(3):
				tmp = tmp or x[i][choice[j][k]]
			y[i][0] = y[i][0] and tmp
		y[i][1] = 1 - y[i][0]
		
	# settings
	input_N = 20
	output_N = 2
	populationSize = 3

	learning_rate = 1e-3
	velocity_decay = 0.9
	regularization_l1 = 1e-2
	regularization_l2 = 1e-2
	rmsprop_rho = -1
	gradientClip = 1

	iterNum = 100
	batchSize = 100

	# model
	model.init(input_N, output_N, populationSize, learning_rate, velocity_decay, regularization_l1, regularization_l2, rmsprop_rho, gradientClip)
	for evolutionCnt in range(300):
		print('evolution', evolutionCnt)
		with open('resultFile.txt', 'a') as f:
			f.write('evolution {}\n'.format(evolutionCnt))
		model.fit(-1, x, y, iterNum, batchSize)
		score = np.zeros(populationSize)
		for netId in range(populationSize):
			score[netId] = model.test(netId, x, y)
		model.display()
		model.evolution(np.argmax(score))
		if model.kbhit():
			break

'''
data_N = 10000

for N in range(10, 11):
	print('N =', N)
	with open('resultFile.txt', 'a') as f:
		f.write('N = {}\n'.format(N))
	# dataset
	x = np.zeros([data_N, 2*N])
	y = np.zeros([data_N, 2])

	for i in range(data_N):
		for j in range(2*N):
			#x[i][j] = np.random.uniform(0,1)
			x[i][j] = np.random.randint(0,2)

	for i in range(data_N):
		y[i][0] = True
		for j in range(N):
			y[i][0] = y[i][0] and (x[i][j*2] or x[i][j*2+1])
		y[i][1] = 1 - y[i][0]
		
	# settings
	input_N = 2*N
	output_N = 2
	populationSize = 3

	learning_rate = 1e-3
	velocity_decay = 0.9
	regularization_l1 = 1e-2
	regularization_l2 = 1e-2
	rmsprop_rho = -1
	gradientClip = 1

	iterNum = 100
	batchSize = 100

	# model
	model.init(input_N, output_N, populationSize, learning_rate, velocity_decay, regularization_l1, regularization_l2, rmsprop_rho, gradientClip)
	for evolutionCnt in range(300):
		print('evolution', evolutionCnt)
		with open('resultFile.txt', 'a') as f:
			f.write('evolution {}\n'.format(evolutionCnt))
		model.fit(-1, x, y, iterNum, batchSize)
		score = np.zeros(populationSize)
		for netId in range(populationSize):
			score[netId] = model.test(netId, x, y)
		model.display()
		model.evolution(np.argmax(score))
		if model.kbhit():
			break
'''
