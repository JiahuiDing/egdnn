import egdnn_python as model
import numpy as np
import matplotlib.pylab as plt

# dataset
x = np.zeros([64, 6])
y = np.zeros([64, 2])

for i in range(64):
	for j in range(6):
		x[i][j] = (i >> j) & 1

for i in range(64):
	y[i][0] = (x[i][0] or x[i][1]) and (x[i][2] or x[i][3]) or (x[i][4] and x[i][5])
	y[i][1] = 1 - y[i][0]
	
for i in range(64):
	print(x[i], end = ' ')
	print(y[i])


# settings
input_N = 6
output_N = 2
populationSize = 10

learning_rate = 1e-3
velocity_decay = 0.9
regularization_l1 = 1e-3
regularization_l2 = 1e-3
rmsprop_rho = -1
gradientClip = 1

iterNum = 100
batchSize = 100

# model
model.init(input_N, output_N, populationSize, learning_rate, velocity_decay, regularization_l1, regularization_l2, rmsprop_rho, gradientClip)
for evolutionCnt in range(100000):
	print('evolution', evolutionCnt)
	model.fit(-1, x, y, iterNum, batchSize)
	score = np.zeros(populationSize)
	for netId in range(populationSize):
		score[netId] = model.test(netId, x, y)
	model.display()
	model.evolution(np.argmax(score))
	if model.kbhit():
		break

y_predict = model.predict_batch(0, x)
for i in range(64):
	print(x[i], end = ' ')
	print(y[i], end = ' ')
	print(y_predict[i])
