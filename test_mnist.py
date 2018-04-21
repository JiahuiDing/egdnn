import egdnn_python as model
import numpy as np
import keras
from keras.datasets import mnist
import matplotlib.pylab as plt

# settings
input_N = 784
output_N = 10
populationSize = 3

learning_rate = 1e-3
velocity_decay = 0.9
regularization_l1 = 1e-2
regularization_l2 = 1e-2
rmsprop_rho = -1
gradientClip = 1

iterNum = 5
batchSize = 100

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_train = x_train.astype('float64')
x_train /= 255
y_train = keras.utils.to_categorical(y_train, output_N)

x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float64')
x_test /= 255
y_test = keras.utils.to_categorical(y_test, output_N)

# model
model.init(input_N, output_N, populationSize, learning_rate, velocity_decay, regularization_l1, regularization_l2, rmsprop_rho, gradientClip)
for evolutionCnt in range(100000):
	print('evolution', evolutionCnt)
	model.fit(-1, x_train[:50000], y_train[:50000], iterNum, batchSize)
	score = np.zeros(populationSize)
	for netId in range(populationSize):
		score[netId] = model.test(netId, x_train[59000:], y_train[59000:])
	model.display()
	model.evolution(np.argmax(score))
	if model.kbhit():
		break

prediction = np.zeros([populationSize, 10, output_N])
for netId in range(populationSize):
	prediction[netId] = model.predict_batch(netId, x_train[:10])
for i in range(10):
	for netId in range(populationSize):
		print(prediction[netId][i])
	plt.imshow(x_train[i].reshape(28,28), cmap = 'gray')
	plt.show()
