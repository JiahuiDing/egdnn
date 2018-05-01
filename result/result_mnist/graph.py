import numpy as np
import matplotlib.pyplot as plt

file = open('result_mnist_2.txt', 'r')
lines = file.readlines()

N = []
loss = []
accuracy = []
neuronNum = []
connectionNum = []

for i in range(704):
	line0 = lines[7*i]
	line1 = lines[7*i+1]
	line4 = lines[7*i+4]
	
	N += [int(line0)]
	loss += [float(line1.split()[1])]
	accuracy += [float(line1.split()[2])]
	neuronNum += [int(line4.split()[1])]
	connectionNum += [int(line4.split()[2])]
	
plt.figure()
plt.xlabel('iteration')
plt.ylabel('loss')
plt.plot(N, loss)
plt.show()

plt.figure()
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.plot(N, accuracy)
plt.show()

plt.figure()
plt.xlabel('iteration')
plt.ylabel('number of neurons')
plt.plot(N, neuronNum)
plt.show()

plt.figure()
plt.xlabel('iteration')
plt.ylabel('number of connections')
plt.plot(N, connectionNum)
plt.show()
