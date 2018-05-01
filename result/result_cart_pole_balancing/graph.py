import numpy as np
import matplotlib.pyplot as plt

file = open('result_cart_pole_balancing_19.txt', 'r')
lines = file.readlines()

N = []
balancingCnt = []
neuronNum = []
connectionNum = []

for i in range(327):
	line0 = lines[5*i]
	line1 = lines[5*i+1]
	line2 = lines[5*i+2]
	
	N += [int(line0) * 10]
	balancingCnt += [float(line1)]
	neuronNum += [int(line2.split()[1])]
	connectionNum += [int(line2.split()[2])]
	
plt.figure()
plt.xlabel('iteration')
plt.ylabel('balancing count')
plt.plot(N, balancingCnt)
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
