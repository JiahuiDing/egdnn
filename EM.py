import matplotlib.pyplot as plt
import numpy as np # only use numpy to perform argmax during test

# load trainData
trainDataFile = open('traindata.txt', 'r')
trainData = []
for data in trainDataFile:
	trainData.append([int(feature) for feature in data.split()])

# load testData
testDataFile = open('testdata.txt', 'r')
testData = []
for data in testDataFile:
	testData.append([int(feature) for feature in data.split()])

# give a number for each variable
# 0 : Sloepnea
# 1 : Foriennditis
# 2 : Degar spots
# 3 : TRIMONO-HT/S
# 4 : Dunetts Syndrome

# construct 5 initial CPTs
# for convinence, those CPTs are global variable
CPT = [{} for i in range(5)]
CPT_variable = [[] for i in range(5)]

CPT_variable[0] = [0, 3, 4]
CPT[0] = {'110' : 0.01,
		'111' : 0.01,
		'112' : 0.01,
		'100' : 0.02,
		'101' : 0.6,
		'102' : 0.6,
		'010' : 0.99,
		'011' : 0.99,
		'012' : 0.99,
		'000' : 0.98,
		'001' : 0.4,
		'002' : 0.4}

CPT_variable[1] = [1, 4]
CPT[1] = {'10' : 0.02,
		'11' : 0.8,
		'12' : 0.3,
		'00' : 0.98,
		'01' : 0.2,
		'02' : 0.7}

CPT_variable[2] = [2, 4]
CPT[2] = {'10' : 0.02,
		'11' : 0.3,
		'12' : 0.8,
		'00' : 0.98,
		'01' : 0.7,
		'02' : 0.2}
		
CPT_variable[3] = [3]
CPT[3] = {'1' : 0.1,
		'0' : 0.9}

CPT_variable[4] = [4]
CPT[4] = {'0' : 0.5,
		'1' : 0.25,
		'2' : 0.25}

# Calculate Pr(data)
def CalProb(data):
	prob = 1
	for i in range(5):
		tmp = ''
		for j in CPT_variable[i]:
			tmp += str(data[j])
		prob *= CPT[i][tmp]
	return prob

# Calculate sum of weight in weight_table
def CalWeightSum(feature_value, weight_table):
	weight_sum = 0.0
	for key0 in weight_table:
		flag = True
		for key1 in feature_value:
			if key0[key1] != feature_value[key1]:
				flag = False
				break
		if flag == True:
			weight_sum += weight_table[key0]
	return weight_sum

# test the performance
def test():
	rightCnt = 0
	for data in testData:
		prob = [0.0 for i in range(3)]
		for i in range(3):
			prob[i] = CalProb(data[:4] + [i])
		if np.argmax(np.array(prob)) == data[4]:
			rightCnt += 1
	print('accuracy = ', rightCnt / len(testData))

test()

# perform EM algorithm
maxIter = 30
data_likelihood = [0.0 for i in range(maxIter)]
for iterCnt in range(maxIter):
	# E step
	weight_table = {}
	for data in trainData:
		if data[4] != -1: 
			# Dunetts Syndrome is known
			data_str = str(data[0]) + str(data[1]) + str(data[2]) + str(data[3]) + str(data[4])
			if data_str in weight_table:
				weight_table[data_str] += 1.0
			else:
				weight_table[data_str] = 1.0
			data_likelihood[iterCnt] += 1.0
		else:
			# Dunetts Syndrome is unknown
			prob = [0.0 for i in range(3)]
			data_str = ['' for i in range(3)]
			prob_sum = 0
			for i in range(3):
				prob[i] = CalProb(data[:4] + [i])
				data_str[i] = str(data[0]) + str(data[1]) + str(data[2]) + str(data[3]) + str(i)
				prob_sum += prob[i]				
			for i in range(3):
				prob[i] /= prob_sum
				if data_str[i] in weight_table:
					weight_table[data_str[i]] += prob[i]
				else:
					weight_table[data_str[i]] = prob[i]
			data_likelihood[iterCnt] += prob_sum
	
	# M step
	for i in range(5):
		for key in CPT[i]:
			feature_value = {}
			for j in range(len(CPT_variable[i])):
				feature_value[CPT_variable[i][j]] = key[j]
			CPT[i][key] = CalWeightSum(feature_value, weight_table)
			del feature_value[i]
			CPT[i][key] /= CalWeightSum(feature_value, weight_table)

test()

plt.figure()
plt.plot(range(maxIter), data_likelihood)
plt.show()
