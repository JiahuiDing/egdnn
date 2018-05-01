import numpy as np
import matplotlib.pyplot as plt

N = np.array(range(1, 16))
neuronNum = np.array([1, 6, 7, 11, 15, 17, 8, 24, 44, 40, 35, 60, 60, 71, 130])
connectionNum = np.array([5, 56, 68, 143, 203, 219, 69, 403, 1061, 930, 660, 1619, 1885, 2525, 6956])

print(np.corrcoef(N, neuronNum)) # 0.87925449
plt.figure()
plt.xlabel('N')
plt.ylabel('number of neurons')
plt.plot(N, neuronNum)
plt.show()

print(np.corrcoef(N, connectionNum)) # 0.73411522
plt.figure()
plt.xlabel('N')
plt.ylabel('number of connections')
plt.plot(N, connectionNum)
plt.show()


'''
1	1	5
2	6	56
3	7	68
4	11	143
5	15	203
6	17	219
7	8	69
8	24	403
9	44	1061

10	40	930
10	48	1186

11	35	660
11	21	348

12	60	1619
12	71	2259

13	104	5399
13	60	1885

14	128	9058
14	71	2525

15	29	477
15	37	809
15	130	6956
'''
