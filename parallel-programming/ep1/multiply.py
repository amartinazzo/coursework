import numpy as np

D = np.loadtxt('DD.txt')
E = np.loadtxt('EE.txt')

C = np.dot(D, E)

print(C)