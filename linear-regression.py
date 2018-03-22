import matplotlib.pyplot as plt
import numpy as np

N = 100
runs = 1

def linregression():
	X = np.ones((N, 3))
	X[:,1] = np.random.uniform(-1, 1, N) # x1 = x
	X[:,2] = np.random.uniform(-1, 1, N) # x2 = y

	p0 = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
	p1 = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))

	a = (p1[1] - p0[1])/(p1[0] - p0[0])
	b = p0[1] - a*p0[0]

	# Y = np.sign(X[:,2] - (a*X[:,1] + b))
	Y = (a*X[:,1] + b)

	X_pseudo_inv = np.linalg.pinv(X)

	w = np.dot(X_pseudo_inv, Y)

	y_hat = np.dot(X, w)

	print Y
	print y_hat

	E_in = np.sum((y_hat - Y)**2)/N

	return E_in

n=0
E_in = 0
while(n < runs):
	E_in += linregression()
	n += 1

E_in = E_in/runs
print "in sample error: " + str(E_in)