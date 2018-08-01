import matplotlib.pyplot as plt
import numpy as np

N = 20 # 100
runs = 1 # 1000
plot = True

def perceptron():
	X = np.ones((N, 3))
	X[:,1] = np.random.uniform(-1, 1, N) # x1 = x
	X[:,2] = np.random.uniform(-1, 1, N) # x2 = y

	# print X

	p0 = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
	p1 = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))

	a = (p1[1] - p0[1])/(p1[0] - p0[0])
	b = p0[1] - a*p0[0]

	Y = np.sign(X[:,2] - (a*X[:,1] + b))

	w = np.zeros(3)

	halt = False
	w_iterations = 0

	while(not halt):
		count = 0
		for i in range(N):
			if np.sign(np.dot(w, X[i,:])) != Y[i]:
				w = w + Y[i] * X[i,:]
				w_iterations += 1
				
				if plot:
					y_hat = np.dot(X, w)
					print y_hat
					plt.plot(X[:,1], y_hat, color=(0,0,0, w_iterations*0.05))

			else:
				count += 1
		if count == N:
			halt = True

	plt.show()

	g = np.sign(np.dot(X, w))

	# print "w final"
	# print w

	print "Y"
	print Y

	print "g(x)"
	print g

	print "w iterations: " + str(w_iterations)

	# test
	M = 10*N
	X_test = np.ones((M, 3))
	X_test[:,1] = np.random.uniform(-1, 1, M) # x1 = x
	X_test[:,2] = np.random.uniform(-1, 1, M) # x2 = y

	g_test = np.sign(np.dot(X_test, w))
	f_test = np.sign(X_test[:,2] - (a*X_test[:,1] + b))

	p_error = np.sum(np.abs(g_test-f_test)/2)/M

	return (w_iterations, p_error)

n = 0
w_iterations = 0
p_error = 0
while(n < runs):
	(w, p) = perceptron()
	w_iterations += w
	p_error += p
	n += 1
mean_w_iteration = w_iterations/runs
mean_p_error = p_error/runs

print "mean iterations: " + str(mean_w_iteration)
print "mean P[f != g]: " + str(mean_p_error)
