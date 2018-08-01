import matplotlib.pyplot as plt
import numpy as np

n_coins = 1000
n_flips = 10
runs = 100000

plot = True

# heads = 1
# tails = 0

def toss_coins():
	min_heads = 10
	rand = np.random.randint(n_coins)

	tosses = np.random.randint(2, size=(n_coins, n_flips, runs))
	coin_1 = tosses[0,:,:]
	coin_rand = tosses[rand,:,:]

	sums = np.sum(tosses, axis=1)
	min_index = np.argmin(sums, axis=0)
	coin_min = np.zeros((n_flips, runs))

	# TODO avoid this for loop

	for i in range(runs):
		coin_min[:, i] = tosses[min_index[i],:,i]

	nu_1 = np.mean(coin_1, axis=0)
	nu_rand = np.mean(coin_rand, axis=0)
	nu_min = np.mean(coin_min, axis=0)

	return (nu_1, nu_rand, nu_min)


(nu_1, nu_rand, nu_min) = toss_coins()

nu_1_mean = np.mean(nu_1)
nu_rand_mean = np.mean(nu_rand)
nu_min_mean = np.mean(nu_min)

print "nu 1 mean: " + str(nu_1_mean)
print "nu rand mean: " + str(nu_rand_mean)
print "nu min mean: " + str(nu_min_mean)

if plot:
	x = np.linspace(0, runs, num=runs)
	fig, axes = plt.subplots(2, 3)
	axes[0,0].plot(x, nu_1)
	axes[0,0].set_title(r'$\nu_1 distribution$')
	axes[1,0].hist(nu_1)
	axes[1,0].set_title(r'$\nu_1 histogram$')

	axes[0,1].plot(x, nu_rand)
	axes[0,1].set_title(r'$\nu_{rand} distribution$')
	axes[1,1].hist(nu_rand)
	axes[1,1].set_title(r'$\nu_{rand} histogram$')

	axes[0,2].plot(x, nu_min)
	axes[0,2].set_title(r'$\nu_{min} distribution$')
	axes[1,2].hist(nu_min)
	axes[1,2].set_title(r'$\nu_{min} histogram$')
	plt.show()
