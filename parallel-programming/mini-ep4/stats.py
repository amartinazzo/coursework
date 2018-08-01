import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

bakery = pd.read_csv('bakery.csv')
gate = pd.read_csv('gate.csv')

fig1 = plt.figure()

ax = fig1.add_subplot(211, projection='3d')
ax.set_title('tempo de execução (ns)')
ax.set_xlabel('num_threads')
ax.set_ylabel('total_time')
ax.scatter(bakery['num_threads'], bakery['total_time'], bakery['running_time'], c='b', label='bakery')
ax.scatter(gate['num_threads'], gate['total_time'], gate['running_time'], c='g', label='gate')
ax.legend()

ax2 = fig1.add_subplot(212, projection='3d')
ax2.set_title('desvio padrão dos acessos das threads')
ax2.set_xlabel('num_threads')
ax2.set_ylabel('total_time')
ax2.scatter(bakery['num_threads'], bakery['total_time'], bakery['stddev_accesses'], c='b', label='bakery')
ax2.scatter(gate['num_threads'], gate['total_time'], gate['stddev_accesses'], c='g', label='gate')

plt.show()