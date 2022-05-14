import numpy as np
import matplotlib.pyplot as plt

conv_3 = np.load("qtable_convergence.npy")
conv_4 = np.load("rewards_list_DD.npy")

# plt.plot(conv_3, label="Convergence of Q-learning")
plt.plot(conv_4, label="Convergence of Double Deep Q-learning")
plt.title("Comparison of the convergence of Q-learning and Double Deep Q-learning", fontsize = 15)
plt.xlabel("Timesteps", fontsize=13)
plt.ylabel("Average reward", fontsize=13)
plt.legend()
plt.show()