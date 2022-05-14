import numpy as np
import matplotlib.pyplot as plt

rewards_prob3 = np.load("avg_reward_prob3.npy")
rew = np.load("avg_rew_per_req.npy")
rewards_prob4 = np.load("avg_reward_prob4.npy")
np.save("avg_reward_prob4.npy", rewards_prob4)
optimum = np.load("optimum.npy")

plt.plot(rewards_prob3, label="Average rewards of Q-learning")
plt.plot(rewards_prob4, label="Average rewards of Double Deep Q-learning")
plt.plot(optimum[:,3], label="Rewards for the optimum allocation")
plt.title("Comparison of the optimal allocation of classes to the Q-learning and Deep Double Q-learning")
plt.xlabel("Allocated requests",fontsize=13)
plt.ylabel("Average/optimum reward",fontsize=13)
plt.legend()
plt.show()