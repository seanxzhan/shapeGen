import numpy as np
import matplotlib.pyplot as plt

# gen_loss = np.load('./gen_loss.npy')
# gen_loss_1 = np.load('./gen_loss_1.npy')

# disc_loss = np.load('./disc_loss.npy')
# disc_loss_1 = np.load('./disc_loss_1.npy')

# gen_L = np.concatenate((gen_loss, gen_loss_1))
# disc_L = np.concatenate((disc_loss, disc_loss_1))

np.load('./gen_L.npy', gen_L)
np.load('./disc_L.npy', disc_L)

plt.plot(gen_L, label = "G Loss")
plt.plot(disc_L, label = "D Loss")
plt.xlabel('Epoch')
plt.ylim([-2, 5])
plt.legend()
plt.show()