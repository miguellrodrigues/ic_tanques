import matplotlib.pyplot as plt
import numpy as np

plt.style.use([
    'grid',
    'science',
    'notebook',
])

Ts = ['0.1', '4', '8']

h3_ = []
h4_ = []

fig, axs = plt.subplots(2, 1, figsize=(15, 5))
iterations = 10001

for i in range(len(Ts)):
    t = Ts[i]

    time = np.arange(0, float(t)*iterations, float(t))

    h3 = np.load(f'./data/h3_{t}_py.npy')
    h4 = np.load(f'./data/h4_{t}_py.npy')

    axs[0].step(time[:100], h3[:100], label=f'$T={t}$')
    axs[1].step(time[:100], h4[:100], label=f'$T={t}$')


axs[0].set_ylabel('h3')
axs[1].set_ylabel('h4')

axs[1].set_xlabel('time')

axs[0].legend()
axs[1].legend()

plt.show()
