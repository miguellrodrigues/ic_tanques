import numpy as np
import matplotlib.pyplot as plt
from experiment_loader import load_data
from scipy.interpolate import *


np.set_printoptions(precision=9, suppress=True)

plt.style.use([
    'grid',
    'science',
    'notebook',
])

# #

def q_34(diff):
    return -0.164211*diff**2+31.428672*diff+107.711073

# #

r = 31
mu = 40
sigma = 55

u = np.array(range(10, 50, 5))

diffs = []
y = []

for i in u:
    t, levels = load_data(f'./q34_data/VAZAO_COMUNICANTE_{i}', keys=['t', 'Level3CM', 'Level4CM'])

    h1 = levels[0][-1]
    h2 = levels[1][-1]

    diff = h1 - h2

    R34 = diff / q_34(diff)

    diffs.append(diff)
    y.append(R34)


diffs = np.array(diffs)
y = np.array(y)

x =  np.arange(
    np.min(diffs) - .1,
    np.max(diffs) + .1,
    .1
)

print(np.min(diffs))
print(np.max(diffs))

thetas = np.polyfit(diffs, y, 2)
print(thetas)

y_hat = np.polyval(thetas, x)

plt.plot(diffs, y, 'o', label='R34 measured')
plt.plot(x, y_hat, label='R34 approximated')

plt.show()
