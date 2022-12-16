import numpy as np
import matplotlib.pyplot as plt
from experiment_loader import load_data
from scipy.optimize import curve_fit
from scipy.interpolate import *


np.set_printoptions(precision=6, suppress=True)

plt.style.use([
    'grid',
    'science',
    'notebook',
])

# #

def q_in(u):
    return .215374*u**2 - .411661*u + 180.133588

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

    diff = (h1 - h2)

    R34 = diff / q_in(i)

    diffs.append(diff)
    y.append(R34)


diffs = np.array(diffs)
y = np.array(y)

x =  np.arange(
    np.min(diffs),
    np.max(diffs) + 1,
    .1
)


thetas = np.polyfit(diffs, y, 3)
print(thetas)

y_hat = np.polyval(thetas, x)

plt.plot(diffs, y, 'o', label='R34 measured')
plt.plot(x, y_hat, label='R34 approximated')

plt.show()
