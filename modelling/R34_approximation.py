import numpy as np
import matplotlib.pyplot as plt
from experiment_loader import load_data
from scipy.optimize import curve_fit


np.set_printoptions(precision=6, suppress=True)

plt.style.use([
    'grid',
    'science',
    'notebook',
])

# #

def q_in(u):
    return 147.465291*np.exp(.030642*u)

# #

r = 31
mu = 40
sigma = 55

u = range(10, 35, 5)

diffs = []
y = []

for i in u:
    t, levels = load_data(f'./q34_data/VAZAO_COMUNICANTE_{i}', keys=['t', 'Level3CM', 'Level4CM'])

    h1 = levels[0][-1]
    h2 = levels[1][-1]

    diff = h1 - h2

    R34 = diff / q_in(i)

    diffs.append(diff)
    y.append(R34)

# # # # # # # # # # # # # # # # # # # #

diffs = np.array(diffs)

x =  np.arange(
    np.min(diffs),
    np.max(diffs) + .1,
    .1
)

y = np.array(y)

thetas = np.polyfit(diffs, y, 3)
y_hat = np.polyval(thetas, x)

print(thetas)

plt.plot(diffs, y, 'o', label='R34 measured')
plt.plot(x, y_hat, label='R34 approximated')
plt.xlabel('h1 - h2')
plt.ylabel('R34')
plt.legend()
plt.show()
