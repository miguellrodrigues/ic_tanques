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
    return 147.46531*np.exp(-.030642*u)

# #

r = 31
mu = 40
sigma = 55

u = range(10, 35, 5)

x = []
y = []

for i in u:
    t, levels = load_data(f'./q34_data/VAZAO_COMUNICANTE_{i}', keys=['t', 'Level3CM', 'Level4CM'])

    h1 = levels[0][-1]
    h2 = levels[1][-1]

    diff = h1 - h2

    R34 = diff / q_in(i)

    x.append(diff)
    y.append(R34)

# # # # # # # # # # # # # # # # # # # #

def func(x, a, b):
    return a * np.exp(-b * x)

x = np.array(x)
y = np.array(y)

popt, pcov = curve_fit(func, x, y)
y_hat = func(x, *popt)

print(popt)

plt.plot(x, y_hat, label='R34 approximated')
plt.plot(x, y, 'o', label='R34 measured')
plt.xlabel('h1 - h2')
plt.ylabel('R34')
plt.legend()
plt.show()
