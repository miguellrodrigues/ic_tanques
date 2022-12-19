import numpy as np
import matplotlib.pyplot as plt
from experiment_loader import load_data
from scipy.interpolate import lagrange


np.set_printoptions(precision=6, suppress=True)

plt.style.use([
    'grid',
    'science',
    'notebook',
])

# #

qin = np.array([0.100771  , 4.435961, 139.190782])
def q_in(u):
    return np.polyval(qin, u)

# #

r = 31
mu = 40
sigma = 55

u = range(10, 50, 5)

h = []
y = []

for i in u:
    t, levels = load_data(f'./q34_data/VAZAO_COMUNICANTE_{i}', keys=['t', 'Level3CM', 'Level4CM'])

    h1 = levels[0][-1]
    h2 = levels[1][-1]

    qout = q_in(i)

    h.append(h2)
    y.append(qout)

# # # # # # # # # # # # # # # # # # # #

h = np.array(h)

x =  np.arange(
    np.min(h) - .1,
    np.max(h) + .1,
    .1
)

y = np.array(y)

thetas = np.polyfit(h, y, 2)
y_hat = np.polyval(thetas, x)

print(thetas)

plt.plot(h, y, 'o', label='qout measured')
plt.plot(x, y_hat, label='qout approximated')
plt.xlabel('h2')
plt.ylabel('Rout')
plt.legend()
plt.show()
