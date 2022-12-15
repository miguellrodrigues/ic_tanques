import numpy as np
import matplotlib.pyplot as plt
from experiment_loader import load_data


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

u = range(10, 50, 5)

z = []
y = []

for i in u:
    t, levels = load_data(f'./q34_data/VAZAO_COMUNICANTE_{i}', keys=['t', 'Level3CM', 'Level4CM'])

    h1 = levels[0][-1]
    h2 = levels[1][-1]

    Rout = (h2 / q_in(i))

    z.append(h2)
    y.append(Rout)

# # # # # # # # # # # # # # # # # # # #

z = np.array(z)

x =  np.arange(
    np.min(z),
    np.max(z) + 1,
    .1
)

y = np.array(y)

thetas = np.polyfit(z, y, 2)
y_hat = np.polyval(thetas, x)

print(thetas)

plt.plot(z, y, 'o', label='Rout measured')
plt.plot(x, y_hat, label='Rout approximated')
plt.xlabel('h2')
plt.ylabel('Rout')
plt.legend()
plt.show()
