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

diffs = []
y = []

for i in u:
    t, levels = load_data(f'./q34_data/VAZAO_COMUNICANTE_{i}', keys=['t', 'Level3CM', 'Level4CM'])

    h1 = levels[0][-1]
    h2 = levels[1][-1]

    diff = (h1 - h2)

    diffs.append(diff)
    y.append(q_in(i))

# # # # # # # # # # # # # # # # # # # #

diffs = np.array(diffs)

x =  np.arange(
    np.min(diffs),
    np.max(diffs) + 1,
    .1
)

y = np.array(y)

thetas = np.polyfit(diffs, y, 1)
y_hat = np.polyval(thetas, x)

print(thetas)

plt.plot(diffs, y, 'o', label='q34 measured')
plt.plot(x, y_hat, label='q34 approximated')
plt.xlabel('h1 - h2')
plt.ylabel('q34')
plt.legend()
plt.show()
