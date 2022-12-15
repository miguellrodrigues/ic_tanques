import numpy as np
import matplotlib.pyplot as plt
from experiment_loader import load_data


np.set_printoptions(precision=6, suppress=True)

plt.style.use([
    'grid',
    'science',
    'notebook',
])

r = 31
mu = 40
sigma = 55

u = range(10, 70, 5)
y = []

for i in u:
    t, levels = load_data(f'./qin_data/VAZAO_ENTRADA_{i}')
    h = levels[0]

    area = (3 * r / 5) * (2.7 * r - ((np.cos(2.5*np.pi*h - mu)) / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((h - mu)**2) / (2 * sigma ** 2)))
    volume = (np.pi*r**2 - area)*h

    flow = np.gradient(volume, t)
    medium_flow = np.mean(flow)

    y.append(medium_flow)

# # # # # # # # # # # # # # # # # # # #

u = np.array(u)
y = np.array(y)

x =  np.arange(
    np.min(u),
    np.max(u) + .1,
    .1
)

thetas = np.polyfit(u, y, 2)
print(thetas)

y_hat = np.polyval(thetas, x)

plt.plot(u, y, 'o', label='flow measured')
plt.plot(x, y_hat, label='flow approximated')
plt.xlabel('Pump2PC')
plt.ylabel('Flow rate cm³/s')
plt.legend()
plt.show()
