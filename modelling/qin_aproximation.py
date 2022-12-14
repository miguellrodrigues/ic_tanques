import numpy as np
import matplotlib.pyplot as plt
from experimente_loader import load_data


np.set_printoptions(precision=6, suppress=True)

plt.style.use([
    'notebook',
    'grid'
])

t, h = load_data('./qin_data/VAZAO_ENTRADA_30')
h = h[0]

r = .31
mu = .40
sigma = .55

area = (3 * r / 5) * (2.7 * r - ((np.cos(2.5*np.pi*h - mu)) / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((h - mu)**2) / (2 * sigma ** 2)))
volume = (np.pi*r**2 - area)*h

flow = np.diff(volume) / np.diff(t)

# # # # # # # # # # # # # # # # # # # #
_v = volume[:-1]

thetas = np.polyfit(volume[:-1], flow, 1)
print(thetas)

_t = t[:-1]

y = np.polyval(thetas, _v)

plt.plot(_t, y, label='flow approximated')
plt.plot(_t, flow, 'o', label='flow measured')
plt.legend()
plt.show()
