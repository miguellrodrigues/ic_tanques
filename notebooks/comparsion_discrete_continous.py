from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(3, suppress=True)

plt.style.use([
    'notebook',
    'high-vis',
    'grid',
])

h3c_py     = np.load('./continous/data/h3_py.npy')
h4c_py     = np.load('./continous/data/h4_py.npy')

h3d_py     = np.load('./discrete/data/h3_py.npy')
h4d_py     = np.load('./discrete/data/h4_py.npy')


plt.figure()
plt.plot(h3c_py, '-', label='h3 contínuo')
plt.plot(h3d_py, '-', label='h3 discreto')
plt.plot(h4c_py, '-', label='h4 contínuo')
plt.plot(h4d_py, '-', label='h4 discreto')
plt.legend(loc='best')
plt.show()
