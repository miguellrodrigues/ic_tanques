from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(3, suppress=True)

plt.style.use([
    'notebook',
    'high-vis',
    'grid',
])


h3_matlab = np.array(loadmat('./mats/h3_matlab.mat')['h3'])
h4_matlab = np.array(loadmat('./mats/h4_matlab.mat')['h4'])


h3_py     = np.load('./discrete/data/h3_py.npy')
h4_py     = np.load('./discrete/data/h4_py.npy')


plt.figure()
plt.plot(h3_matlab, '-', label='matlab')
plt.plot(h3_py, label='python')
plt.plot(h4_matlab, '-', label='matlab')
plt.plot(h4_py, label='python')
plt.legend()
plt.show()
