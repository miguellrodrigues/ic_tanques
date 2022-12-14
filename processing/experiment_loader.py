from scipy.io import loadmat
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt


plt.style.use([
    'notebook',
    'grid',
])


experiment_name = 'sin_.5'
data = loadmat(f'./data/{experiment_name}.mat')

h3 = np.array(data['Level3CM'][0])
h4 = np.array(data['Level4CM'][0])
u  = np.array(data['Pump2PC' ][0])
t  = np.array(data['t' ][0])

# # # # #

b = np.array([0.13575525, 0.13575525])
a = np.array([1.,        -0.7284895])

filtered_h3 = sig.lfilter(b, a, h3)
filtered_h4 = sig.lfilter(b, a, h4)
filtered_u = sig.lfilter (b,  a, u)

plt.figure()
plt.plot(h3, 'k')
plt.plot(h4, 'k')
plt.plot(filtered_h3)
plt.plot(filtered_h4)

plt.figure()
plt.plot(u)

plt.show()

cut_idx = input('Cut index: ')
cut_idx = int(cut_idx)

h3 = h3[cut_idx:]
h4 = h4[cut_idx:]
u  =  u[cut_idx:]

filtered_h3 = filtered_h3[cut_idx:]
filtered_h4 = filtered_h4[cut_idx:]
filtered_u  = filtered_u[cut_idx:]

np.save(f'./data/{experiment_name}_h3.npy', filtered_h3)
np.save(f'./data/{experiment_name}_h4.npy', filtered_h4)
np.save(f'./data/{experiment_name}_u.npy',  u)
np.save(f'./data/{experiment_name}_t.npy', t)

plt.figure()
plt.plot(filtered_h3, linewidth=2, label='filtered h3')
plt.plot(filtered_h4, linewidth=2, label='filtered h4')
plt.savefig(f'./data/{experiment_name}_h3_h4.png')

plt.legend()

plt.figure()
plt.plot(u, linewidth=2, label='u')
plt.legend()
plt.savefig(f'./data/{experiment_name}_u.png')

plt.show()
