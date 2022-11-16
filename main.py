import scipy.io as sio
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt


mat = sio.loadmat('./data/data.mat')

Level3CM = np.array(mat['Level3CM'][0])
Level4CM = np.array(mat['Level4CM'][0])
Pum2PC   = np.array(mat['Pump2PC'][0])


cut_off_freq = 0.01 # rad/s
sampling_period = .5 # seconds

# first order filter with 10 Hz cut-off frequency
b, a = sig.butter(1, cut_off_freq/2*np.pi, 'low', analog=False, fs=1/sampling_period)

# print transfer function fo the filter
print('b = ', b)
print('a = ', a)

# filter the data
Level3CM_filtered = sig.lfilter(b, a, Level3CM)
Level4CM_filtered = sig.lfilter(b, a, Level4CM)

# # # # # # # # # # #

r = .31
mu = .40
sigma = .55
A2 = .3019

z1_bounds = np.array([124.6572, 216.3098])
z2_bounds = np.array([118.4579, 338.5220])
z3_bounds = np.array([.0004, .0048])

def calculate_pertinence_functions_values(h3, h4):
    diff = h3 - h4

    R34 = (0.2371*diff+6.9192)*10
    q0  = (18.6367*h4+700.6831)*1e-4
    a1 = (3*r/5)*(2.7*r-((np.cos(2.5*np.pi*(h3-8)*1e-2-mu))/(sigma*np.sqrt(2*np.pi)))*np.exp(-(((h3-8)*1e-2-mu)**2)/(2*sigma**2)))

    z1 = 1/R34
    z2 = q0/h4
    z3 = 1/a1

    M1 = (z1 - z1_bounds[0])/(z1_bounds[1] - z1_bounds[0])
    N1 = (z2 - z2_bounds[0])/(z2_bounds[1] - z2_bounds[0])
    O1 = (z3 - z3_bounds[0])/(z3_bounds[1] - z3_bounds[0])

    M2 = 1-M1
    N2 = 1-N1
    O2 = 1-O1

    return M1, N1, O1, M2, N2, O2

# # # # # # # # # # #

M1_real, N1_real, O1_real, _, _, _ = calculate_pertinence_functions_values(Level3CM, Level4CM)
M1_filtered, N1_filtered, O1_filtered, _, _, _ = calculate_pertinence_functions_values(Level3CM_filtered, Level4CM_filtered)

# create 3 subplots each one for a pertinence function
# fig, axs = plt.subplots(3, 1)
# fig.suptitle('Pertinence functions')

# axs[0].plot(M1_real, label='M1')
# axs[0].plot(M1_filtered, label='M1 filtered')
# axs[0].set_title('M1')

# axs[1].plot(N1_real, label='N1')
# axs[1].plot(N1_filtered, label='N1 filtered')
# axs[1].set_title('N1')

# axs[2].plot(O1_real, label='O1')
# axs[2].plot(O1_filtered, label='O1 filtered')
# axs[2].set_title('O1')

plt.plot(N1_real)
plt.plot(N1_filtered)
plt.show()

# plot the data
# plt.plot(Level3CM, label='Level3CM')
# plt.plot(Level4CM, label='Level4CM')

# plt.plot(Level3CM_filtered, 'k', label='Level3Filt')
# plt.plot(Level4CM_filtered, 'k', label='Level4Filt')

# plt.legend()
# plt.show()
