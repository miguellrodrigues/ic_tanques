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

plt.figure()
w, h = sig.freqs(b, a)
plt.semilogx(w, 20 * np.log10(abs(h)))
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency

# filter the data
Level3CM_filtered = sig.lfilter(b, a, Level3CM)
Level4CM_filtered = sig.lfilter(b, a, Level4CM)

plt.figure()
plt.plot(Level3CM)
plt.plot(Level4CM)

plt.plot(Level3CM_filtered)
plt.plot(Level4CM_filtered)

# # # # # # # # # # #

r = .31
mu = .40
sigma = .55
A2 = .3019

z1_bounds = np.array([-.5160, 2.3060])
z2_bounds = np.array([.0033, .0077])
z3_bounds = np.array([3.4478, 18.4570])

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

M1_real = M1_real[100:]
N1_real = N1_real[100:]
O1_real = O1_real[100:]

M1_filtered = M1_filtered[100:]
N1_filtered = N1_filtered[100:]
O1_filtered = O1_filtered[100:]

# create 3 subplots each one for a pertinence function
fig, axs = plt.subplots(3, 1)
fig.suptitle('Pertinence functions')

axs[0].plot(M1_real, label='M1')
axs[0].plot(M1_filtered, label='M1 filtered')
axs[0].set_title('M1')
axs[0].set_ylim(0, 1)

axs[1].plot(N1_real, label='N1')
axs[1].plot(N1_filtered, label='N1 filtered')
axs[1].set_title('N1')
axs[1].set_ylim(0, 1)

axs[2].plot(O1_real, label='O1')
axs[2].plot(O1_filtered, label='O1 filtered')
axs[2].set_title('O1')
axs[2].set_ylim(0, 1)

max_m1_diff = np.max(np.abs(M1_real - M1_filtered))
max_n1_diff = np.max(np.abs(N1_real - N1_filtered))
max_o1_diff = np.max(np.abs(O1_real - O1_filtered))

print(' ')
print('max_m1_diff = ', max_m1_diff)
print('max_n1_diff = ', max_n1_diff)
print('max_o1_diff = ', max_o1_diff)
print(' ')

plt.show()
