import scipy.io as sio
import scipy.signal as sig
import control as ct
import numpy as np
import matplotlib.pyplot as plt

plt.style.use([
    'science',
    'notebook',
    'grid',
])

exp_names = [
    'error_0',
    'filter_plus_mf_sampled',
    'filter_plus_sin',
    'only_filter'
]

cut_off_freq = (2*np.pi)*1e-1 # rad/s
sampling_period = .5 # seconds

b, a = sig.butter(1, cut_off_freq, 'low', analog=True)

G = ct.tf(b, a)

# ct.bode(G, dB=True, Hz=True, omega_limits=(1e-2, 1e2), omega_num=1000)
# plt.show()

Z = ct.c2d(G, sampling_period, 'tustin')

num_z, den_z = ct.tfdata(Z)

b = num_z[0][0]
a = den_z[0][0]

# print transfer function fo the filter
print('b = ', b)
print('a = ', a)

'''
plt.figure()
w, h = sig.freqs(b, a)
plt.semilogx(w, 20 * np.log10(abs(h)))
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.savefig('butterworth_filter_frequency_response.png', dpi=300)
'''

def do_a_lot_of_things(l3, l4, ppc, exp_name):
    # filter the data
    Level3CM_filtered = sig.lfilter(b, a, l3)
    Level4CM_filtered = sig.lfilter(b, a, l4)

    _, axs = plt.subplots(2, 1, sharex=True)

    axs[0].plot(Level3CM_filtered, 'b-', label='Level3CM filtered')
    axs[0].plot(Level4CM_filtered, 'r-', label='Level4CM filtered')

    axs[1].plot(ppc, 'k-', label='PumpPC')
    plt.legend()
    plt.savefig(f'filtered_data_{exp_name}.png', dpi=300)

    # # # # # # # # # # #

    r = .31
    mu = .40
    sigma = .55

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
    _, axs = plt.subplots(3, 1)

    axs[0].plot(M1_filtered, label='M1')
    axs[0].set_title('M1')
    axs[0].set_ylim(.15, .22)

    axs[1].plot(N1_filtered, label='N1')
    axs[1].set_title('N1')
    axs[1].set_ylim(.044, .385)

    axs[2].plot(O1_filtered, label='O1')
    axs[2].set_title('O1')
    # axs[2].set_ylim(.3, 1)

    plt.savefig(f'./pertinence_functions_{exp_name}.png', dpi=300)
    plt.show()


for exp_name in exp_names:
    mat = sio.loadmat(f'./data/{exp_name}.mat')

    Level3CM = np.array(mat['Level3CM'][0])
    Level4CM = np.array(mat['Level4CM'][0])
    Pum2PC   = np.array(mat['Pump2PC'] [0])

    do_a_lot_of_things(Level3CM, Level4CM, Pum2PC, exp_name)
