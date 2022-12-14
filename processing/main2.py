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
    'sin_1.5',
]

color = ['b-', 'r-', 'g-']

cut_off_freq = (2*np.pi)*1e-1 # rad/s
sampling_period = .5 # seconds

b, a = sig.butter(1, cut_off_freq, 'low', analog=True)
cut_idx = 400
i=0

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

def do_a_lot_of_things(l3, l4, ppc, exp_name, t):
    # filter the data
    Level3CM_filtered = sig.lfilter(b, a, l3)
    Level4CM_filtered = sig.lfilter(b, a, l4)

    Level3CM_filtered = Level3CM_filtered[cut_idx:]
    Level4CM_filtered = Level4CM_filtered[cut_idx:]

    # # # # # # # # # # #

    r = .31
    mu = .40
    sigma = .55

    z1_bounds = np.array([-.5160, 2.3060])
    z2_bounds = np.array([.0033, .0077])
    z3_bounds = np.array([3.4478, 18.4570])

    def calculate_pertinence_functions_values(h3, h4,h3r,h4r,exp_name,t):
        if exp_name == 'error_0':
            h3 = h3r
            h4 = h4r

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

        def sampled(W):
            W[W<1/3] = 1/6
            W[(W>1/3) & (W<2/3)] = 3/6
            W[W>2/3] = 5/6
            return W

        if exp_name == 'filter_plus_sin':
            M1 = 0.5*M1 + np.sin(0.5*t)*0.5*0.75
            M1 = np.clip(M1, 0, 1)
            N1 = 0.5*N1 + np.sin(0.5*t)*0.5*0.75
            N1 = np.clip(N1, 0, 1)
            O1 = 0.5*O1 + np.sin(0.5*t)*0.5*0.75
            O1 = np.clip(O1, 0, 1)

        elif exp_name == 'filter_plus_mf_sampled':
            M1 = sampled(M1)
            N1 = sampled(N1)
            O1 = sampled(O1)

        M2 = 1-M1
        N2 = 1-N1
        O2 = 1-O1

        return M1, N1, O1, M2, N2, O2

    # # # # # # # # # # #

    M1_filtered, N1_filtered, O1_filtered, _, _, _ = calculate_pertinence_functions_values(Level3CM_filtered, Level4CM_filtered, Level3CM[cut_idx:], Level4CM[cut_idx:], exp_name,t)

    return Level3CM_filtered, Level4CM_filtered, M1_filtered, N1_filtered, O1_filtered

_, axs_h = plt.subplots(3, 1, sharex=True)
_, axs_MF = plt.subplots(3, 1)

for exp_name in exp_names:
    mat = sio.loadmat(f'./data/{exp_name}.mat')

    Level3CM = np.array(mat['Level3CM'][0])
    Level4CM = np.array(mat['Level4CM'][0])
    Pum2PC   = np.array(mat['Pump2PC'] [0])
    t = np.arange(0,(len(Level3CM)/2), 0.5)

    t  =  t[:-cut_idx]
    Pum2PC = Pum2PC[cut_idx:]

    Level3CM_filtered, Level4CM_filtered, M1_filtered, N1_filtered, O1_filtered = do_a_lot_of_things(Level3CM, Level4CM, Pum2PC, exp_name, t)

    axs_h[0].plot(t,Level3CM_filtered, 'r', label='Level3CM')
    axs_h[1].plot(t,Level4CM_filtered, 'g', label='Level4CM')
    axs_h[2].plot(t,Pum2PC, 'b', label='PumpPC')

    axs_h[0].legend()
    axs_h[1].legend()
    axs_h[2].legend()

    axs_MF[0].plot(t,M1_filtered, 'k-', label='M1')
    axs_MF[1].plot(t,N1_filtered, 'k-', label='N1')
    axs_MF[2].plot(t,O1_filtered, 'k-', label='O1')

    axs_MF[0].legend()
    axs_MF[1].legend()
    axs_MF[2].legend()

    i=i+1

plt.show()
