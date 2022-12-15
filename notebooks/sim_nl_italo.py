import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sig

# #

b = np.array([
    0.13575525, 0.13575525
])

a = np.array([
    1.,        -0.7284895
])

h3_exp = np.load('./experiments/h1_exp.npy')[:-1]
h4_exp = np.load('./experiments/h2_exp.npy')[:-1]

h3_exp = sig.lfilter(b, a, h3_exp)
h4_exp = sig.lfilter(b, a, h4_exp)

# #


plt.close('all')

Ts = 4
Tf = 10000
samples = int(Tf/Ts)

t = np.linspace(1e-12, Tf, samples)

h3 = np.zeros(samples)
h4 = np.zeros(samples)

h3_zero = 1e-6
h4_zero = 1e-6

h3[0] = h3_zero
h4[0] = h4_zero

r = 31
mu = 4
sigma = 55
a4 = np.pi*r**2

u = np.empty(len(t))

degs = np.array([
    25, 35, 30, 40
])

frac = int(samples/4)

u[:frac] = degs[0]
u[frac:frac*2] = degs[1]
u[frac*2:frac*3] = degs[2]
u[frac*3:] = degs[3]

def q_in(u):
    return .215374*u**2 - .411661*u + 180.133588


def q_34(diff):
    return (33.082144*diff + 89.99671)


def q_out(h4):
    return 86.518236*np.sqrt(h4) - 11.642853


for i in range(1, len(t)):
    H = np.array([
        [h3[i - 1]],
        [h4[i - 1]]
    ])

    diff = h3[i - 1] - h4[i - 1]

    qin = q_in(u[i - 1])
    q34 = q_34(diff)
    qout = q_out(h4[i - 1])

    a1 = (3 * r / 5) * (2.7 * r - ((np.cos(2.5*np.pi*h3[i - 1] - mu)) / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((h3[i - 1] - mu)**2) / (2 * sigma ** 2)))

    h3_dot = (qin - q34)/a1
    h4_dot = (q34 - qout)/a4

    # termino calculo area T3
    h3[i] = h3[i - 1] + h3_dot*Ts
    h4[i] = h4[i - 1] + h4_dot*Ts


plt.style.use([
    'notebook',
    'grid',
])

plt.figure()
plt.plot(t, h3, label='h3 sim')
plt.plot(t, h3_exp, label='h3 exp')

plt.plot(t, h4, label='h4 sim')
plt.plot(t, h4_exp, label='h4 exp')

plt.legend()
plt.show()
