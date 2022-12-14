import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')

Ts = .5
t = np.arange(0,5000,Ts)

h3 = np.empty(len(t))
h3.fill(np.nan)
h4 = np.empty(len(t))
h4.fill(np.nan)

r = 31
mu = 4
sigma = 55
a4 = 3019


h3[0]=0
h4[0]=1e-3

u = np.empty(len(t))
u.fill(35)
Kb = 16.99

R12_t = np.zeros(len(t))
q0_t = np.zeros(len(t))
a1_t = np.zeros(len(t))

for i in range(0,len(t)-1,1):
    H = np.array([
        [h3[i]],
        [h4[i]]
    ])

    diff = h3[i] - h4[i]

    R12 = (.412*diff+11.488)*1e-3
    q0  = (12.241*h4[i]+868.674)
    a1  = (3*r/5)*(2.7*r-((np.cos(2.5*np.pi*(h3[i]-8)-mu))/(sigma*np.sqrt(2*np.pi)))*np.exp(-(((h3[i]-8)-mu)**2)/(2*sigma**2)))

    R12_t[i] = R12
    q0_t[i] = q0
    a1_t[i] = a1

    z1 = 1/R12
    z2 = q0/h4[i]
    z3 = 1/a1

    A = np.array([
        [-z1*z3, z1*z3],
        [z1/a4, (-z1-z2)/a4]
    ])

    B = np.array([
        [Kb*z3],
        [.0]
    ])

    h_dot = A@H + B*u[i]

    #termino calculo area T3
    h3[i+1] = h3[i] + h_dot[0]*Ts
    h4[i+1] = h4[i] + h_dot[1]*Ts


plt.style.use([
    'notebook',
    'grid',
])

plt.figure()
plt.plot(t,h3, label='h3')
plt.plot(t,h4, label='h4')
plt.legend()

R12_t[-1] = R12_t[-2]
q0_t[-1] = q0_t[-2]
a1_t[-1] = a1_t[-2]

plt.figure()
plt.plot(h3-h4, R12_t, label='R12')
plt.legend()


plt.figure()
plt.plot(h4, q0_t, label='q0')
plt.legend()


plt.figure()
plt.plot(h3, a1_t, label='a1')
plt.legend()


plt.legend()
plt.show()
