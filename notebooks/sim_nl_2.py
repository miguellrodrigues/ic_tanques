import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sig
from sklearn.metrics import median_absolute_error

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


def R_34(diff):
    return 0.000006*diff**3 -0.000266*diff**2 + 0.003971*diff + 0.005568

def q_out(h4):
    return 9.785984*h4 + 156.704985


def simulate(a, b, c):
    Ts = 4
    Tf = 10000
    samples = int(Tf/Ts)

    t = np.linspace(1e-12, Tf, samples)

    h3_t = np.zeros(samples)
    h4_t = np.zeros(samples)

    h3_zero = 1e-3
    h4_zero = 1e-3

    h3_t[0] = h3_zero
    h4_t[0] = h4_zero

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

    for i in range(1, len(t)):
        h3 = h3_t[i - 1]
        h4 = h4_t[i - 1]

        H = np.array([
            [h3],
            [h4]
        ])

        diff = h3 - h4

        R34 = a*R_34(diff)
        qout = b*q_out(h4)

        a3 = (3 * r / 5) * (2.7 * r - ((np.cos(2.5*np.pi*h3 - mu)) / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((h3 - mu)**2) / (2 * sigma ** 2)))

        z1 = 1/R34
        z2 = qout/h4
        z3 = 1/a3

        A = np.array([
            [-z1*z3, z1*z3],
            [z1/a4, (-z1-z2)/a4]
        ])

        B = np.array([
            [c*z3],
            [.0]
        ])

        h_dot = A@H + B*u[i]

        h3_t[i] = h3 + h_dot[0, 0]*Ts
        h4_t[i] = h4 + h_dot[1, 0]*Ts

    return h3_t, h4_t


def find_optimal_parameters(y_ode, a_0, b_0, c_0):
    P = np.array([a_0, b_0, c_0])
    dP = np.array([.001, .001, .001], dtype=np.float64)
    best_err = 100

    # Diferenciao de alteracao de 'dP' no caso de falha
    k_si = .01

    iterations = 0
    max_iter = 200

    while best_err > .1:
        if iterations > max_iter:
            break

        for i in range(len(P)):
            P[i] += dP[i]

            # obtem erro
            h3_t, _ = simulate(P[0], P[1], P[2])
            err = median_absolute_error(y_ode, h3_t)

            print(P, err, iterations)

            if err < best_err:
                best_err = err

                dP[i] *= 1 + k_si
            else:
                P[i] -= 2 * dP[i]

                h3_t, _ = simulate(P[0], P[1], P[2])
                err = median_absolute_error(h3_t, y_ode)

                if err < best_err:
                    best_err = err
                    dP[i] *= 1 + k_si
                else:
                    P[i] += dP[i]
                    dP[i] *= 1 - k_si

        iterations += 1

    return P


# P = find_optimal_parameters(h3_exp, 0.84617798, 1.09323753, 12.89976026)

h3_t, h4_t = simulate(0.84617798, 1.09323753, 12.89976026)

plt.figure()
plt.plot(h3_t, label='h3')
plt.plot(h4_t, label='h4')
plt.plot(h3_exp, label='h3_exp')
plt.plot(h4_exp, label='h4_exp')
plt.legend()
plt.show()
