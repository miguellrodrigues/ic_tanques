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

def simulate(a, b, c):
    Ts = 4
    Tf = 10000
    samples = int(Tf/Ts)

    t = np.linspace(1e-12, Tf, samples)

    h3_t = np.zeros(samples)
    h4_t = np.zeros(samples)

    h3_zero = 1e-6
    h4_zero = 1e-6

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

    def q_in(u):
        return .215374*u**2 - .411661*u + 180.133588


    def q_34(diff):
        return 33.082144*diff + 89.99671


    def q_out(h4):
        return 9.785984*h4 + 156.704985


    for i in range(1, len(t)):
        h3 = h3_t[i - 1]
        h4 = h4_t[i - 1]

        H = np.array([
            [h3],
            [h4]
        ])

        diff = h3 - h4

        qin = a*q_in(u[i])
        q34 = b*q_34(diff)
        qout = c*q_out(h4)

        a3 = (3 * r / 5) * (2.7 * r - ((np.cos(2.5*np.pi*h3 - mu)) / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((h3 - mu)**2) / (2 * sigma ** 2)))

        # termino calculo area T3
        h3_t[i] = h3 + ( (qin - q34 )/a3 )*Ts
        h4_t[i] = h4 + ( (q34 - qout)/a4 )*Ts

    return h3_t, h4_t


from sklearn.metrics import median_absolute_error


def find_optimal_parameters(y_ode, a_0, b_0, c_0):
    P = np.array([a_0, b_0, c_0])
    dP = np.array([.001, .001, .001], dtype=np.float64)
    best_err = 100

    # Diferenciao de alteracao de 'dP' no caso de falha
    k_si = .01

    iterations = 0
    max_iter = 100

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


P = find_optimal_parameters(h3_exp, 1.07792213, 1.1388081, 1.12487676)

h3_t, h4_t = simulate(*P)

plt.figure()
plt.plot(h3_t, label='h3')
plt.plot(h4_t, label='h4')
plt.plot(h3_exp, label='h3_exp')
plt.plot(h4_exp, label='h4_exp')
plt.legend()
plt.show()
