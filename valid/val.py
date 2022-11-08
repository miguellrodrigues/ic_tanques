import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


plt.style.use([
    'notebook',
    'grid',
    {'figure.dpi': 150}
])

# dados simulacao

r = .31
mu = .4
sigma = .55
a4 = .3019

coeff = np.array([165.48, 16.46, 29.4, -147.01, -156.93, -83.93])

a_1 = coeff[0]
a_2 = coeff[1]
a_3 = coeff[2]

b_1 = coeff[3]
b_2 = coeff[4]
b_3 = coeff[5]


def nl_area(h3, h4):
    _a2 = np.cos(2.5*np.pi * (h4 - mu)) / (sigma * np.sqrt(2 * np.pi))
    _a3 = np.exp(-((h4 - mu)**2) / (2 * sigma**2))

    a3 = ((3*r)/5) * (2.7*r - (_a2 * _a3))

    return a3, [_a2, _a3]


def non_linear_space_state(_, s, a3, u, z):
    h3, h4 = s

    if h3 < 0:
        h3 = 0

    if h4 < 0:
        h4 = 0

    beta = (b_3 - b_1) / (a3 * h3)
    zeta = (b_2 - b_3) / (a4 * h4)

    A = np.array([
        [-(a_3/a3) - ((a_1*np.sqrt(h3))/(a3*h3)) + beta, (a_3/a3)],
        [(a_3/a4), -(a_3/a4) + zeta]
    ])

    B = np.array([
        [0],
        [a_2/a4]
    ])

    h = np.array([
        [h3 - z/a3],
        [h4]
    ])

    return 10e-5 * (A@h + B*u).flatten()


sim_time = 12500
sim_step = 1

iterations = int(sim_time / sim_step)

time = np.arange(1e-12, sim_time, sim_step)

h3_zero = 1
h4_zero = 1

h = np.empty((iterations, 2))
h[0] = [h3_zero, h4_zero]

z = 0

u = np.ones(iterations)

u[0:2500] = 42
u[2500:5000] = 37
u[5000:7500] = 28
u[7500:10000] = 55
u[10000:12500] = 47

a3, _ = nl_area(h3_zero, h4_zero)

for i in range(1, iterations):
    t = time[i-1]

    h3, h4 = h[i-1]

    sol = solve_ivp(
        non_linear_space_state,
        t_span=(t, t + sim_step),
        y0=h[i-1],
        t_eval=(t, t + sim_step),
        args=(a3, u[i], z),
    )

    h[i] = sol.y[:, -1]

    _a3, _ = nl_area(h[i, 0], h[i, 1])

    delta_h3 = h[i, 0] - h[i-1, 0]
    delta_a3 = a3 - _a3

    z += (delta_a3/sim_step) * delta_h3

    a3 = _a3

print(z)

np.save('h3_z.npy', h[:, 0])
np.save('h4_z.npy', h[:, 1])

plt.plot(time, h[:, 0], label='h3')
plt.plot(time, h[:, 1], label='h4')
plt.legend()
plt.show()
