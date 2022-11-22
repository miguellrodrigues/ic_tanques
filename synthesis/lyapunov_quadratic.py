import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(3, suppress=True)

plt.style.use([
    'notebook',
    'high-vis',
    'grid',
])

A = [np.load(f'./data/A_{i}.npy') for i in range(8)]
B = [np.load(f'./data/B_{i}.npy') for i in range(8)]

n = A[0].shape[0]
nu = B[0].shape[1]

n_matrices = len(A)

W = cvx.Variable((n, n), symmetric=True)
Q = cvx.Variable((n, n), symmetric=True)
Z = [cvx.Variable((nu, n)) for _ in range(n_matrices)]

constraints = [
  W >> np.eye(n)*1e-6
]

for i in range(n_matrices):
  for j in range(i, n_matrices):
    alpha = .5 if i == j else 1

    Ai = A[i]
    Aj = A[j]

    Bi = B[i]
    Bj = B[j]

    Zi = Z[i]
    Zj = Z[j]

    LMI = W@(Ai+Aj).T + (Zj.T@Bi.T + Zi.T@Bj.T) + (Ai+Aj)@W + Bi@Zj + Bj@Zi

    constraints.append(LMI << -np.eye(n)*1e-6)

prob = cvx.Problem(cvx.Minimize(0), constraints)
prob.solve(verbose=False, solver='MOSEK')

print(prob.status)

import matplotlib.pyplot as plt

P = np.linalg.inv(W.value)

print(' ')
print(P)
print(' ')
print(np.linalg.eigvals(P))
print(' ')

for i in range(n_matrices):
  Ki = Z[i].value @ P

  print(f'K{i} = ', Ki)
  print(' ')

theta_step = .01

theta = np.arange(0, 2*np.pi, theta_step)

circle = np.array([np.cos(theta), np.sin(theta)])
output = np.linalg.inv(np.linalg.cholesky(P)) @ circle

plt.plot(output[0,:], output[1,:])
plt.show()
