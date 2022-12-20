import cvxpy as cvx
import matplotlib.pyplot as plt
import numpy as np

plt.style.use(['science', 'notebook', 'grid'])

np.set_printoptions(precision=3, suppress=True)

A_matrices = [np.load(f'./data/A_{i}.npy') for i in range(8)]
B_matrices = [np.load(f'./data/B_{i}.npy') for i in range(8)]

n_matrices = len(A_matrices)

n = A_matrices[0].shape[0]
m = B_matrices[0].shape[1]

# #

q = 1
r = .8

# #

W = cvx.Variable((n, n), symmetric=True)
Z = [cvx.Variable((m, n)) for _ in range(n_matrices)]

epsilon = 1e-6
constraints = [W >> cvx.Constant(np.eye(n) * epsilon), ]

for i in range(n_matrices):
  for j in range(i, n_matrices):
    alpha = .5 if i == j else 1

    Ai = A_matrices[i]
    Aj = A_matrices[j]

    Bi = B_matrices[i]
    Bj = B_matrices[j]

    Zi = Z[i]
    Zj = Z[j]

    LMI = cvx.bmat([[-r * W, q * W + alpha * ((Ai + Aj) @ W + (Bi @ Zj + Bj @ Zi))],
                    [q * W + alpha * (W @ (Ai + Aj).T + (Zj.T @ Bi.T + Zi.T @ Bj.T)), -r * W]])

    constraints.append(LMI << -cvx.Constant(np.eye(2 * n) * epsilon))

prob = cvx.Problem(cvx.Minimize(0), constraints)
prob.solve(verbose=False, solver='MOSEK')

print(prob.status)

if prob.status != 'optimal':
  print('Not optimal')
  exit(0)

theta = np.linspace(0, 2 * np.pi, 100)
x = r * np.cos(theta) - q
y = r * np.sin(theta)

plt.plot(x, y)

K = []

W_ = np.zeros((n, n)) + W.value
P = np.linalg.inv(W_)

print(' ')
for i in range(n_matrices):
  Ki = Z[i].value @ P

  print(Ki)
  K.append(Ki)

  Acl = A_matrices[i] + B_matrices[i] @ Ki

  eig_vals = np.linalg.eigvals(Acl)
  plt.scatter(eig_vals.real, eig_vals.imag, marker='x', color='C0')

plt.show()
np.save('./data/K.npy', K)
