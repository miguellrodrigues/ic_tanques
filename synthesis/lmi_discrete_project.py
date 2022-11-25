import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(3, suppress=True)

plt.style.use([
    'notebook',
    'high-vis',
    'grid',
])

A_matrices = [np.load(f'./data/A_{i}.npy') for i in range(8)]
B_matrices = [np.load(f'./data/B_{i}.npy') for i in range(8)]

def find_controller(A, B):
  n = A[0].shape[0]
  m = B[0].shape[1]

  n_matrices = len(A)

  W = cvx.Variable((n, n), symmetric=True)
  S = cvx.Variable((m, m), diag=True)
  Z = cvx.Variable((m, n))

  L = [cvx.Variable((m, n)) for _ in range(n_matrices)]

  constraints = [W >> np.eye(n) * 1e-6]

  rho = 30

  def LMI(i, j):
    alpha = .5 if i == j else 1

    Ai = A[i]
    Aj = A[j]

    Bi = B[i]
    Bj = B[j]

    Li = L[i]
    Lj = L[j]

    # # LMI 1
    LMI1_11 = -W
    LMI1_12 = -Z.T
    LMI1_13 = alpha * ( (Li.T@Bj.T + Lj.T@Bi.T) + (W.T @ (Ai.T + Aj.T)) )

    LMI1_21 = -Z
    LMI1_22 = -2 * S
    LMI1_23 =  S @ (alpha * (Bi.T + Bj.T))

    LMI1_31 = alpha * ( (Ai + Aj) @ W + (Bi@Lj + Bj@Li) )
    LMI1_32 = alpha * (Bi + Bj) @ S
    LMI1_33 = -W

    LMI1_1 = cvx.hstack([LMI1_11, LMI1_12, LMI1_13])
    LMI1_2 = cvx.hstack([LMI1_21, LMI1_22, LMI1_23])
    LMI1_3 = cvx.hstack([LMI1_31, LMI1_32, LMI1_33])

    LMI1 = cvx.vstack([LMI1_1, LMI1_2, LMI1_3])

    # # END LMI 1

    # # LMI 2

    LMI2_11 = W
    LMI2_12 = alpha * (Li.T + Lj.T) - Z.T
    LMI2_21 = alpha * (Li + Lj) - Z
    LMI2_22 = np.array([
      [rho ** 2]
    ])

    LMI2_1 = cvx.hstack([LMI2_11, LMI2_12])
    LMI2_2 = cvx.hstack([LMI2_21, LMI2_22])

    LMI2 = cvx.vstack([LMI2_1, LMI2_2])

    return LMI1, LMI2

  for i in range(n_matrices):
    for j in range(i, n_matrices):
      LMI1, LMI2 = LMI(i, j)

      constraints.append(LMI1 << -np.eye(LMI1.shape[0]) * 1e-6)
      constraints.append(LMI2 >> np.eye(LMI2.shape[0]) * 1e-6)

  prob = cvx.Problem(
    cvx.Minimize(0),
    constraints
  )

  prob.solve(verbose=False, solver='MOSEK')
  print('status: ', prob.status)
  print(' ')

  if prob.status == 'infeasible':
    return None, None

  K = []

  W_arr = np.array(W.value)
  P_arr = np.linalg.inv(W_arr)

  for i in range(n_matrices):
    Li = np.array(L[i].value)
    Ki = Li@P_arr

    K.append(Ki)

  return K, P_arr

if __name__ == '__main__':
  K, P = find_controller(A_matrices, B_matrices)

  if K == None:
    print('No controller found')
    exit()

  print(np.linalg.eigvals(P))
  print(' ')

  import matplotlib.pyplot as plt

  theta_step = .01

  theta = np.arange(0, 2*np.pi, theta_step)

  circle = np.array([np.cos(theta), np.sin(theta)])
  output = np.linalg.inv(np.linalg.cholesky(P)) @ circle

  plt.plot(output[0], output[1])
  plt.show()
