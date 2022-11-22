import numpy as np
import control as ct
import matplotlib.pyplot as plt

A_matrices = [np.load(f'../data/vertices/A{i}.npy') for i in range(8)]
B_matrices = [np.load(f'../data/vertices/B{i}.npy') for i in range(8)]

C = np.eye(4)
D = np.zeros((4, 1))

ts = .01

for i in range(len(A_matrices)):
    Ai = A_matrices[i]
    Bi = B_matrices[i]

    sys = ct.StateSpace(Ai, Bi, C, D)

    sampled_sys = ct.c2d(sys, ts, 'zoh')

    print(np.abs(np.linalg.eigvals(sampled_sys.A)))

    np.save(f'../data/vertices/sampled_A{i}.npy', sampled_sys.A)
    np.save(f'../data/vertices/sampled_B{i}.npy', sampled_sys.B)
