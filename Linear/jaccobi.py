import numpy as np
import torch

def Jacobi(A, x, b, n):
    times = 0
    D = np.diag(np.diag(A))
    L = -np.array(np.tril(A, -1))
    U = -np.array(np.triu(A, 1))
    D_inv = np.linalg.inv(D)
    while times < n:
        x = D_inv.dot( b + L.dot(x) + U.dot(x) )
        times += 1
    return x

A = np.array([[2, -1, 0], [-1, 3, -1], [0, -1, 2]], dtype=np.float32)
B = np.array([1, 8, 5],dtype=np.float32)
x = np.zeros(3)
A_tensor = torch.from_numpy(A)
b_tensor = torch.from_numpy(B)
print(Jacobi(A, x, B, 3000))
print(torch.linalg.solve(A_tensor, b_tensor))