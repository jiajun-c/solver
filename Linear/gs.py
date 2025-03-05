import numpy as np
import torch

def gauss_seidel(A, b, tol=1e-6, max_iter=100):
    times = 0 
    x = np.zeros(A.shape[0])
    D, L, U = np.diag(np.diag(A)), -np.array(np.tril(A, -1)), -np.array(np.triu(A, 1))
    D_L_inv = np.linalg.inv((D-L))
    while times < max_iter:
        x = D_L_inv.dot((b + U.dot(x)))
        times += 1
    return x

# 示例
A = np.array([[4, -1, 0, 0],
              [-1, 4, -1, 0],
              [0, -1, 4, -1],
              [0, 0, -1, 3]], dtype=float)
b = np.array([15, 10, 10, 10], dtype=float)
A_tensor = torch.from_numpy(A)
B_tensor = torch.from_numpy(b)

golden = torch.linalg.solve(A_tensor, B_tensor)
print(golden)
solution = gauss_seidel(A, b, 10000)
print("Solution:", solution)