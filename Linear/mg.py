import numpy as np

def gauss_seidel(A, x, b, iterations=1):
    """
    高斯-赛德尔迭代法
    :param A: 系数矩阵
    :param x: 当前解向量
    :param b: 右侧向量
    :param iterations: 迭代次数
    :return: 更新后的解向量
    """
    N = len(b)
    for _ in range(iterations):
        for i in range(N):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i+1:], x[i+1:])
            x[i] = (b[i] - sum1 - sum2) / A[i, i]
    return x

def restriction(residual, coarse_size):
    """
    限制操作：将细网格上的残差限制到粗网格上
    :param residual: 细网格上的残差
    :param coarse_size: 粗网格的大小
    :return: 粗网格上的残差
    """
    coarse_residual = np.zeros(coarse_size)
    for i in range(coarse_size):
        coarse_residual[i] = residual[2 * i]
    return coarse_residual

def interpolation(correction, coarse_size, fine_size):
    """
    插值操作：将粗网格上的校正插值到细网格上
    :param correction: 粗网格上的校正
    :param coarse_size: 粗网格的大小
    :param fine_size: 细网格的大小
    :return: 细网格上的校正
    """
    fine_correction = np.zeros(fine_size)
    for i in range(coarse_size):
        fine_correction[2 * i] = correction[i]
        fine_correction[2 * i + 1] = correction[i]
    return fine_correction

def two_grid_solver(A, b, x0, coarse_size, max_iter=10, tol=1e-6):
    """
    两网格求解器
    :param A: 系数矩阵
    :param b: 右侧向量
    :param x0: 初始解向量
    :param coarse_size: 粗网格的大小
    :param max_iter: 最大迭代次数
    :param tol: 收敛容差
    :return: 求解后的解向量
    """
    N = len(b)
    x = x0.copy()
    for iter in range(max_iter):
        # Step 1: Presmoothing
        x = gauss_seidel(A, x, b, iterations=3)
        
        # Step 2: Compute residual
        residual = b - np.dot(A, x)
        
        # Step 3: Restriction to coarse grid
        coarse_residual = restriction(residual, coarse_size)
        
        # Step 4: Solve on coarse grid
        Ac = A[::2, ::2]  # Coarse grid matrix (simplified)
        ec = np.linalg.solve(Ac, coarse_residual)
        
        # Step 5: Interpolation to fine grid
        fine_correction = interpolation(ec, coarse_size, N)
        
        # Step 6: Update solution
        x += fine_correction
        
        # Step 7: Postsmoothing (optional)
        x = gauss_seidel(A, x, b, iterations=3)
        
        # Check convergence
        if np.linalg.norm(residual, ord=np.inf) < tol:
            print(f"Converged after {iter + 1} iterations.")
            break
    
    return x
import torch
# 示例
if __name__ == "__main__":
    # 定义系数矩阵 A 和右侧向量 b
    A = np.array([[4, -1, 0, 0],
                  [-1, 4, -1, 0],
                  [0, -1, 4, -1],
                  [0, 0, -1, 3]], dtype=float)
    b = np.array([15, 8, 10, 10], dtype=float)
    A_tensor = torch.from_numpy(A)
    b_tensor = torch.from_numpy(b)
    print(torch.linalg.solve(A_tensor, b_tensor))
    # 初始解向量
    x0 = np.zeros_like(b)
    
    # 粗网格大小
    coarse_size = 2
    
    # 调用两网格求解器
    solution = two_grid_solver(A, b, x0, coarse_size)
    
    print("Solution:", solution)