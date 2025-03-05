import numpy as np
import matplotlib.pyplot as plt

N = 64
L = 1
h = L/N
phi = np.zeros(N+1)
f = np.array([np.sin(np.pi*i*h)/2 + np.sin(16*np.pi*i*h)/2 for i in range(0,N+1)])
# 使用G-S迭代进行平滑处理
def smoothing(phi,f,h)->np.array:
    N = len(phi)-1
    res = np.zeros(N+1)
    for j in range(1,N):
        res[j] = (phi[j+1]+res[j-1]-h**2*f[j])/2
    return res

# 计算残差
def residual(phi,f,h)->np.array:
    N = len(phi)-1
    res = np.zeros(N+1)
    res[1:N] = f[1:N]-(phi[0:N-1]-2*phi[1:N]+phi[2:N+1])/h**2
    return res

# 将细网格转换为粗网格
def restriction(r) -> np.array:
    coarseLen = int((len(r) - 1)/2)
    res = np.zeros(coarseLen+1)
    for i in range(1, coarseLen):
        res[i] = (r[2*i- 1] + 2*r[2*i] + r[2*i + 1])/4
    return res

# def restriction(r)->np.array:
#     N = int((len(r)-1)/2)
#     res = np.zeros(N+1)
#     for j in range(2,N+1):
#         res[j-1] = (r[2*j-3]+2*r[2*j-2]+r[2*j-1])/4
#     return res

# 将粗网格转换为细网格，对于偶数的点，直接使用粗网格的点信息，对于奇数的点，则是使用
def prolongation(eps)->np.array:
    N = (len(eps)-1)*2
    res = np.zeros(N+1)
    for j in range(2,N+1,2):
        res[j-1] = (eps[int(j/2-1)]+eps[int(j/2)])/2
    for j in range(1,N+2,2):
        res[j-1] = eps[int((j+1)/2-1)]
    return res

def V_Cycle(phi,f,h):
    phi = smoothing(phi,f,h)
    r = residual(phi,f,h)
    rhs = restriction(r)
    eps = np.zeros(len(rhs))
    if len(eps)-1 == 2:
        eps = smoothing(eps,rhs,2*h)
    else:
        eps = V_Cycle(eps,rhs,2*h)
    phi = phi + prolongation(eps)
    phi = smoothing(phi,f,h)
    return phi

resi = []
for cnt in range(0,1001):
    phi = V_Cycle(phi,f,h)
    r = residual(phi,f,h)

    r_disp = max(abs(r))
    resi.append(r_disp)
    print("cnt: {}  r_disp: {}\n".format(cnt,r_disp))
    if max(abs(r)) < 0.0000001:
        print("converge at {} iterations".format(cnt*10))
        break