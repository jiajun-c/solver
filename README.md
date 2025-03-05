# solver

## 1. Jaccobi 迭代

Jaccobi迭代中使用上次迭代的x值去更新本次的x值, 其公式可以写为

$Dx^{k+1} = (L + U)x^k + b$

将两边同时乘以D的逆，可以得到本次x的值

```python3
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
```

## 2. G-S迭代

G-S迭代有所不同，本次迭代的左下侧将会参与到迭代中

$Dx^{k+1} = Lx^{k+1} + Ux^k + b$

$(D-L)x^{k+1} = Ux^k + b$

两边同时乘以D-L的逆可以得到 x 本次迭代的结果

```python3
def gauss_seidel(A, b, tol=1e-6, max_iter=100):
    times = 0 
    x = np.zeros(A.shape[0])
    D, L, U = np.diag(np.diag(A)), -np.array(np.tril(A, -1)), -np.array(np.triu(A, 1))
    D_L_inv = np.linalg.inv((D-L))
    while times < max_iter:
        x = D_L_inv.dot((b + U.dot(x)))
        times += 1
    return x
```

## 3. SOR 迭代

在GS迭代的基础上我们可以实现SOR迭代

$Dx^{k+1} = Lx^{k+1} + U x^k + b
$

$x^{k+1} = D^{-1}(Lx^{k+1} + Ux^k + b)$

改写为如下的形式，第二项作为的纠正项，

$x^{k+1} = x^k + D^{-1}(Lx^{k+1} + Ux^k - Dx^k + b)$

给第二项增加一个修正项 w来获得更快的收敛速度，简称松弛因子，SOR的迭代格式为

$x^{k+1} = x^k + wD^{-1}(Lx^{k+1} + Ux^k - Dx^k + b)$

当w等于1的时候将被转换为GS方法

