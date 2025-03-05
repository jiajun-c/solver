import numpy as np
import matplotlib.pyplot as plt

N = 64
L = 1
dx = L/N

old = np.zeros(N+1)
new = np.zeros(N+1)
tmp = np.array([np.sin(np.pi*i*dx)/2 + np.sin(16*np.pi*i*dx)/2 for i in range(1,N)])
resi = [0]
resi[0] = max(abs(tmp))

for i in range(0, 10000):
    for j in range(1, N):
        new[j] = (old[j+1] + new[j-1] - dx**2*tmp[j-1])/2
    new[0] = new[N] = 0
    r = tmp - (new[0:N-1] - 2*new[1:N] + new[2:N+1])/dx**2
    resi.append(max(abs(r)))
    if (max(abs(r)) < 0.00001):
        print("converge at {} iter".format(i))
        break
    old = new