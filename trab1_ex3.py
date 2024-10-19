import numpy as np
import matplotlib.pyplot as plt

def u(x):
    return np.heaviside(x, 1)

def x(n):
    return u(n)     

def y(y1, y2, n):
    return 2*x(n) + x(n-1) + 1/4*y2

def yAnalitico(n):
    return -2*(1/2)**n + 0 *(-1/2)**n + 4

y1 = 0
y2 = 0

start = 0
end = 10
linspace = np.arange(start, end, 1) 
print(linspace)
y0 = np.array([0]*(end-start), dtype=np.float64)

for n in linspace:
    yn = y(y1, y2, n)
    y0[n] = yn

    y2 = y1
    y1 = yn

print(y0[0])

plt.scatter(linspace, y0, s = 9, label = "Resposta numérica")
plt.scatter(linspace, yAnalitico(linspace), s = 5, label = "Resposta analítica")
# plt.scatter(linspace, x(linspace))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()