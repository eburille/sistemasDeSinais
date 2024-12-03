import numpy as np
import matplotlib.pyplot as plt

def u(x):
    return np.heaviside(x, 1)

def x(n):
    return u(n) 

def y(y1, y2, n):
    return 1.6*y1 - 0.63*y2 + 0.5*x(n-5) - 1.5*x(n-6)

def yAnalitico(n):
    return ( -13.42*0.7**(n-5) +  47.25*0.9**(n-5)  -  33.33)*u(n-5)

y1 = 0
y2 = 0

start = -6
end = 60
n = np.arange(start, end, 1) 
y0 = np.array([0]*(end-start), dtype=np.float64)

for index, n_ in enumerate(n):
    yn = y(y1, y2, n_)
    y0[index] = yn
    
    print(yn, n_)
    y2 = y1
    y1 = yn

plt.scatter(n, y0, s = 18, label = "Resposta numérica")
plt.scatter(n, yAnalitico(n), s = 5, label = "Resposta analítica")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.xlabel("n")
plt.ylabel("y[n]")
plt.show()