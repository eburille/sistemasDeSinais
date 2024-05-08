import numpy as np
import matplotlib.pyplot as plt

def u(x):
    return np.heaviside(x, 1)

def h(n):
    return u(n+10) - 2*u(n) + u(n-4)

def x(n):
    return u(n-2)

def respAnalitica(x):
    return u(x+8)*(9+x) - u(x-1)*(9+x) + u(x-1)*(11-x) - u(x-6)*(11-x) + u(x-6)*6

start = -15
end = 16

space = np.arange(start, end, 1)
print((space))
print(h(space))

y = np.convolve(x(space), h(space), mode='same')

print((y))

plt.scatter(space, x(space), label = "x(n)")
plt.scatter(space, h(space), label = "h(n)")
plt.scatter(space, y, label = "convolução x(n)*h(n)")
plt.scatter(space, respAnalitica(space), label = "Resposta analítica")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
