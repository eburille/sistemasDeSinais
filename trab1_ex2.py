import numpy as np
import matplotlib.pyplot as plt

def u(x):
    return np.heaviside(x, 1)

def h(t):
    return -2*u(t-1) + 2*u(t-2)

def x(t):
    return u(t) - u(t-1)

def respAnalitica(x):
    return u(x-1)*(2-2*x) - u(x-2)*(2-2*x) + u(x-2)*(2*x-6) - u(x-3)*(2*x-6)

start = -9
end = 9
passo = 0.01

space = np.arange(start, end, passo)
print((space))
print(h(space))

y = np.convolve(x(space), h(space), mode='same')
y = y * 0.01 # Normalização
print((y))

plt.plot(space, x(space), label = "x(n)")
plt.plot(space, h(space), label = "h(n)")
plt.plot(space, y, label = "convolução x(n)*h(n)")
plt.plot(space, respAnalitica(space), label = "Resposta analítica")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
