import numpy as np
import matplotlib.pyplot as plt

def u(x):
    return np.heaviside(x, 1)

def h(t):
    return np.exp(-2*(t+1)) * u(t+1)

def x(t):
    return u(t) + (1-t)*u(t-1) + (t-2)*u(t-2)

def respAnalitica(t):
    i2 = ( np.exp(-2*(t+1))/2* (np.exp(2*(t+1)) -1))    * u(t+1) * u(-t)

    i3 = (np.exp(-2*(t+1)) * (1/2*(np.e**2-1) - 3/4*np.e**2) + 3/4 - t/2) * u(t) * u(-t+1)

    i4 = (np.exp(-2*(t+1))* (1/2*(np.e**2 - 1) + (np.e**4/4-3/4*np.e**2))) * u(t-1) * u(-t+3)
    return i2  + i3 + i4

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
