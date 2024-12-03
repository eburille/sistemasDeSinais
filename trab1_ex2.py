import numpy as np
import matplotlib.pyplot as plt

def u(x):
    return np.heaviside(x, 1)

def h(t):
    return 2*np.exp(-3*(t-1)) * u(t-1) 

def x(t):
    return u(t) + u(t-1) - 2*u(t-2) 

def respAnalitica(t):
    i2 = ( 2/3*(1-np.exp(-3*(t-1)))  )    * u(t-1) * u(-t+2)

    i3 = ( 2/3*np.exp(-3*(t-1)) * (2*np.exp(3*(t-1)) - np.exp(3) - 1 )  )    * u(t-2) * u(-t+3)

    i4 = (  2/3*np.exp(-3*(t-1)) * (2*np.exp(6) - np.exp(3) -1 ) ) * u(t-3) #* u(-t+4)
    return i2  + i3 + i4

start = -5
end = 5
passo = 0.001

t = np.arange(start, end, passo)
print((t))
print(h(t))

y = np.convolve(x(t), h(t), mode='same')
y = y * 0.001 # Normalização
print((y))

# plt.plot(t, x(t), label = "x(n)")
# plt.plot(t, h(t), label = "h(n)")
plt.plot(t, y, label = "convolução x(n)*h(n)")
plt.plot(t, respAnalitica(t), label = "Resposta analítica")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.xlabel("t")
plt.ylabel("x(t)*h(t)")
plt.show()

