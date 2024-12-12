import numpy as np
import matplotlib.pyplot as plt

def u(x):
    return np.heaviside(x, 1)

def h(n):
    return ( 5.75*0.7**(n-5) - 5.25*0.9**(n-5) )*u(n-5)

def x(n):
    return 2*np.cos(np.pi/2*n)

def respAnalitica(n):
    caso1 = 1.9256*np.cos(np.pi/2*n)
    caso2 = 0#  ( (-6.4)*(0.8)**n * (1-1.25**(n+1))  ) * u(n)*u(-n+4)
    caso3 = 0#  (13.13*0.8**n) * u(n-5) #*u(-n+5)
    caso4 = 0#  0 #1     * u(n-6)
    return caso1 + caso2 + caso3 + caso4


start = -15
end = 16

n = np.arange(start, end, 1)
# print((n))# print(h(n)

y = np.convolve(x(n), h(n), mode='same')

print(respAnalitica(n))

# plt.scatter(n, x(n), label = "x(n)", s=5)
# plt.scatter(n, h(n), label = "h(n)", s=4)
plt.scatter(n, y, label = "convolução x(n)*h(n)", s=18)
plt.scatter(n, respAnalitica(n), label = "Resposta analítica", s=10)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.xlabel("n")
plt.ylabel("x[n]*h[n]")
plt.show()
