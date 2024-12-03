import numpy as np
import matplotlib.pyplot as plt

# Série de Fourrier
def X(k):
    if k == 0:
        return 0
    return 2/(k*np.pi**2)  * (1j*np.pi/(2) * np.cos(k*np.pi/2) - 1j/k * np.sin(k*np.pi/2))

# reconstrói a função no tempo baseado na série de fourrier
def x(t):
    f = 0
    for k in range(k_inicial, k_inicial+numDeSenoides):

        f = f +  X(k)*np.exp(W0*1j*t*k)
        
    return f

if __name__ == "__main__":
    k_inicial = -20
    numDeSenoides = 40

    # Frequência natural
    W0 = 2

    start = -2*3.1415
    end = 2*3.1415
    numOfSteps = (end - start) * 150

    linspace = np.arange(start, end, 1/numOfSteps)

    plt.plot(linspace, x(linspace))
    plt.show()