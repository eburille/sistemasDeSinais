import numpy as np
import matplotlib.pyplot as plt

# Série de Fourrier
def X(k):
    return 89.8*(1/(np.pi*(1-2*k)) + 1/(np.pi*(1+2*k)))

# reconstrói a função no tempo baseado na série de fourrier
def x(t):
    f = 0
    for k in range(k_inicial, k_inicial+numDeSenoides):

        f = f +  X(k)*np.exp(120j*np.pi*t*k)
        
    return f


if __name__ == "__main__":
    k_inicial = -20
    numDeSenoides = 40

    start = -0.03
    end = 0.03
    numOfSteps = (end - start) * 500000


    linspace = np.arange(start, end, 1/numOfSteps)

    plt.plot(linspace, x(linspace))
    plt.show()