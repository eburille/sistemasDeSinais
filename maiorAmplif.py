
import matplotlib.pyplot as plt
import numpy as np

def H(w):
    # Retorna o ḿódulo da função de transferência
    return (100*np.sqrt(20**2 + w**2)) / (np.sqrt(w**2 + 150**2) * np.sqrt((w-1)**2 + 2**2) * np.sqrt((w+1)**2 + 2**2))

def dB(x):
    # Transforma o modulo da função de transferência de dB
    return 20*np.log10(x)


if __name__ == "__main__":
    start = -1
    end = 2

    linspace = np.linspace(start, end, 5000)
    freq =  0
    maior_modulo = 0
    for x in linspace:
        modulo = dB(H(x))
        if modulo > maior_modulo:
            maior_modulo = modulo
            freq = x
    print(maior_modulo, x)