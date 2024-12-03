
import matplotlib.pyplot as plt
import numpy as np

def H(w):
    return (100*np.sqrt(20**2 + w**2)) / (np.sqrt(w**2 + 150**2) * np.sqrt((w-1)**2 + 2**2) * np.sqrt((w+1)**2 + 2**2))

def fase(w):
    # Retorna a fase
    return ( -np.arctan(w/20) - np.arctan(w/150) - np.arctan((w-1)/2) - np.arctan((w+1)/2))*  360/(2*np.pi)

def w(x):
    # Retorna o eixo X do diagrama de bode
    return 10**x

def dB(x):
    # Transforma o modulo da função de transferência de dB
    return 20*np.log10(x)


if __name__ == "__main__":
    start = -3
    end = 6

    linspace = np.linspace(start, end, 5000)

    plt.plot(w(linspace), dB(H(w(linspace))))
    plt.xlabel("w")
    plt.ylabel("dB")
# 
    # plt.plot(w(linspace), (fase(w(linspace))))
    # plt.xlabel("w")
    # plt.ylabel("Graus")
    
    plt.xscale('log')
    plt.show()