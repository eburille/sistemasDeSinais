import numpy as np
import matplotlib.pyplot as plt

from toolBox import u
# Aplica o método de Euler para encontrar o próximo valor de um sistema baseado na derivada dos estados q
## start = Ponto de inicio do sistema
## end = Ponto final do sistema
## h = Tamanho do passo a ser utilizado
## dq = Função para calcular a derivada dq do estado do sistema
## y0 = Condições iniciais

# saídas => Retorna um array com o intervalo de tempo enviado como parametro t
#            e um array de mesmo tamanho com o estado do sistema para cada tempo t  
def metEuler(start, end, h, dq, y0):
    numOfSteps = int((end-start)/h)
    t = np.linspace(start, end, numOfSteps)
    q = np.array( [[0.0] * len(y0) for step in range(numOfSteps)] )

    n = 0
    # Seta as condições iniciais
    for eq in range(len(y0)):
        q[n][eq] = y0[eq]
    
    for i in range(0, len(t) - 1):
        nextPos = q[n] + h*dq(q[n], t[n])
         
        n += 1
        q[n] = nextPos

    return (t, q)

#Calcula a derivada do sistema de estados no ponto t baseado Nas matrizes A e B
def dq(q, t):
    
    # print(x(t)f)
    # sleep(0.5)f
    dq1 = A @ q
    dq2 = B * x(t) + B_ * dx(t)
    return dq1 + dq2

# Entrada do sistema
def x(t):
    return 1
# Derivada da entrada sistema
def dx(t):
    return 0 #np.cos(0.5*t)*0.5

# Calcula a resposta do sistema baseado no estado q
## Ajustar as matrizes C e D de acordo com o sistema 
def y(q):
    res = np.array([0.0]*len(q))
    for lin in range(len(q)):
        res[lin] = C @ q[lin] 
    return res

# Resposta analítica do sistema 
def yAnalitico(t):
    return 0

if __name__ == "__main__":


    k = 7.6545
    b1 = 0.4899
    b2 = 13.6842
    m = 10.582

    # Condições iniciais
          #        y(0) dy(0)
    y0 = np.array([0.0, 0.0])

    start = 0
    end = 8
    h = 0.0001

    A = np.array([[0.0, 1.0],
                  [-2*k/m, -(b1+b2)/m]])
    
    B = np.array([0.0, b2/m])
    
    B_ =  np.array([0.0, 0.0])

    C = np.array([1,0])
    D = np.array([0])

    t, q = metEuler(start, end, h, dq, y0)

    plt.plot(t, y(q), label = "sol numerico")


#    plt.plot(t, yAnalitico(t), label = "sol analitica")


#    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#    plt.grid()
#    plt.xlabel("t")
#    plt.ylabel("y(t)")
    plt.grid()
    plt.show()