import numpy as np
import matplotlib.pyplot as plt

def metEuler(start, end, h, df, f0):
    numOfSteps = int((end-start)/h)
    linspace = np.linspace(start, end, numOfSteps)
    sol = np.array( [[0.0] * len(f0) for step in range(numOfSteps)] )

    pos = 0
    for eq in range(len(f0)):
        sol[pos][eq] = f0[eq]
        
    for x in range(len(linspace) - 1):
        nextPos = sol[pos] + h*df(sol[pos], linspace[pos])

        pos += 1
        sol[pos] = nextPos
    
    return (linspace, sol)

def x(t):
    return np.sin(0.5*t)

def dx(t):
    return 0.5*np.cos(0.5*t)

def df(sol, t):
    q = np.array([[0.0,1.0],[-2.0,-2.0]]) @ sol
    q2 = np.array([0.0,1.0])*x(t) + np.array([0.0,2.0]) * dx(t)
    return q + q2

def y(q):
    res = np.array([0.0]*len(q))
    for lin in range(len(q)):
        res[lin] = np.array([1, 0]) @ q[lin]
    return res

def yAnalitico(x):
    return np.exp(-x)*(-77/65 * np.cos(x) - 34/65 * np.sin(x)) + 12/65 * np.cos(x/2) + 44/65 * np.sin(x/2)
 
f0 = np.array([-1.0, 1.0])

start = 0
end = 8
h = 0.0001

linspace, sol = metEuler(start, end, h, df, f0)

plt.plot(linspace, y(sol), label = "sol numerico")
plt.plot(linspace, yAnalitico(linspace), label = "sol analitica")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()