import numpy as np

# A - Matriz dos coeficientes
# x - Vetor de chutes iniciais
# b - Vetor com termos constantes
# m - Máx iter
# E - Tolerancia

def Jacobi(A, x, b, m, E):
    stop = 0
    n = len(b)
    k = 0

    while(stop == 0 and k < m):
        x_old = np.copy(x)

        for i in range(0, n):
            sum = 0
            for j in range(0, n):
                if(j != i):
                    sum += A[i][j] * x_old[j]

            if(A[i][i] == 0):
                raise ValueError("Elemento diagonal é nulo")

            x[i] = (b[i] - sum) / A[i][i]

        k += 1

        if(np.linalg.norm(x - x_old)/np.linalg.norm(x) < E):
            stop = 1

    return x    