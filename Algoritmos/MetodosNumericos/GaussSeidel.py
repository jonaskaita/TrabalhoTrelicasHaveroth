import numpy as np

# A - Matriz de coeficientes
# x - Vetor de chutes iniciais
# b - Vetor de termos constantes
# m - Máx iter
# E - Tolerancia

def GaussSeidel(A, x, b, m, E):
    n = len(b)
    k = 0
    
    while(k < m):
        x_old = np.copy(x)

        for i in range(0, n):
            sum = 0
            for j in range(0, n):
                sum +=  A[i][j] * x[j]
            for j in range(i, n):
                sum += A[i][j] * x_old[j]

            if(A[i][i] == 0):
                raise ValueError("Elemento diagonal é nulo")

            x[i] = (b[i] - sum) / A[i][i]

        if(np.linalg.norm(x - x_old)/np.linalg.norm(x) < E):
            break
        
        x_old = np.copy(x)
        k += 1

    return x
