import numpy as np

def criterio_sassenfield(A) :
    n = len(A)
    beta = np.zeros(n)

    for i in range(n):
        soma = 0.0

        for j in range(i):
            soma += abs(A[i][j]) * beta[j]

        for j in range(i + 1, n):
            soma += abs(A[i][j])

        if A[i][i] == 0:
            raise ValueError("Matriz singular")

        beta[i] = soma / abs(A[i][i])

    return np.max(beta) < 1

def gauss_seidel(A, x, b, m=1000, E=10e-16):
    if not criterio_sassenfield(A):
        print("Matriz não cumpre o criterio de Sassenfield")

    n = len(b)
    k = 0
    
    while k < m:
        x_old = np.copy(x)

        for i in range(n):
            sum = 0
            for j in range(i):
                sum +=  A[i][j] * x[j]
            for j in range(i+1, n):
                sum += A[i][j] * x_old[j]

            x[i] = (b[i] - sum)/A[i][i]

        k += 1

        if np.linalg.norm(x - x_old)/np.linalg.norm(x) < E:
            break
        
    return x, k
