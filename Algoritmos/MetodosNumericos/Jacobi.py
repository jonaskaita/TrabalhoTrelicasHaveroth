import numpy as np

def criterio_das_linhas(A) : 
    for i in range(A.shape[0]):
        soma = A[i, :].sum() - A[i, i]
        if soma >= A[i , i]:
            return False
        
    return True

def jacobi(A, x, b, m=1000, E=10e-16):
    if not criterio_das_linhas(A):
        print("Matriz não cumpre o critério das linhas")

    n = len(b)
    k = 0

    while k < m:
        x_old = np.copy(x)

        for i in range(0, n):
            sum = 0
            for j in range(0, n):
                if j != i:
                    sum += A[i][j] * x_old[j]

            if A[i][i] == 0:
                raise ValueError("Matriz singular")

            x[i] = (b[i] - sum)/A[i][i]

        k += 1

        if np.linalg.norm(x - x_old)/np.linalg.norm(x) < E:
            break

    return x, k