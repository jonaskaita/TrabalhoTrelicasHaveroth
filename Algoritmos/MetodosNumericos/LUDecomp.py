import numpy as np
from Algoritmos.MetodosNumericos.Subs import substituicao_direta, substituicao_retroativa

def decomposicao_LU(A, n):
    L = np.eye(n)
    U = A.copy()
    P = np.arange(n)

    for k in range(n):
        pivot = abs(U[k][k])
        index = k

        for i in range(k + 1, n):
            if abs(U[i][k]) > pivot:
                pivot = abs(U[i][k])
                index = i

        if pivot == 0:
            raise ValueError("Matriz singular")

        if k != index:
            U[[k, index]] = U[[index, k]]

            if k > 0:
                L[[k, index], :k] = L[[index, k], :k]

            P[[k, index]] = P[[index, k]]

        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]

            for j in range(k, n):
                U[i][j] -= L[i][k] * U[k][j]

    return L, U, P

def fatoracao_LU(A, b):
    n = len(A)

    L, U, P = decomposicao_LU(A, n)
    b_perm = b[P]

    y = substituicao_direta(L, b_perm)
    x = substituicao_retroativa(U, y)

    return x