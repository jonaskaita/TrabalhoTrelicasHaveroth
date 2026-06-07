import numpy as np

def substituicao_retroativa(U, y):
    n = len(y)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        soma = 0
        for j in range(i + 1, n):
            soma += U[i][j] * x[j]

        x[i] = (y[i] - soma) / U[i][i]

    return x

def substituicao_direta(L, b):
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        soma = 0
        for j in range(i):
            soma += L[i][j] * y[j]

        y[i] = b[i] - soma

    return y
        