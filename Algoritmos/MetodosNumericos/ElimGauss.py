import numpy as np
from Algoritmos.MetodosNumericos.Subs import substituicao_retroativa

def eliminacao_gaussiana(A, b, n):
    for k in range (0, n):
        pivot = abs(A[k][k])
        index = k
        
        for i in range(k+1, n):
            if(pivot < abs(A[i][k])):
                pivot = abs(A[i][k])
                index = i
        
        if pivot == 0:
            raise ValueError("Matriz singular")
        
        if k != index:
            A[[k, index]] = A[[index, k]]
            b[[k, index]] = b[[index, k]]
        
        for i in range(k+1, n):
            m = A[i][k] / A[k][k]
            A[i][k] = 0
            
            for j in range(k+1, n):
                A[i][j] = A[i][j] - m*A[k][j]
            
            b[i] = b[i] - m*b[k]

    x = substituicao_retroativa(A, b)
    return x