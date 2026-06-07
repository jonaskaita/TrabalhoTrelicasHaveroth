import numpy as np
from Algoritmos.MetodosNumericos.Subs import BackSubstitution

# Este algoritmo de eliminação gaussiana utiliza de 
# pivotamento para minimizar problemas de divisão por 0

def Gaussian_Elimination(A, b, n):
    
    x = np.zeros(n)
    
    for k in range (0, n):
        pivot = abs(A[k][k])
        index = k
        
        for i in range(k+1, n):
            if(pivot < abs(A[i][k])):
                pivot = abs(A[i][k])
                index = i
        
        if(pivot == 0):
            raise ValueError("[EG] A é singular (pivot = 0)")
        
        if(k != index):
            A[[k, index]] = A[[index, k]]
            b[[k, index]] = b[[index, k]]
        
        for i in range(k+1, n):
            m = A[i][k] / A[k][k]
            A[i][k] = 0
            
            for j in range(k+1, n):
                A[i][j] = A[i][j] - m*A[k][j]
            
            b[i] = b[i] - m*b[k]
            
    return(BackSubstitution(A, b, x, n))