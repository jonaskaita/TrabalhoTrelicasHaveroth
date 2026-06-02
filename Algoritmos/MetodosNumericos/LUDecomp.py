import numpy as np


def LU_decomposition(A, n):
    L = np.eye(n)
    U = A
    
    for k in range(0, n-1):
        
        pivot = abs(U[k][k])
        index = k
        
        for i in range(k+1, n-1):
            if(pivot < abs(U[k][k])):
                pivot = abs(U[i][k])
                index = i
                
        if(pivot == 0):
            raise ValueError("[LU] A é singular (pivot = 0)")
        
        if(k != index):
            U[k][i] = U[index][i]
        
        for i in range(k+1, n-1):
            L[i][k] = U[i][k]/U[k][k]
            U[i][k] = 0
            
            for j in range(k+1, n-1):
                U[i][j] = U[i][j] - L[i][k]*U[k][j]
                
    return(L, U)