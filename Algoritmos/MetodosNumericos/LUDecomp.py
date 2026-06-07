import numpy as np


def LU_decomposition(A, n):
    L = np.eye(n)
    U = A.copy()
    
    for k in range(0, n):
        
        pivot = abs(U[k][k])
        index = k
        
        for i in range(k+1, n):
            if(pivot < abs(U[i][k])):
                pivot = abs(U[i][k])
                index = i
                
        if(pivot == 0):
            raise ValueError("[LU] A é singular (pivot = 0)")
        
        if(k != index):
            U[[k, index]] = U[[index, k]]
        
        for i in range(k+1, n):
            L[i][k] = U[i][k]/U[k][k]
            U[i][k] = 0
            
            for j in range(k+1, n):
                U[i][j] = U[i][j] - L[i][k]*U[k][j]
                
    return(L, U)