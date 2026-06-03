import numpy as np

def BackSubstitution(U, b, x, n):
    
    for i in range(n - 1 , -1, -1):
        soma = 0
        for j in range(i + 1, n):
            soma += U[i][j] * x[j]
            
        if(U[i][i] == 0):
            raise ValueError("Matriz é Singular (Uii = 0 detectado)")  
        
        x[i] = (b[i] - soma)/U[i][i]
        
    return(x)

def FrontSubstitution(L, y, b, n):
    
    for i in range(n):
        soma = 0
        for j in range(0, i):
            soma += L[i][j] * y[j]
        
        if(L[i][i] == 0):
            raise ValueError("Matriz é Singular (Lii = 0 detectado)")
        
        y[i] = (b[i] - soma)/L[i][i]
        
    return(y)
        