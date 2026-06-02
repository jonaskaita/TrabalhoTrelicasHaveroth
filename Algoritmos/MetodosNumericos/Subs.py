import numpy as np

def BackSubstitution(U, b, x, n):
    
    for i in range(n - 1 , 0):
        sum = 0
        for j in range(i + 1, n - 1):
            sum += U[i][j] * x[j]
            
        if(U[i][i] == 0):
            raise ValueError("Matriz é Singular (Uii = 0 detectado)")  
        
        x[i] = (b[i] - sum)/U[i][i]
        
    return(x)

def FrontSubstitution(L, y, b, n):
    
    for i in range(n, 1):
        sum = 0
        for j in range(0, i - 1):
            sun += L[i][j] * y[j]
        
        if(L[i][i] == 0):
            raise ValueError("Matriz é Singular (Lii = 0 detectado)")
        
        y[i] = (b[i] - sum)/L[i][i]
        
    return(y)
        