import numpy as np

def Rigidez(x1, y1, x2, y2, EA = 2*10**7):
    h = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    lx = (x2 - x1)/h
    ly = (y2 - y1)/h

    K = EA/h * np.array([[lx*lx, lx*ly, -lx*lx, -lx*ly],
                         [lx*ly, ly*ly, -lx*ly, -ly*ly],
                         [-lx*lx, -lx*ly, lx*lx, lx*ly],
                         [-lx*ly, -ly*ly, lx*ly, ly*ly]])
    
    return K