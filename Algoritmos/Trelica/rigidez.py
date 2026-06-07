import numpy as np

def Rigidez(x1, y1, x2, y2, h, EA = 2*10**7):
    lx = (x2 - x1)/h
    ly = (y2 - y1)/h

    K = EA/h * np.array([[lx*lx, lx*ly, -lx*lx, -lx*ly],
                         [lx*ly, ly*ly, -lx*ly, -ly*ly],
                         [-lx*lx, -lx*ly, lx*lx, lx*ly],
                         [-lx*ly, -ly*ly, lx*ly, ly*ly]])
    
    return K