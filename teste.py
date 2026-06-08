import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Algoritmos.Trelica.Ktotal import KTotal
from Algoritmos.Trelica.Kreduzida import KReduzida
from Algoritmos.MetodosNumericos.ElimGauss import eliminacao_gaussiana
from Algoritmos.MetodosNumericos.GaussSeidel import gauss_seidel
from Algoritmos.MetodosNumericos.Jacobi import jacobi
from Algoritmos.MetodosNumericos.LUDecomp import fatoracao_LU

#=======================================================#
#ESSE ARQUIVO FOI UTILIZADO PELO GRUPO COM O OBJETIVO
#DE COMPARAR NOSSAS IMPLEMENTAÇÕES COM O NUMPY (E O
#EXEMPLO DO ENUNCIADO DO PROJETO), COM O PROPÓSITO DE
#AFERIR A CORRETUDE DOS MÉTODOS IMPLEMENTADOS
#=======================================================#

A = np.array([
    [20., -1.,  2.,  0.,  1., -2.,  0.,  0.,  1., -1.],
    [-1., 21., -1.,  3.,  0.,  0.,  1., -1.,  0.,  2.],
    [ 2., -1., 22., -1.,  1.,  0., -2.,  0.,  1.,  0.],
    [ 0.,  3., -1., 23., -2.,  1.,  0.,  1., -1.,  0.],
    [ 1.,  0.,  1., -2., 24., -1.,  2.,  0.,  0., -1.],
    [-2., 0.,  0.,  1., -1., 25., -1.,  2.,  0.,  1.],
    [ 0.,  1., -2., 0.,  2., -1., 26., -1.,  1.,  0.],
    [ 0., -1., 0.,  1.,  0.,  2., -1., 27., -2.,  1.],
    [ 1.,  0.,  1., -1., 0.,  0.,  1., -2., 28., -1.],
    [-1., 2.,  0.,  0., -1., 1.,  0.,  1., -1., 29.]
])

b = np.array([
    15.,
    27.,
    18.,
    22.,
    35.,
    11.,
    19.,
    24.,
    31.,
    17.
])


max = int(1e4)
tol = 1e-6
print(f"Eliminação Gaussiana: \n{eliminacao_gaussiana(A, b)}")
x0 = np.zeros(A.shape[0])
print(f"Gauss-Seidel: \n{gauss_seidel(A, x0, b, max, tol)}")
x0 = np.zeros(A.shape[0])
print(f"Jacobi: {jacobi(A, x0, b, max, tol)}")
print(f"Fatoraçõa LU: \n{fatoracao_LU(A, b)}")

print(f"Numpy: \n{np.linalg.solve(A, b)}")

print("===========================================================================")


E = 200e9
A = 1e-4

connects = [
    (0, 0),  # nó 1
    (4, 0),  # nó 2
    (8, 0),  # nó 3
    (4, 3)   # nó 4
]

bars = [
    (0, 1, 4), # elemento 1
    (1, 2, 4), # elemento 2
    (0, 3, 5), # elemento 3
    (1, 3, 3), # elemento 4
    (2, 3, 5)  # elemento 5
]

M = KTotal(bars, connects)

print(M)

M_esperada = np.array([
 [189/500, 12/125, -1/4, 0, 0, 0, -16/125, -12/125],
 [12/125, 9/125, 0, 0, 0, 0, -12/125, -9/125],
 [-1/4, 0, 1/2, 0, -1/4, 0, 0, 0],
 [0, 0, 0, 1/3, 0, 0, 0, -1/3],
 [0, 0, -1/4, 0, 189/500, -12/125, -16/125, 12/125],
 [0, 0, 0, 0, -12/125, 9/125, 12/125, -9/125],
 [-16/125, -12/125, 0, 0, -16/125, 12/125, 32/125, 0],
 [-12/125, -9/125, 0, -1/3, 12/125, -9/125, 0, 179/375]
])

print(print(np.allclose(M, M_esperada*E*A)))

print("===========================================================================")

Mred, f = KReduzida(M, [0, 2], [], [1])

Mred_esperada = np.array([
    [1/2, 0, 0],
    [0, 32/125, 0],
    [0, 0, 179/375]
])

print(Mred)

print(Mred_esperada*E*A)

print(print(np.allclose(Mred, Mred_esperada*E*A)))

