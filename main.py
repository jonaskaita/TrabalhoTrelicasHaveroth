import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Algoritmos.Trelica.Ktotal import KTotal
from Algoritmos.Trelica.Kreduzida import KReduzida
from Algoritmos.MetodosNumericos.ElimGauss import eliminacao_gaussiana
from Algoritmos.MetodosNumericos.LUDecomp import fatoracao_LU, decomposicao_LU
from Algoritmos.MetodosNumericos.Jacobi import jacobi
from Algoritmos.MetodosNumericos.GaussSeidel import gauss_seidel
from Algoritmos.MetodosNumericos.Subs import substituicao_retroativa, substituicao_direta

# Definições iniciais
bars = []
connects = []

r = 2
alpha = 30

# Montar os nós
connects.append((0,0))
for theta in range(180, -1, -alpha):
    
    theta_rad = math.radians(theta)
    x = math.cos(theta_rad)*r
    y = math.sin(theta_rad)*r

    connects.append((x, y))

# Montar as arestas
for i in range(1, len(connects)):
    bars.append((0, i, r))

dist = 2*r*math.sin(math.radians(alpha/2))
for i in range(1, len(connects)-1):
    bars.append((i, i+1, dist))


# Calcular o Ktotal
Ktot = KTotal(bars, connects)
Kred, free = KReduzida(Ktot)

tempos = {"elim gauss": [], "fat lu out": [], "fat lu in": [], "jacobi": [], "gauss seidel": []}
iteracoes = {"jacobi": [], "gauss seidel": []}

L, U, P = decomposicao_LU(Kred)
for b in range (1, 101):
    F = 1000*b
    print("==============================================================")
    print(f"Força aplicada: {F}N")
    vec_forcas = np.array([
        0, -F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ], dtype=float)
    Fred = vec_forcas[free]
    ini = time.perf_counter()

    r = eliminacao_gaussiana(Kred, Fred)

    fim = time.perf_counter()
    tempos["elim gauss"].append(fim - ini)

    print("resultado da elim gauss: ", r)

    ini = time.perf_counter()
    
    b_perm = Fred[P]
    y = substituicao_direta(L, b_perm)
    x = substituicao_retroativa(U, y)
    fim = time.perf_counter()
    tempos["fat lu out"].append(fim - ini)

    print("resultado da fat lu out: ", x)

    ini = time.perf_counter()

    r = fatoracao_LU(Kred, Fred)

    fim = time.perf_counter()
    tempos["fat lu in"].append(fim - ini)

    print("resultado da fat lu in: ", r)

    ini = time.perf_counter()

    x_inicial = np.zeros(Kred.shape[0])
    r, k = jacobi(Kred, x_inicial, Fred)

    fim = time.perf_counter()
    tempos["jacobi"].append(fim - ini)
    iteracoes["jacobi"].append(k)

    print("resultado do jacobi: ", r)

    ini = time.perf_counter()

    x_inicial = np.zeros(Kred.shape[0])
    r, k = gauss_seidel(Kred, x_inicial, Fred)

    fim = time.perf_counter()
    tempos["gauss seidel"].append(fim - ini)
    iteracoes["gauss seidel"].append(k)

    print("resultado do gauss seidel: ", r)


for k, v in tempos.items():
    print(f"Tempo médio no {k} é {sum(v)/1000}")

print(f"Quantidade de iterações da eliminacao Gausiana: {Kred.shape[0] - 1}")
print(f"Quantidade de iterações da fatoracao Lu: {Kred.shape[0] - 1}")

for k, v in iteracoes.items():
    print(f"Quantidade média de iterações da {k}: {sum(v)/1000:0.2f}")