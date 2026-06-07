import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Algoritmos.Trelica.Ktotal import KTotal
from Algoritmos.MetodosNumericos.ElimGauss import eliminacao_gaussiana
from Algoritmos.MetodosNumericos.LUDecomp import fatoracao_LU
from Algoritmos.MetodosNumericos.Jacobi import jacobi
from Algoritmos.MetodosNumericos.GaussSeidel import gauss_seidel

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

tempos = {"elim gauss": [], "fat lu": [], "jacobi": [], "gauss seidel": []}
iteracoes = {"jacobi": [], "gauss seidel": []}

for b in range (1, 101):
    F = 1000*b
    x_inicial = np.zeros(16)
    vec_forcas = np.array([
        0, 0, 0, 0, 0, 0, 0, -F, 0, 0, 0, 0, 0, 0, 0, 0
    ], dtype=float)

    ini = time.perf_counter()

    r = eliminacao_gaussiana(Ktot, vec_forcas)

    fim = time.perf_counter()
    tempos["elim gauss"].append(fim - ini)

    print("resultado da elim gauss: ", r)

    ini = time.perf_counter()

    r = fatoracao_LU(Ktot, vec_forcas)

    fim = time.perf_counter()
    tempos["fat lu"].append(fim - ini)

    print("resultado da fat lu: ", r)

    ini = time.perf_counter()

    r, k = jacobi(Ktot, x_inicial, vec_forcas)

    fim = time.perf_counter()
    tempos["jacobi"].append(fim - ini)
    iteracoes["jacobi"].append(k)

    print("resultado do jacobi: ", r)

    ini = time.perf_counter()

    r, k = gauss_seidel(Ktot, x_inicial, vec_forcas)

    fim = time.perf_counter()
    tempos["gauss seidel"].append(fim - ini)
    iteracoes["gauss seidel"].append(k)

    print("resultado do gauss seidel: ", r)


for k, v in tempos.items():
    print(f"Tempo médio no {k} é {sum(v)/1000}")

print(f"Quantidade de iterações da eliminacao Gausiana: {Ktot.shape[0] - 1}")
print(f"Quantidade de iterações da fatoracao Lu: {Ktot.shape[0] - 1}")

for k, v in iteracoes.items():
    print(f"Quantidade média de iterações da {k}: {sum(v)/1000:0.2f}")