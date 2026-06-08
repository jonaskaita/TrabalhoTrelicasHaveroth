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

desloc_max = []
solucoes = []

ini_lu = time.perf_counter()
L, U, P = decomposicao_LU(Kred)
end_lu = time.perf_counter()
time_lu_out = end_lu - ini_lu

for b in range (1, 101):
    F = 1000*b
    print("==============================================================")
    print(f"Força aplicada: {F}N")
    vec_forcas = np.array([
        0, -F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ], dtype=float)
    Fred = vec_forcas[free]

    ###############
    # Elim. Gauss #
    ###############

    ini = time.perf_counter()

    r = eliminacao_gaussiana(Kred, Fred)

    fim = time.perf_counter()
    tempos["elim gauss"].append(fim - ini)

    print("resultado da elim gauss: ", r)
    
    u_global = np.zeros(Ktot.shape[0])
    for i, grau in enumerate(free):
        u_global[grau] = r[i]
    
    solucoes.append(u_global.copy())
    #desloc_max.append(deslocamento_no(connects, u_global))

    ###############
    # Fat. LU out #
    ###############

    ini = time.perf_counter()
    
    b_perm = Fred[P]
    y = substituicao_direta(L, b_perm)
    x = substituicao_retroativa(U, y)
    fim = time.perf_counter()
    tempos["fat lu out"].append(fim - ini)

    print("resultado da fat lu out: ", x)
    
    ##############
    # Fat. LU in #
    ##############

    ini = time.perf_counter()

    r = fatoracao_LU(Kred, Fred)

    fim = time.perf_counter()
    tempos["fat lu in"].append(fim - ini)

    print("resultado da fat lu in: ", r)

    ##########
    # Jacobi #
    ##########

    ini = time.perf_counter()

    x_inicial = np.array([10.0] * Kred.shape[0])
    r, k = jacobi(Kred, x_inicial, Fred)

    fim = time.perf_counter()
    tempos["jacobi"].append(fim - ini)
    iteracoes["jacobi"].append(k)

    print("resultado do jacobi: ", r)
    print("Número de iterações do jacobi: ", k)

    ################
    # Gauss Seidel #
    ################

    ini = time.perf_counter()

    x_inicial = np.array([10.0] * Kred.shape[0])
    r, k = gauss_seidel(Kred, x_inicial, Fred)

    fim = time.perf_counter()
    tempos["gauss seidel"].append(fim - ini)
    iteracoes["gauss seidel"].append(k)

    print("resultado do gauss seidel: ", r)
    print("Número de iterações do gauss seidel: ", k)

# Fazendo o gráfico de barras dos tempos, todos

labels = ["Elim. Gaussiana", 
          "Fatoração LU\nfatoração\núnica", 
          "Fatoração LU\nfatoração por iteração",
          "Jacobi", 
          "Gauss Seidel"
          ]
values = [sum(tempos["elim gauss"])/100.0, 
          (sum(tempos["fat lu out"]) + time_lu_out)/100.0, 
          sum(tempos["fat lu in"])/100.0, 
          sum(tempos["jacobi"])/100.0, 
          sum(tempos["gauss seidel"])/100.0
          ]
colors = ["red", "blue", "green", "orange", "purple"]

plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=colors)

plt.ylabel("Tempo médio (s)")

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.show()

# Apenas os métodos menores

labels = ["Elim. Gaussiana", 
          "Fatoração LU\nfatoração única", 
          "Fatoração LU\nfatoração por iteração", 
          ]
values = [sum(tempos["elim gauss"])/100.0, 
          (sum(tempos["fat lu out"]) + time_lu_out)/100.0, 
          sum(tempos["fat lu in"])/100.0, 
          ]
colors = ["red", "blue", "green", "orange", "purple"]

plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=colors)
plt.ylabel("Tempo médio (s)")

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.show()

###########################
# Fzendo o grafico das iterações
##############################

labels = [
          "Jacobi", 
          "Gauss Seidel"
          ]
values = [
          sum(iteracoes["jacobi"])/100.0, 
          sum(iteracoes["gauss seidel"])/100.0
          ]
colors = ["orange", "purple"]

plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=colors)

plt.ylabel("Média de iterações")

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.show()

#####################
# PARTE DA ANIMAÇÃO #
#####################
# Tentei fazer no notebook mas não consegui até o momento
# Parece que um ponto que deveria ser fixo ta se deslocando (o mais a direita)