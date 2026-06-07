import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Algoritmos.Trelica.Ktotal import KTotal

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

for b in range (1, 101):
    F = 1000*b

