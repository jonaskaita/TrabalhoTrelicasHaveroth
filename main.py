import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Definições iniciais
bars = []
connects = []

r = float(input())
alpha = 30

# Montar os nós
connects.append((0,0))
for theta in range(180, -1, -alpha):
    
    theta_rad = math.radians(theta)
    x = math.cos(theta_rad)*r
    y = math.sin(theta_rad)*r

    connects.append((x, y))

# Montar as arestas
for connect in connects[1:]:
    bars.append(((0, 0), connect))

for i in range(1, len(connects)-1):
    bars.append((connects[i], connects[i+1]))