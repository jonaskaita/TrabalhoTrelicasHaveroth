import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

print("Insira a quantidade de nós: ")
num_nodes = int(input())

bars = []
connects = []

for i in range(num_nodes):
    print(f"Insira coordenada x do nó {i + 1}:")
    x = float(input())
    print(f"Insira coordenada y do nó {i + 1}:")
    y = float(input())
    connects.append((x, y))

print("Insira a quantidade de barras: ")
num_bars = int(input())

for i in range (num_bars):
    print()
    print(f"Barra {i + 1}")
    print("Insira o número do nó 1 (1, 2, 3, ...): ")
    node1 = int(input()) - 1
    print("Insira o número do nó 2 (1, 2, 3, ...): ")
    node2 = int(input()) - 1
    bars.append((node1, node2))