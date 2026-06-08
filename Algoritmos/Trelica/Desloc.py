import numpy as np

def deslocamento_max(connects, u_global):
    max_desloc = 0

    for no in range(len(connects)):
        ux = u_global[2*no]
        uy = u_global[2*no + 1]

        desloc = np.sqrt(ux**2 + uy**2)

        if desloc > max_desloc:
            max_desloc = desloc

    return max_desloc