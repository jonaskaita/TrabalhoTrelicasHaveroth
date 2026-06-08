import numpy as np

def deslocamento_no(connects, u_global):
    max_disp = 0

    for no in range(len(connects)):
        ux = u_global[2*no]
        uy = u_global[2*no + 1]

        disp = np.sqrt(ux**2 + uy**2)

        if disp > max_disp:
            max_disp = disp

    return max_disp