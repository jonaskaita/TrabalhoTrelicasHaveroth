import numpy as np

def KTotal(Kvector, bars, connects):
    n = len(connects)
    Ktot = np.zeros((2*n, 2*n))
    
    for b in range(len(bars)):
        k = Kvector[b]
        # Aqui pegamos cada barra da lista de barras,
        # cada K associado a uma barra
        # E pra cada barra, vamos pegar os indices dos
        # nós 1 e 2 que dizem respeito
        # aos indices 0, 1, 2, 3 da matriz K(e) respectivamente.
        # (Como cada nó tem 2 graus de liberdade, então para indexar
        # os graus de liberdade, temos que fazer 2*indice e 2*indice + 1)
        
        # Para cada grau de liberdade, somaremos seu valor no nó correspondente
        # em Ktotal, ou seja, para o indice 00, vamos pegar 2*no1,
        # e somar o K(e)[0][0] em Ktotal[2*no1][2*no1] e assim para cada
        # combinação de todos os indices dentro de cada K(e)
        
        no1, no2, _ = bars[b]
        
        graus_lib = [2*no1, 2*no1 + 1, 2*no2, 2*no2 + 1]
        
        for i_Ke in range(len(k)):
            for j_Ke in range(len(k)):
                Ktot[graus_lib[i_Ke]][graus_lib[j_Ke]] += k[i_Ke][j_Ke]
                
    return(Ktot)