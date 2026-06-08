import numpy as np

def KReduzida(Ktot, fixedxy=None, fixedx=None, fixedy=None):
    if fixedxy is None:
        fixedxy = [1]
    if fixedx is None:
        fixedx = []
    if fixedy is None:
        fixedy = [7]
    #Perceba que os paâmetros com valores padrão estão condizentes com a imagem de referência

    fixed = []
    n = Ktot.shape[0]

    #Primeiro, será calculado os graus de liberdade que correspondem aos pontos fixos
    #Para nós fixos em x e y
    for no in fixedxy:
        fixed.append(2*no)
        fixed.append(2*no + 1)
    #Para nós fixos apenas em x
    for no in fixedx:
        fixed.append(2*no)
    #Para nós fixos apenas em y
    for no in fixedy:
        fixed.append(2*no + 1)

    #Com base nisso, pegamos os graus de liberdade que não estão fixos
    free = [
        i for i in range(n)
        if i not in fixed
    ]

    #Agora, mantemos apenas as linha e colunas referentes aos graus de liberdade não fixos
    Kreduzida = Ktot[np.ix_(free, free)]

    return Kreduzida, free