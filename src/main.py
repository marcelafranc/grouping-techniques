import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import seaborn as sns
from iris import *
from wine import *

if __name__ == '__main__':
    escolhas = [1,2,3,4,5,6,7]

    while True:
        
        print("O que quer fazer?:")
        print("[1] Mostar Iris sem agrupamento")
        print("[2] Agrupar Iris (Hierarquico)")
        print("[3] Agrupar Iris (Particional)")
        print("[4] Mostar Wine sem agrupamento")
        print("[5] Agrupar Wine (Hierarquico)")
        print("[6] Agrupar Wine (Particional)")
        print("[7] Sair")
    
        resposta = int(input())

        if resposta not in escolhas: print("Escolha uma opcao valida")

        if resposta == 1: printIris() #iris sem agrupar
        if resposta == 2: hierarquicoIris() #iris hierarquico
        if resposta == 3: particionalIris() #iris particional
        if resposta == 4: printWine() #wine sem agrupar
        if resposta == 5: hierarquicoWine() # wine hierarquico
        if resposta == 6: particionalWine() #wine particional
        if resposta == 7: exit()
