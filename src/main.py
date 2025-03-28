import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import seaborn as sns


wine = load_wine()
iris = load_iris()

def printWine():
    
    df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)

    # Scatter plot de duas features
    sns.scatterplot(x=df_wine.iloc[:, 0], y=df_wine.iloc[:, 1])
    plt.xlabel(wine.feature_names[0])
    plt.ylabel(wine.feature_names[1])
    plt.title("Visualização do Dataset Wine sem Agrupamento")
    plt.show()

def printIris():
    # Carregar o dataset Iris
    
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Scatter plot de duas features
    sns.scatterplot(x=df_iris.iloc[:, 0], y=df_iris.iloc[:, 1])
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title("Visualização do Dataset Iris sem Agrupamento")
    plt.show()


#def particionalIris():

#def particionalWine():

#def hierarquicoIris():

#def hierarquicoWine():

if __name__ == '__main__':
    escolhas = [1,2,3,4,5,6,7]

    while True:
        
        print("O que quer fazer?:")
        print("[1] Mostar Iris sem agrupamento")
        print("[2] Agrupar Iris (Particional)")
        print("[3] Agrupar Iris (Hierarquico)")
        print("[4] Mostar Wine sem agrupamento")
        print("[5] Agrupar Wine (Hierarquico)")
        print("[6] Agrupar Wine (Particional)")
        print("[7] Sair")
    
        resposta = int(input())

        if resposta not in escolhas: print("Escolha uma opcao valida")

        if resposta == 7: exit()

        if resposta == 1: printIris() #iris sem agrupar

        if resposta == 4: printWine() #wine sem agrupar

        #if resposta == 2 : #iris particinal

        #if resposta == 3: #iris hierarquico

        #if resposta == 5: #wine particional

        #if resposta == 6: #wine hierarquico


    
