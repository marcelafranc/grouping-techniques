import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


# Carregar o dataset Wine
wine = load_wine()

# Exibicao da base de dados Wine
def printWine():
    
    X = wine.data
    y = wine.target
    df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
    

    # Scatter plot de duas features
    #sns.scatterplot(x=df_wine.iloc[:, 0], y=df_wine.iloc[:, 1])

    # Plotar grafico com corzinha
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.xlabel(wine.feature_names[0]) #tirar?
    plt.ylabel(wine.feature_names[1]) #tirar?
    plt.title("Visualização do Dataset Wine sem Agrupamento")
    plt.colorbar(label='Classe')
    plt.show()

# Agrupamneto Hierarquico do Wine
def hierarquicoWine():
    X = wine.data
    k = 3  # Sei que Wine tem 3 grupos

    # Preparar figuras para os dendrogramas e gráficos
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))  # DENDROGRAMA
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))  # GRAFICO DE DISPERSAO

    # Lista metodos linkage e titulos p o grafico
    linkages = ["ward", "complete", "average", "single"]
    titles = [
        "Método Ward",
        "Método Complete Linkage",
        "Método Average Linkage",
        "Método Single Linkage"
    ]

    # Aplicacao dos metodos linkage
    for i, linkage_method in enumerate(linkages):
        # Cálculo do linkage
        Z = linkage(X, method=linkage_method)
        
        # Gerar dendrograma
        dendrogram(Z, ax=axes1[i//2, i%2])
        axes1[i//2, i%2].set_title(f"Dendrograma - {titles[i]}")

        # Aplicacao do metodo de linkage!!!
        clustering = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
        y_hr = clustering.fit_predict(X)

        # Gerar grafico de dispersao
        axes2[i//2, i%2].scatter(X[:, 0], X[:, 1], c=y_hr, cmap="viridis", s=50, edgecolor='k')
        axes2[i//2, i%2].set_title(titles[i])

    # Ajusta layout
    fig1.tight_layout()
    fig2.tight_layout()

    # MOSTRAR (VAI GERAR DUAS JANELAS, UMA PRO GRAFICO E UMA PRO DENDROGAMA)
    plt.show()






#def particionalWine():
