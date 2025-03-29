import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import MinMaxScaler

# Carregar o dataset Iris
iris = load_iris()

# Exibicao da base de dados Iris
def printIris():
    X = iris.data
    y = iris.target
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Normalizar os dados
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # Plotar grafico com corzinha
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.xlabel(iris.feature_names[0]) #tirar?
    plt.ylabel(iris.feature_names[1]) #tirar?
    plt.title("Visualização do Dataset Iris sem Agrupamento")
    plt.colorbar(label='Classe')
    plt.show()

# Método do Cotovelo para determinar k
def metodo_cotovelo(X_normalized):
    inertias = []
    # Tentar diferentes números de clusters (de 1 a 10 clusters)
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_normalized)
        inertias.append(kmeans.inertia_)

    # Plotar o gráfico do cotovelo
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), inertias, marker='o')
    plt.title('Método do Cotovelo para Determinação de k')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inertia')
    plt.show()

# Agrupamento Hierarquico da Iris
def hierarquicoIris():
    X = iris.data

    # Normalizar os dados
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    #COTOVELO
    metodo_cotovelo(X_normalized)

    # QUANTOS CLUSTERS
    k = int(input("Digite o número de clusters (k): "))

    # Verificar se o valor de k é válido (maior ou igual a 1)
    while k < 1:
        print("Número de clusters inválido. O valor de k deve ser maior ou igual a 1.")
        k = int(input("Digite o número de clusters (k): "))

    # figuras pros grafico e dendro
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

    # MOSTRAR (VAI GERAR DUAS JANELAS, UMA PRO GRAFICO E UMA PRO DENDROGRAMA)
    plt.show()

# Agrupamento Particional da Iris
#def particionalWine():
