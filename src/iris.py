import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import MinMaxScaler

# Carregar o dataset Iris
iris = load_iris()

def printIris():
    # Carregar o dataset Iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Normalizar os dados
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # Atributos para o gráfico
    petal_length = X_normalized[:, 2]  # Petal Length
    petal_width = X_normalized[:, 3]   # Petal Width

    # Formas geométricas baseadas na espécie
    markers = ['o', '^', 's']  # 'o' para Setosa, '^' para Versicolor, 's' para Virginica

    # Plotando o gráfico sem agrupamento
    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.scatter(petal_width[y == i], petal_length[y == i], marker=markers[i], label=iris.target_names[i], edgecolor='k', alpha=0.7)

    plt.xlabel('Petal Width (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.title("Visualização do Dataset Iris sem Agrupamento")
    plt.legend(title='Species')
    plt.show()

# Função para determinar automaticamente o número de clusters
def determinar_n_clusters(X):
    Z = linkage(X)
    distancias = Z[:, 2]  # Coluna das distâncias das fusões
    dif_dist = np.diff(distancias)  # Diferença entre alturas consecutivas
    maior_salto = np.argmax(dif_dist)  # Índice do maior salto
    n_clusters = len(X) - maior_salto  # Número de clusters
    return n_clusters

# Função para plotar dendrograma
def plot_dendrogram(X):
    Z = linkage(X)
    plt.figure(figsize=(10, 6))
    dendrogram(Z)
    plt.title("Dendrograma")
    plt.xlabel("Amostras")
    plt.ylabel("Distância")
    plt.show()

# Agrupamento Hierárquico da Iris
def hierarquicoIris():
    X = iris.data

    # Normalizar os dados
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # Exibir dendrograma
    plot_dendrogram(X)
    
    # DETERMINA AUTOMATICAMENTE O NUMERO DE CLUSTERS
    n_clusters = determinar_n_clusters(X)
    print(f"Número de clusters sugerido: {n_clusters}")
    
    # Figuras grafico dispersao
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Lista metodos linkage e titulos
    linkages = ["ward", "complete", "average", "single"]
    titles = [
        "Método Ward",
        "Método Complete Linkage",
        "Método Average Linkage",
        "Método Single Linkage"
    ]

    # Aplicacao dos metodos linkage
    for i, linkage_method in enumerate(linkages):
        # Aplicacao AGLOMERATIVO
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        y_hr = clustering.fit_predict(X)

        # Grafico de dispersao
        axes[i//2, i%2].scatter(X[:, 0], X[:, 1], c=y_hr, cmap="viridis", s=50, edgecolor='k')
        axes[i//2, i%2].set_title(titles[i])

    # Ajusta layout
    fig.tight_layout()

    # Mostrar graficos
    plt.show()

